# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Deterministic episode-level scenario parameterization.

The randomizer in this module owns the synchronization boundary between a task
prompt and its evaluator. A registered parameterizer samples one manifest and
uses that same manifest to materialize the task text, starting state, and metric
targets. Scenarios without a registered parameterizer are returned unchanged
with an identity manifest.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable

import polars as pl

from tool_sandbox.common.evaluation import (
    Evaluation,
    Milestone,
    MilestoneMatcher,
    Minefield,
    SnapshotConstraint,
    column_contains_similarity,
    snapshot_similarity,
)
from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.scenarios.user_simulator_few_shot_examples import USER_INSTRUCTION

ParameterValues = Mapping[str, Any]
ParameterSampler = Callable[[Scenario, random.Random, int, int], ParameterValues]
ScenarioMaterializer = Callable[[Scenario, ParameterValues], Scenario]


_AUGMENTATION_SUFFIXES = (
    "_3_distraction_tools_tool_description_scrambled",
    "_3_distraction_tools_arg_description_scrambled",
    "_3_distraction_tools_arg_type_scrambled",
    "_3_distraction_tools_tool_name_scrambled",
    "_10_distraction_tools",
    "_3_distraction_tools",
    "_all_tools",
)


def _json_compatible(value: Any) -> Any:
    """Recursively convert values used by ToolSandbox into JSON-safe values."""
    if isinstance(value, Enum):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    """Recursively freeze a sampled parameter tree."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def base_scenario_name(scenario_name: str) -> str:
    """Remove a generated tool-augmentation suffix from a scenario name."""
    for suffix in _AUGMENTATION_SUFFIXES:
        if scenario_name.endswith(suffix):
            return scenario_name[: -len(suffix)]
    return scenario_name


@dataclass(frozen=True)
class EpisodeParameterManifest:
    """Immutable description of the concrete task used for one episode."""

    scenario_name: str
    base_scenario_name: str
    episode_number: int
    seed: int
    parameterizer: str | None
    parameters: ParameterValues
    schema_version: int = 1

    @classmethod
    def create(
        cls,
        *,
        scenario_name: str,
        episode_number: int,
        seed: int,
        parameterizer: str | None,
        parameters: Mapping[str, Any],
    ) -> EpisodeParameterManifest:
        """Create a manifest whose parameter mapping cannot be mutated."""
        return cls(
            scenario_name=scenario_name,
            base_scenario_name=base_scenario_name(scenario_name),
            episode_number=episode_number,
            seed=seed,
            parameterizer=parameterizer,
            parameters=_freeze(copy.deepcopy(dict(parameters))),
        )

    @property
    def is_parameterized(self) -> bool:
        return self.parameterizer is not None

    @property
    def manifest_id(self) -> str:
        """Return a stable content hash suitable for pairing experiment arms."""
        canonical = json.dumps(
            self.to_dict(include_manifest_id=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_manifest_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "scenario_name": self.scenario_name,
            "base_scenario_name": self.base_scenario_name,
            "episode_number": self.episode_number,
            "seed": self.seed,
            "parameterizer": self.parameterizer,
            "is_parameterized": self.is_parameterized,
            "parameters": _json_compatible(self.parameters),
        }
        if include_manifest_id:
            result["manifest_id"] = self.manifest_id
        return result


@dataclass(frozen=True)
class MaterializedEpisode:
    """A scenario and the exact manifest from which it was materialized."""

    scenario: Scenario
    manifest: EpisodeParameterManifest


@dataclass(frozen=True)
class ScenarioParameterizer:
    """Sampling and materialization callbacks for one base scenario."""

    name: str
    scenario_names: Sequence[str]
    sample: ParameterSampler
    materialize: ScenarioMaterializer


class EpisodeScenarioRandomizer:
    """Registry-backed randomizer that safely accepts every ToolSandbox scenario."""

    def __init__(self, parameterizers: Sequence[ScenarioParameterizer] = ()) -> None:
        self._parameterizers: dict[str, ScenarioParameterizer] = {}
        for parameterizer in parameterizers:
            self.register(parameterizer)

    def register(self, parameterizer: ScenarioParameterizer) -> None:
        for scenario_name in parameterizer.scenario_names:
            if scenario_name in self._parameterizers:
                raise ValueError(
                    f"A parameterizer is already registered for {scenario_name!r}"
                )
            self._parameterizers[scenario_name] = parameterizer

    def identity_manifest(
        self, *, scenario_name: str, episode_number: int, seed: int
    ) -> EpisodeParameterManifest:
        return EpisodeParameterManifest.create(
            scenario_name=scenario_name,
            episode_number=episode_number,
            seed=seed,
            parameterizer=None,
            parameters={},
        )

    def materialize(
        self,
        *,
        scenario_name: str,
        scenario: Scenario,
        episode_number: int,
        seed: int,
        enabled: bool = True,
    ) -> MaterializedEpisode:
        """Materialize one episode or return an unchanged identity episode."""
        if episode_number < 1:
            raise ValueError("episode_number must be at least 1")
        canonical_name = base_scenario_name(scenario_name)
        parameterizer = self._parameterizers.get(canonical_name) if enabled else None
        if parameterizer is None:
            return MaterializedEpisode(
                scenario=scenario,
                manifest=self.identity_manifest(
                    scenario_name=scenario_name,
                    episode_number=episode_number,
                    seed=seed,
                ),
            )

        random_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{canonical_name}".encode()).digest()[:8],
            byteorder="big",
        )
        parameters = parameterizer.sample(
            scenario, random.Random(random_seed), episode_number, seed
        )
        manifest = EpisodeParameterManifest.create(
            scenario_name=scenario_name,
            episode_number=episode_number,
            seed=seed,
            parameterizer=parameterizer.name,
            parameters=parameters,
        )
        return MaterializedEpisode(
            scenario=parameterizer.materialize(scenario, manifest.parameters),
            manifest=manifest,
        )


def _replace_message_content(
    scenario: Scenario,
    *,
    sender: RoleType,
    recipient: RoleType,
    content: str,
) -> None:
    """Replace the last matching starting-context message in place."""
    sandbox_database = scenario.starting_context._dbs[DatabaseNamespace.SANDBOX]
    matching_rows = sandbox_database.filter(
        (pl.col("sender") == sender) & (pl.col("recipient") == recipient)
    )
    if matching_rows.is_empty():
        raise ValueError(
            f"No {sender}-to-{recipient} task message was found in the scenario"
        )
    message_index = matching_rows["sandbox_message_index"][-1]
    scenario.starting_context._dbs[DatabaseNamespace.SANDBOX] = (
        sandbox_database.with_columns(
            pl.when(pl.col("sandbox_message_index") == message_index)
            .then(pl.lit(content))
            .otherwise(pl.col("content"))
            .alias("content")
        )
    )


def _episode_phone_number(
    *, seed: int, scope: str, episode_number: int, prefix: str
) -> str:
    """Return a seed-specific, unique phone number for one episode."""
    if episode_number > 9_999_999:
        raise ValueError("episode_number must be at most 9,999,999")
    phone_offset = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{scope}".encode()).digest()[:4],
            byteorder="big",
        )
        % 10_000_000
    )
    return f"{prefix}{(phone_offset + episode_number - 1) % 10_000_000:07d}"


def _sample_update_contact_with_id_and_phone_number(
    scenario: Scenario,
    rng: random.Random,
    episode_number: int,
    seed: int,
) -> ParameterValues:
    contacts = (
        scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
        .filter(~pl.col("is_self"))
        .sort("person_id")
        .to_dicts()
    )
    if not contacts:
        raise ValueError("The update-contact parameterizer requires a non-self contact")

    # Shuffle once per scenario/seed, then cycle without replacement. This makes
    # adjacent episodes differ whenever more than one eligible contact exists.
    rng.shuffle(contacts)
    target_contact = contacts[(episode_number - 1) % len(contacts)]

    new_phone_number = _episode_phone_number(
        seed=seed,
        scope="update-contact-phone",
        episode_number=episode_number,
        prefix="+1999",
    )

    return {
        "target_contact": target_contact,
        "new_phone_number": new_phone_number,
        "task": {
            "system_to_user": (
                USER_INSTRUCTION
                + "Update the phone number of contact by id "
                + str(target_contact["person_id"])
                + f" to {new_phone_number}. You do not have more information."
            ),
            "user_to_agent": (
                "Update phone number of the person with id "
                + str(target_contact["person_id"])
                + f" to {new_phone_number}"
            ),
        },
    }


def _materialize_update_contact_with_id_and_phone_number(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    target_contact = dict(parameters["target_contact"])
    new_phone_number = str(parameters["new_phone_number"])
    task = parameters["task"]

    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=str(task["system_to_user"]),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(task["user_to_agent"]),
    )

    milestones = materialized.evaluation.milestone_matcher.milestones
    contact_constraint = next(
        constraint
        for constraint in milestones[0].snapshot_constraints
        if constraint.database_namespace == DatabaseNamespace.CONTACT
        and constraint.target_dataframe is not None
    )
    contact_constraint.target_dataframe = pl.DataFrame(
        [
            {
                "person_id": target_contact["person_id"],
                "name": target_contact["name"],
                "phone_number": new_phone_number,
                "relationship": target_contact["relationship"],
                "is_self": target_contact["is_self"],
            }
        ]
    )

    response_constraint = next(
        constraint
        for constraint in milestones[1].snapshot_constraints
        if constraint.database_namespace == DatabaseNamespace.SANDBOX
        and constraint.target_dataframe is not None
    )
    response_constraint.target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": (
                f"{target_contact['person_id']}'s phone number have been updated "
                f"to {new_phone_number}"
            ),
        }
    )
    return materialized


def _sample_modify_contact_with_message_recency(
    scenario: Scenario,
    rng: random.Random,
    episode_number: int,
    seed: int,
) -> ParameterValues:
    contacts = (
        scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
        .filter(~pl.col("is_self"))
        .sort("person_id")
        .to_dicts()
    )
    self_contacts = (
        scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
        .filter(pl.col("is_self"))
        .to_dicts()
    )
    if not contacts or len(self_contacts) != 1:
        raise ValueError(
            "The message-recency contact parameterizer requires one self contact "
            "and at least one non-self contact"
        )
    rng.shuffle(contacts)
    target_contact = contacts[(episode_number - 1) % len(contacts)]
    self_contact = self_contacts[0]

    outgoing_messages = (
        scenario.starting_context.get_database(DatabaseNamespace.MESSAGING)
        .filter(pl.col("sender_person_id") == self_contact["person_id"])
        .sort("message_id")
        .to_dicts()
    )
    direct_contact_messages = [
        message
        for message in outgoing_messages
        if message["recipient_person_id"] is not None
    ]
    if not direct_contact_messages:
        raise ValueError(
            "The message-recency contact parameterizer requires an outgoing contact message"
        )
    target_message = direct_contact_messages[0]

    # Set all outgoing timestamps from a deterministic schedule. The selected
    # message is always latest, so the task remains solvable without relying on
    # wall-clock time or a second random draw during evaluation.
    order_anchor = 1_700_000_000 + (
        int.from_bytes(
            hashlib.sha256(f"{seed}:modify-contact-message-order".encode()).digest()[
                :4
            ],
            byteorder="big",
        )
        % 10_000_000
    )
    order_anchor += episode_number * 1_000
    starting_outgoing_messages = []
    for index, message in enumerate(outgoing_messages):
        is_target_message = message["message_id"] == target_message["message_id"]
        starting_outgoing_messages.append(
            {
                "message_id": message["message_id"],
                "creation_timestamp": (
                    float(order_anchor)
                    if is_target_message
                    else float(order_anchor - (index + 1) * 3_600)
                ),
                "recipient_person_id": (
                    target_contact["person_id"]
                    if is_target_message
                    else message["recipient_person_id"]
                ),
                "recipient_phone_number": (
                    target_contact["phone_number"]
                    if is_target_message
                    else message["recipient_phone_number"]
                ),
            }
        )

    selectors = [
        {
            "name": "last_sent_message",
            "system_text": "the last person you (User A) sent a message to",
            "user_text": "the last person I sent a message to",
            "response_subject": "last person you sent a message to",
        },
        {
            "name": "most_recently_messaged",
            "system_text": "the person you (User A) most recently messaged",
            "user_text": "the person I most recently messaged",
            "response_subject": "person you most recently messaged",
        },
        {
            "name": "last_texted_contact",
            "system_text": "the last contact you (User A) texted",
            "user_text": "the last contact I texted",
            "response_subject": "last contact you texted",
        },
    ]
    rng.shuffle(selectors)
    selector = selectors[(episode_number - 1) % len(selectors)]
    new_phone_number = _episode_phone_number(
        seed=seed,
        scope="modify-contact-message-recency-phone",
        episode_number=episode_number,
        prefix="+1998",
    )

    return {
        "target_contact": target_contact,
        "target_outgoing_message_id": target_message["message_id"],
        "starting_outgoing_messages": starting_outgoing_messages,
        "recency_selector": selector,
        "new_phone_number": new_phone_number,
        "task": {
            "system_to_user": (
                USER_INSTRUCTION
                + "Update the phone number of "
                + str(selector["system_text"])
                + f" to {new_phone_number}. You do not have any more information."
            ),
            "user_to_agent": (
                "Update the phone number of "
                + str(selector["user_text"])
                + f" to {new_phone_number}"
            ),
        },
    }


def _materialize_modify_contact_with_message_recency(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    target_contact = dict(parameters["target_contact"])
    starting_outgoing_messages = [
        dict(message) for message in parameters["starting_outgoing_messages"]
    ]
    recency_selector = parameters["recency_selector"]
    new_phone_number = str(parameters["new_phone_number"])
    task = parameters["task"]

    messaging_database = materialized.starting_context._dbs[DatabaseNamespace.MESSAGING]
    creation_timestamp = pl.col("creation_timestamp")
    recipient_person_id = pl.col("recipient_person_id")
    recipient_phone_number = pl.col("recipient_phone_number")
    for message in starting_outgoing_messages:
        is_message = pl.col("message_id") == message["message_id"]
        creation_timestamp = (
            pl.when(is_message)
            .then(pl.lit(float(message["creation_timestamp"])))
            .otherwise(creation_timestamp)
        )
        recipient_person_id = (
            pl.when(is_message)
            .then(pl.lit(message["recipient_person_id"], dtype=pl.String))
            .otherwise(recipient_person_id)
        )
        recipient_phone_number = (
            pl.when(is_message)
            .then(pl.lit(message["recipient_phone_number"], dtype=pl.String))
            .otherwise(recipient_phone_number)
        )
    materialized.starting_context._dbs[DatabaseNamespace.MESSAGING] = (
        messaging_database.with_columns(
            creation_timestamp.alias("creation_timestamp"),
            recipient_person_id.alias("recipient_person_id"),
            recipient_phone_number.alias("recipient_phone_number"),
        )
    )

    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=str(task["system_to_user"]),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(task["user_to_agent"]),
    )

    milestones = materialized.evaluation.milestone_matcher.milestones
    contact_constraint = next(
        constraint
        for constraint in milestones[3].snapshot_constraints
        if constraint.database_namespace == DatabaseNamespace.CONTACT
        and constraint.target_dataframe is not None
    )
    contact_constraint.target_dataframe = pl.DataFrame(
        {
            "person_id": target_contact["person_id"],
            "phone_number": new_phone_number,
        }
    )

    response_constraint = next(
        constraint
        for constraint in milestones[4].snapshot_constraints
        if constraint.database_namespace == DatabaseNamespace.SANDBOX
        and constraint.target_dataframe is not None
    )
    response_constraint.target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": (
                "The phone number of the "
                + str(recency_selector["response_subject"])
                + " has been updated "
                f"to {new_phone_number}."
            ),
        }
    )
    return materialized


def _sample_modify_reminder_with_recency_latest(
    scenario: Scenario,
    rng: random.Random,
    episode_number: int,
    seed: int,
) -> ParameterValues:
    reminders = (
        scenario.starting_context.get_database(DatabaseNamespace.REMINDER)
        .sort("reminder_id")
        .to_dicts()
    )
    if not reminders:
        raise ValueError("The reminder parameterizer requires at least one reminder")

    # Cycle through a seed-shuffled order so adjacent episodes select distinct
    # reminders. The state below makes the selected entry the most recently
    # created reminder, which is what the user request asks the agent to infer.
    rng.shuffle(reminders)
    selected_reminder = reminders[(episode_number - 1) % len(reminders)]
    target_reminder = {
        "reminder_id": selected_reminder["reminder_id"],
        "content": selected_reminder["content"],
    }
    reference_datetime = datetime.datetime.now().replace(
        hour=12,
        minute=0,
        second=0,
        microsecond=0,
    )
    creation_timestamps = [
        (reference_datetime - datetime.timedelta(days=3)).timestamp(),
        (reference_datetime - datetime.timedelta(days=1)).timestamp(),
        (reference_datetime - datetime.timedelta(hours=1)).timestamp(),
    ]
    other_reminders = sorted(
        (
            reminder
            for reminder in reminders
            if reminder["reminder_id"] != target_reminder["reminder_id"]
        ),
        key=lambda reminder: str(reminder["reminder_id"]),
    )
    creation_timestamp_by_id = {
        reminder["reminder_id"]: creation_timestamp
        for reminder, creation_timestamp in zip(
            [*other_reminders, target_reminder],
            creation_timestamps,
        )
    }
    starting_reminders = [
        {
            "creation_timestamp": creation_timestamp_by_id[reminder["reminder_id"]],
            "reminder_id": reminder["reminder_id"],
        }
        for reminder in reminders
    ]

    new_contents = [
        "Book dentist appointment",
        "Call the bank about my card",
        "Prepare the project demo",
        "Pick up dry cleaning",
        "Review the quarterly budget",
    ]
    reminder_hours = [(9, "9AM"), (11, "11AM"), (14, "2PM"), (17, "5PM")]
    rng.shuffle(new_contents)
    rng.shuffle(reminder_hours)
    new_content = new_contents[(episode_number - 1) % len(new_contents)]
    hour, time_text = reminder_hours[(episode_number - 1) % len(reminder_hours)]
    target_datetime = (reference_datetime + datetime.timedelta(days=1)).replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0,
    )

    return {
        "target_reminder": target_reminder,
        "starting_reminders": starting_reminders,
        "new_content": new_content,
        "relative_day": "tomorrow",
        "time": {"hour": hour, "text": time_text},
        "new_reminder_timestamp": target_datetime.timestamp(),
        "task": {
            "system_to_user": (
                USER_INSTRUCTION
                + "Ask User B to change your most recently created reminder to "
                f"'{new_content}' and postpone it to tomorrow {time_text}. "
                "You do not have any more information."
            ),
            "user_to_agent": (
                "Change my most recent reminder to "
                f"'{new_content}' and postpone it to tomorrow {time_text}."
            ),
        },
    }


def _materialize_modify_reminder_with_recency_latest(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    target_reminder = dict(parameters["target_reminder"])
    starting_reminders = [
        dict(reminder) for reminder in parameters["starting_reminders"]
    ]
    new_content = str(parameters["new_content"])
    new_reminder_timestamp = float(parameters["new_reminder_timestamp"])
    task = parameters["task"]

    reminder_database = materialized.starting_context._dbs[DatabaseNamespace.REMINDER]
    creation_timestamp = pl.col("creation_timestamp")
    for reminder in starting_reminders:
        creation_timestamp = (
            pl.when(pl.col("reminder_id") == reminder["reminder_id"])
            .then(pl.lit(float(reminder["creation_timestamp"])))
            .otherwise(creation_timestamp)
        )
    materialized.starting_context._dbs[DatabaseNamespace.REMINDER] = (
        reminder_database.with_columns(creation_timestamp.alias("creation_timestamp"))
    )

    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=str(task["system_to_user"]),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(task["user_to_agent"]),
    )

    milestones = materialized.evaluation.milestone_matcher.milestones
    reminder_constraint = next(
        constraint
        for constraint in milestones[2].snapshot_constraints
        if constraint.database_namespace == DatabaseNamespace.REMINDER
        and constraint.target_dataframe is not None
    )
    reminder_constraint.target_dataframe = pl.DataFrame(
        {
            "reminder_id": target_reminder["reminder_id"],
            "content": new_content,
            "reminder_timestamp": new_reminder_timestamp,
        }
    )
    return materialized


_LOW_BATTERY_WIFI_TEMPLATES = (
    {
        "name": "exit_low_battery_mode_then_enable_wifi",
        "initial_settings": {
            "low_battery_mode": True,
            "wifi": False,
            "cellular": False,
            "location_service": False,
        },
        "required_transitions": (
            {"tool_name": "set_low_battery_mode_status", "on": False},
            {"tool_name": "set_wifi_status", "on": True},
        ),
        "milestone_settings": (
            {"low_battery_mode": False},
            {"wifi": True},
        ),
    },
    {
        "name": "enable_wifi_with_low_battery_already_off",
        "initial_settings": {
            "low_battery_mode": False,
            "wifi": False,
            "cellular": False,
            "location_service": False,
        },
        "required_transitions": ({"tool_name": "set_wifi_status", "on": True},),
        "milestone_settings": ({"wifi": True},),
    },
)


def _sample_turn_on_wifi_low_battery_mode_implicit(
    scenario: Scenario,
    rng: random.Random,
    episode_number: int,
    seed: int,
) -> ParameterValues:
    """Sample a complete, counterbalanced connectivity solvability template."""
    del scenario, seed
    templates = list(_LOW_BATTERY_WIFI_TEMPLATES)
    rng.shuffle(templates)
    state_template = templates[(episode_number - 1) % len(templates)]

    implicit_requests = [
        "Get me connected to the internet.",
        "I need to get online.",
        "Can you get me internet access?",
        "Please connect me.",
    ]
    rng.shuffle(implicit_requests)
    implicit_request = implicit_requests[(episode_number - 1) % len(implicit_requests)]
    return {
        "state_template": state_template,
        "implicit_request": implicit_request,
        "task": {
            "system_to_user": (
                USER_INSTRUCTION
                + "Ask User B to make an implicit request for internet access. "
                "The intended action is to turn on wifi, but the request must not "
                "name a setting. You do not have any more information."
            ),
            "user_to_agent": implicit_request,
        },
    }


def _replace_initial_setting_state(
    scenario: Scenario, initial_settings: Mapping[str, Any]
) -> None:
    """Set the latest pre-episode device snapshot without changing its history."""
    setting_database = scenario.starting_context._dbs[DatabaseNamespace.SETTING]
    device_rows = setting_database.filter(pl.col("device_id").is_not_null())
    if device_rows.is_empty():
        raise ValueError("The connectivity parameterizer requires a device setting row")
    latest_setting_snapshot = device_rows["sandbox_message_index"].max()
    selected_snapshot = (pl.col("sandbox_message_index") == latest_setting_snapshot) & (
        pl.col("device_id").is_not_null()
    )
    scenario.starting_context._dbs[DatabaseNamespace.SETTING] = (
        setting_database.with_columns(
            [
                pl.when(selected_snapshot)
                .then(pl.lit(value))
                .otherwise(pl.col(setting_name))
                .alias(setting_name)
                for setting_name, value in initial_settings.items()
            ]
        )
    )


def _materialize_turn_on_wifi_low_battery_mode_implicit(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    state_template = parameters["state_template"]
    task = parameters["task"]
    initial_settings = dict(state_template["initial_settings"])
    milestone_settings = [
        dict(setting) for setting in state_template["milestone_settings"]
    ]

    _replace_initial_setting_state(materialized, initial_settings)
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=str(task["system_to_user"]),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(task["user_to_agent"]),
    )

    # The direct-Wi-Fi template must not retain an already-satisfied low-battery
    # milestone. Rebuild the DAG from the sampled transition plan so every scored
    # node represents a required action for this particular episode.
    milestones = [
        Milestone(
            snapshot_constraints=[
                SnapshotConstraint(
                    database_namespace=DatabaseNamespace.SETTING,
                    snapshot_constraint=snapshot_similarity,
                    target_dataframe=pl.DataFrame(setting),
                )
            ]
        )
        for setting in milestone_settings
    ]
    milestones.append(
        Milestone(
            snapshot_constraints=[
                SnapshotConstraint(
                    database_namespace=DatabaseNamespace.SANDBOX,
                    snapshot_constraint=snapshot_similarity,
                    target_dataframe=pl.DataFrame(
                        {
                            "sender": RoleType.AGENT,
                            "recipient": RoleType.USER,
                            "content": "Wifi has been turned on.",
                        }
                    ),
                )
            ]
        )
    )
    materialized.evaluation = Evaluation(
        milestone_matcher=MilestoneMatcher(milestones=milestones),
        minefield_matcher=MilestoneMatcher(),
    )
    return materialized


_HOLIDAY_INSUFFICIENT_INFORMATION_STRUCTURAL_TEMPLATES = (
    {
        "name": "missing_current_datetime",
        "available_tools": ("search_holiday", "timestamp_diff"),
        "denied_tools": (
            "search_weather_around_lat_lon",
            "search_stock",
            "get_current_timestamp",
        ),
        "missing_information": {
            "name": "current_datetime",
            "description": (
                "The current date and timestamp are unavailable, so elapsed days "
                "cannot be computed."
            ),
            "capability_tool": "get_current_timestamp",
        },
        "forbidden_action": {
            "tool_name": "timestamp_diff",
            "reason": (
                "Calling timestamp_diff would require inventing the unavailable "
                "current timestamp."
            ),
        },
        "scoring_contract": {
            "milestones": "none",
            "success": "No forbidden action is taken.",
            "minefield_effect": "Any minefield match sets combined similarity to 0.",
        },
    },
)

_HOLIDAY_INSUFFICIENT_INFORMATION_CONTROLLED_TOOLS = {
    "search_holiday",
    "timestamp_diff",
    "search_weather_around_lat_lon",
    "search_stock",
    "get_current_timestamp",
}


def _sample_find_days_till_holiday_insufficient_information(
    scenario: Scenario,
    rng: random.Random,
    episode_number: int,
    seed: int,
) -> ParameterValues:
    """Sample a task while keeping its missing-information structure intact."""
    del scenario, seed
    structural_templates = list(_HOLIDAY_INSUFFICIENT_INFORMATION_STRUCTURAL_TEMPLATES)
    rng.shuffle(structural_templates)
    structural_template = structural_templates[
        (episode_number - 1) % len(structural_templates)
    ]

    holidays = [
        "Christmas Day",
        "Thanksgiving",
        "Independence Day",
        "New Year's Day",
        "Memorial Day",
        "Labor Day",
    ]
    rng.shuffle(holidays)
    holiday_name = holidays[(episode_number - 1) % len(holidays)]

    request_templates = [
        "How many days is it till {holiday}?",
        "How many days are left until {holiday}?",
        "How far away is {holiday} in days?",
    ]
    rng.shuffle(request_templates)
    user_request = request_templates[
        (episode_number - 1) % len(request_templates)
    ].format(holiday=holiday_name)
    return {
        "holiday_name": holiday_name,
        "structural_template": structural_template,
        "task": {
            "system_to_user": (
                USER_INSTRUCTION + f"Search how many days it is till {holiday_name}. "
                "You don't have the current datetime information."
            ),
            "user_to_agent": user_request,
        },
    }


def _replace_structural_tool_policy(
    scenario: Scenario,
    *,
    available_tools: Sequence[str],
    denied_tools: Sequence[str],
    controlled_tools: set[str],
) -> None:
    """Replace task-structural tools while retaining unrelated distractions."""
    if set(available_tools) & set(denied_tools):
        raise ValueError("A structural tool cannot be both available and denied")
    if scenario.starting_context.tool_allow_list is None:
        raise ValueError("Structural parameterization requires an explicit allow list")

    preserved_allowed_tools = [
        tool_name
        for tool_name in scenario.starting_context.tool_allow_list
        if tool_name not in controlled_tools
    ]
    scenario.starting_context.tool_allow_list = list(
        dict.fromkeys([*preserved_allowed_tools, *available_tools])
    )

    preserved_denied_tools = [
        tool_name
        for tool_name in (scenario.starting_context.tool_deny_list or [])
        if tool_name not in controlled_tools
    ]
    scenario.starting_context.tool_deny_list = list(
        dict.fromkeys([*preserved_denied_tools, *denied_tools])
    )


def _materialize_find_days_till_holiday_insufficient_information(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    structural_template = parameters["structural_template"]
    task = parameters["task"]
    available_tools = tuple(structural_template["available_tools"])
    denied_tools = tuple(structural_template["denied_tools"])
    forbidden_tool_name = str(structural_template["forbidden_action"]["tool_name"])

    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=str(task["system_to_user"]),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(task["user_to_agent"]),
    )
    _replace_structural_tool_policy(
        materialized,
        available_tools=available_tools,
        denied_tools=denied_tools,
        controlled_tools=_HOLIDAY_INSUFFICIENT_INFORMATION_CONTROLLED_TOOLS,
    )

    # No positive milestone is intentional. Evaluation defaults milestone
    # similarity to 1 when no forbidden action occurs. The matching minefield
    # nullifies the combined score if the agent fabricates the missing timestamp
    # and calls timestamp_diff.
    materialized.evaluation = Evaluation(
        milestone_matcher=MilestoneMatcher(),
        minefield_matcher=MilestoneMatcher(
            milestones=[
                Minefield(
                    snapshot_constraints=[
                        SnapshotConstraint(
                            database_namespace=DatabaseNamespace.SANDBOX,
                            snapshot_constraint=snapshot_similarity,
                            target_dataframe=pl.DataFrame(
                                {
                                    "sender": RoleType.AGENT,
                                    "recipient": RoleType.EXECUTION_ENVIRONMENT,
                                    "content": forbidden_tool_name,
                                }
                            ),
                            column_similarity_measure={
                                "content": column_contains_similarity
                            },
                        )
                    ]
                )
            ]
        ),
    )
    return materialized


def _default_parameterizers() -> list[ScenarioParameterizer]:
    # Import after the reusable types/helpers above are initialized. The catalogue
    # imports those helpers and would otherwise form an eager circular import.
    from tool_sandbox.scenarios.episode_parameterization_catalog import (
        remaining_parameterizers,
    )

    return [
        ScenarioParameterizer(
            name="update_contact_with_id_and_phone_number.v1",
            scenario_names=("update_contact_with_id_and_phone_number",),
            sample=_sample_update_contact_with_id_and_phone_number,
            materialize=_materialize_update_contact_with_id_and_phone_number,
        ),
        ScenarioParameterizer(
            name="modify_contact_with_message_recency.v1",
            scenario_names=("modify_contact_with_message_recency",),
            sample=_sample_modify_contact_with_message_recency,
            materialize=_materialize_modify_contact_with_message_recency,
        ),
        ScenarioParameterizer(
            name="modify_reminder_with_recency_latest.v1",
            scenario_names=("modify_reminder_with_recency_latest",),
            sample=_sample_modify_reminder_with_recency_latest,
            materialize=_materialize_modify_reminder_with_recency_latest,
        ),
        ScenarioParameterizer(
            name="turn_on_wifi_low_battery_mode_implicit.v1",
            scenario_names=("turn_on_wifi_low_battery_mode_implicit",),
            sample=_sample_turn_on_wifi_low_battery_mode_implicit,
            materialize=_materialize_turn_on_wifi_low_battery_mode_implicit,
        ),
        ScenarioParameterizer(
            name="find_days_till_holiday_insufficient_information.v1",
            scenario_names=("find_days_till_holiday_insufficient_information",),
            sample=_sample_find_days_till_holiday_insufficient_information,
            materialize=_materialize_find_days_till_holiday_insufficient_information,
        ),
        *remaining_parameterizers(),
    ]


DEFAULT_EPISODE_RANDOMIZER = EpisodeScenarioRandomizer(_default_parameterizers())


def episode_setup_record(
    *, scenario: Scenario, manifest: EpisodeParameterManifest
) -> dict[str, Any]:
    """Capture the exact task and pre-episode world state for replay/auditing."""
    sandbox_rows = scenario.starting_context.get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
        drop_sandbox_message_index=False,
    ).to_dicts()
    task_start_indices = [
        row["sandbox_message_index"]
        for row in sandbox_rows
        if row["sender"] == RoleType.SYSTEM
        and row["recipient"] == RoleType.USER
        and row["visible_to"] is None
    ]
    task_start_index = max(task_start_indices) if task_start_indices else 0
    task_messages = [
        row
        for row in sandbox_rows
        if row["sandbox_message_index"] >= task_start_index
        and row["visible_to"] is None
    ]

    starting_state = {
        str(namespace): scenario.starting_context.get_database(namespace).to_dicts()
        for namespace in DatabaseNamespace
        if namespace != DatabaseNamespace.SANDBOX
    }

    def matcher_record(matcher: Any) -> dict[str, Any]:
        return {
            "edge_list": matcher.edge_list,
            "nodes": [
                {
                    "node_index": node_index,
                    "constraints": [
                        {
                            "database_namespace": str(constraint.database_namespace),
                            "snapshot_constraint": getattr(
                                constraint.snapshot_constraint,
                                "__name__",
                                type(constraint.snapshot_constraint).__name__,
                            ),
                            "reference_milestone_node_index": (
                                constraint.reference_milestone_node_index
                            ),
                            "target_dataframe": (
                                None
                                if constraint.target_dataframe is None
                                else constraint.target_dataframe.to_dicts()
                            ),
                        }
                        for constraint in milestone.snapshot_constraints
                    ],
                }
                for node_index, milestone in enumerate(matcher.milestones)
            ],
        }

    return {
        "parameter_manifest": manifest.to_dict(),
        "task_messages": _json_compatible(task_messages),
        "starting_state": _json_compatible(starting_state),
        "evaluation_targets": _json_compatible(
            {
                "milestones": matcher_record(scenario.evaluation.milestone_matcher),
                "minefields": matcher_record(scenario.evaluation.minefield_matcher),
            }
        ),
        "tool_configuration": {
            "allow_list": scenario.starting_context.tool_allow_list,
            "deny_list": scenario.starting_context.tool_deny_list,
            "augmentations": [
                str(value) for value in scenario.starting_context.tool_augmentation_list
            ],
        },
    }
