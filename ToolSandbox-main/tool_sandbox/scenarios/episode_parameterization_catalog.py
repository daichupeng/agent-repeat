# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Parameterizers for the audited sequential-episode scenario catalogue.

Structural conditions are represented as complete templates. They are never
sampled as independent booleans because doing so can silently change whether a
scenario is solvable or what workflow its evaluator expects.
"""

from __future__ import annotations

import copy
import datetime
import json
import random
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import holidays
import polars as pl

from tool_sandbox.common.execution_context import DatabaseNamespace, RoleType
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.scenarios.episode_parameterization import (
    ParameterValues,
    ScenarioParameterizer,
    _episode_phone_number,
    _replace_initial_setting_state,
    _replace_message_content,
    _sample_modify_contact_with_message_recency,
)
from tool_sandbox.scenarios.user_simulator_few_shot_examples import USER_INSTRUCTION

SamplerFactory = Callable[[Scenario, random.Random, int, int], ParameterValues]


_MESSAGE_CONTENTS = (
    "Please call me when you have a minute",
    "The meeting moved to Thursday morning",
    "Can you send me the latest draft?",
    "Dinner is booked for seven",
    "I will arrive about twenty minutes late",
)

_REMINDER_CONTENTS = (
    "Call the dentist",
    "Submit the expense report",
    "Pick up the library books",
    "Review the project proposal",
    "Renew the parking permit",
)

_NEW_CONTACT_NAMES = (
    "Ada Lovelace",
    "Grace Hopper",
    "Katherine Johnson",
    "Donald Knuth",
    "Barbara Liskov",
)

_HOLIDAY_NAMES = (
    "Christmas Day",
    "Thanksgiving",
    "Independence Day",
    "New Year's Day",
    "Memorial Day",
    "Labor Day",
)


def _local_now() -> datetime.datetime:
    return (
        datetime.datetime.now(datetime.timezone.utc).astimezone().replace(tzinfo=None)
    )


def _choose(values: Sequence[Any], rng: random.Random, episode_number: int) -> Any:
    choices = list(values)
    rng.shuffle(choices)
    return choices[(episode_number - 1) % len(choices)]


def _contacts(scenario: Scenario) -> list[dict[str, Any]]:
    contacts = (
        scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
        .filter(~pl.col("is_self"))
        .sort("person_id")
        .to_dicts()
    )
    if not contacts:
        raise ValueError("A contact parameterizer requires a non-self contact")
    return cast(list[dict[str, Any]], contacts)


def _reminders(scenario: Scenario) -> list[dict[str, Any]]:
    reminders = (
        scenario.starting_context.get_database(DatabaseNamespace.REMINDER)
        .sort("reminder_id")
        .to_dicts()
    )
    if not reminders:
        raise ValueError("A reminder parameterizer requires a reminder")
    return cast(list[dict[str, Any]], reminders)


def _replace_nested(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        for original in sorted(replacements, key=len, reverse=True):
            value = value.replace(original, replacements[original])
        return value
    if isinstance(value, list):
        return [_replace_nested(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_nested(item, replacements) for item in value)
    if isinstance(value, dict):
        return {key: _replace_nested(item, replacements) for key, item in value.items()}
    return value


def _replace_dataframe_strings(
    dataframe: pl.DataFrame, replacements: Mapping[str, str]
) -> pl.DataFrame:
    rows = [_replace_nested(row, replacements) for row in dataframe.to_dicts()]
    return pl.DataFrame(rows, schema=dataframe.schema)


def _replace_task_and_evaluation_strings(
    scenario: Scenario, replacements: Mapping[str, str]
) -> None:
    sandbox = scenario.starting_context._dbs[DatabaseNamespace.SANDBOX]
    for sender, recipient in (
        (RoleType.SYSTEM, RoleType.USER),
        (RoleType.USER, RoleType.AGENT),
    ):
        matching = sandbox.filter(
            (pl.col("sender") == sender) & (pl.col("recipient") == recipient)
        )
        if not matching.is_empty():
            index = matching["sandbox_message_index"][-1]
            original = matching["content"][-1]
            sandbox = sandbox.with_columns(
                pl.when(pl.col("sandbox_message_index") == index)
                .then(pl.lit(_replace_nested(original, replacements)))
                .otherwise(pl.col("content"))
                .alias("content")
            )
    scenario.starting_context._dbs[DatabaseNamespace.SANDBOX] = sandbox

    for matcher in (
        scenario.evaluation.milestone_matcher,
        scenario.evaluation.minefield_matcher,
    ):
        for milestone in matcher.milestones:
            for constraint in milestone.snapshot_constraints:
                if constraint.target_dataframe is not None:
                    constraint.target_dataframe = _replace_dataframe_strings(
                        constraint.target_dataframe, replacements
                    )


def _constraint(
    scenario: Scenario,
    *,
    node_index: int,
    namespace: DatabaseNamespace,
    minefield: bool = False,
    occurrence: int = 0,
) -> Any:
    matcher = (
        scenario.evaluation.minefield_matcher
        if minefield
        else scenario.evaluation.milestone_matcher
    )
    matches = [
        constraint
        for constraint in matcher.milestones[node_index].snapshot_constraints
        if constraint.database_namespace == namespace
        and constraint.target_dataframe is not None
    ]
    return matches[occurrence]


def _literal_materializer(scenario: Scenario, parameters: ParameterValues) -> Scenario:
    materialized = copy.deepcopy(scenario)
    _replace_task_and_evaluation_strings(materialized, dict(parameters["replacements"]))
    return materialized


def _parameterizer(
    name: str,
    sampler: SamplerFactory,
    materializer: Callable[
        [Scenario, ParameterValues], Scenario
    ] = _literal_materializer,
    aliases: Sequence[str] = (),
) -> ScenarioParameterizer:
    return ScenarioParameterizer(
        name=f"{name}.v1",
        scenario_names=(name, *aliases),
        sample=sampler,
        materialize=materializer,
    )


def _sample_send_message_phone(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del scenario
    phone = _episode_phone_number(
        seed=seed,
        scope="send-message-phone",
        episode_number=episode_number,
        prefix="+1888",
    )
    content = str(_choose(_MESSAGE_CONTENTS, rng, episode_number))
    return {
        "recipient_phone_number": phone,
        "content": content,
        "replacements": {
            "+12453344098": phone,
            "How's the new album coming along": content,
        },
    }


def _sample_add_contact(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del scenario
    name = str(_choose(_NEW_CONTACT_NAMES, rng, episode_number))
    phone = _episode_phone_number(
        seed=seed,
        scope="add-contact-phone",
        episode_number=episode_number,
        prefix="+1877",
    )
    return {
        "name": name,
        "phone_number": phone,
        "relationship": None,
        "replacements": {"Stephen Sondheim": name, "+19876543210": phone},
    }


def _contact_literal_sampler(
    *,
    original_name: str | None = None,
    original_phone: str | None = None,
    original_person_id: str | None = None,
    original_relationship: str | None = None,
    include_content: bool = False,
) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        contact = dict(_choose(_contacts(scenario), rng, episode_number))
        replacements: dict[str, str] = {}
        if original_name is not None:
            replacements[original_name] = str(contact["name"])
        if original_phone is not None:
            replacements[original_phone] = str(contact["phone_number"])
        if original_person_id is not None:
            replacements[original_person_id] = str(contact["person_id"])
        if original_relationship is not None:
            replacements[original_relationship] = str(contact["relationship"])
        content = None
        if include_content:
            content = str(_choose(_MESSAGE_CONTENTS, rng, episode_number))
            replacements["How's the new album coming along"] = content
        return {
            "target_contact": contact,
            "content": content,
            "seed": seed,
            "replacements": replacements,
        }

    return sample


def _sample_search_sender_message(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del seed
    self_ids = {
        row["person_id"]
        for row in scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
        .filter(pl.col("is_self"))
        .to_dicts()
    }
    messages = [
        row
        for row in scenario.starting_context.get_database(
            DatabaseNamespace.MESSAGING
        ).to_dicts()
        if row["sender_person_id"] not in self_ids
        and row["sender_phone_number"] is not None
    ]
    message = dict(_choose(messages, rng, episode_number))
    return {"matching_message": message, "replacements": {}}


def _materialize_search_sender_message(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    message = dict(parameters["matching_message"])
    content = str(message["content"])
    phone = str(message["sender_phone_number"])
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + f'Find which phone number sent you (User A) the message "{content}". '
            f"It should be {phone}. Do not leak this information. "
            "You do not have more information."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=f'Which phone number sent me "{content}"?',
    )
    _constraint(
        materialized, node_index=1, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": f'{phone} sent you "{content}"',
        }
    )
    return materialized


def _sample_unique_relationship_search(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del seed
    contact = dict(_choose(_contacts(scenario), rng, episode_number))
    relationship = str(
        _choose(("mentor", "neighbor", "accountant", "coach"), rng, episode_number)
    )
    return {
        "target_contact": contact,
        "relationship": relationship,
        "replacements": {},
    }


def _materialize_unique_relationship_search(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    contact = dict(parameters["target_contact"])
    relationship = str(parameters["relationship"])
    database = materialized.starting_context._dbs[DatabaseNamespace.CONTACT]
    materialized.starting_context._dbs[DatabaseNamespace.CONTACT] = (
        database.with_columns(
            pl.when(pl.col("is_self"))
            .then(pl.col("relationship"))
            .when(pl.col("person_id") == contact["person_id"])
            .then(pl.lit(relationship))
            .otherwise(pl.lit("acquaintance"))
            .alias("relationship")
        )
    )
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + f"Search for your (User A's) {relationship}'s name. It should be "
            f"{contact['name']}. Do not leak this information. You do not have "
            f"more information about your {relationship}."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=f"What is the name of my {relationship}?",
    )
    _constraint(
        materialized, node_index=0, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.EXECUTION_ENVIRONMENT,
            "recipient": RoleType.AGENT,
            "tool_trace": json.dumps(
                {
                    "tool_name": "search_contacts",
                    "arguments": {"relationship": relationship},
                }
            ),
        }
    )
    _constraint(
        materialized, node_index=1, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": f"Your {relationship} is {contact['name']}",
        }
    )
    return materialized


def _holiday_date(holiday_name: str, year: int) -> datetime.date:
    calendar = holidays.UnitedStates(years=year)
    matches = [date for date, label in calendar.items() if holiday_name in label]
    if not matches:
        raise ValueError(f"Could not resolve {holiday_name!r} in {year}")
    return cast(datetime.date, min(matches))


def _holiday_sampler(
    original_holiday: str, *, next_occurrence: bool = False
) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        del scenario, seed
        holiday_name = str(_choose(_HOLIDAY_NAMES, rng, episode_number))
        today = _local_now().date()
        year = today.year
        date = _holiday_date(holiday_name, year)
        if next_occurrence and date < today:
            year += 1
            date = _holiday_date(holiday_name, year)
        replacements = {
            original_holiday: holiday_name,
            f"12/25/{today.year}": f"{date.month}/{date.day}/{year}",
        }
        return {
            "holiday_name": holiday_name,
            "year": year,
            "date": date.isoformat(),
            "next_occurrence": next_occurrence,
            "replacements": replacements,
        }

    return sample


def _rewrite_search_holiday_year(value: Any, year: int) -> Any:
    if isinstance(value, list):
        return [_rewrite_search_holiday_year(item, year) for item in value]
    if not isinstance(value, dict):
        return value
    rewritten = {
        key: _rewrite_search_holiday_year(item, year) for key, item in value.items()
    }
    if rewritten.get("tool_name") == "search_holiday":
        arguments = dict(rewritten.get("arguments", {}))
        arguments["year"] = year
        rewritten["arguments"] = arguments
    return rewritten


def _materialize_holiday(scenario: Scenario, parameters: ParameterValues) -> Scenario:
    materialized = _literal_materializer(scenario, parameters)
    if not bool(parameters["next_occurrence"]):
        return materialized

    holiday_name = str(parameters["holiday_name"])
    year = int(parameters["year"])
    date = datetime.date.fromisoformat(str(parameters["date"]))
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + f"Search how many days it is till {holiday_name}. {holiday_name} is "
            f"{date.month}/{date.day}/{year}. Do not leak this information."
        ),
    )
    for milestone in materialized.evaluation.milestone_matcher.milestones:
        for constraint in milestone.snapshot_constraints:
            target = constraint.target_dataframe
            if target is None or "tool_trace" not in target.columns:
                continue
            rows = target.to_dicts()
            changed = False
            for row in rows:
                trace = row.get("tool_trace")
                if not isinstance(trace, str):
                    continue
                parsed = json.loads(trace)
                rewritten = _rewrite_search_holiday_year(parsed, year)
                if rewritten != parsed:
                    row["tool_trace"] = json.dumps(rewritten, ensure_ascii=False)
                    changed = True
            if changed:
                constraint.target_dataframe = pl.DataFrame(rows, schema=target.schema)
    return materialized


def _sample_cellular_template(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del scenario, seed
    template = dict(
        _choose(
            (
                {
                    "initial": True,
                    "target": False,
                    "action": "off",
                    "response": "Cellular service is turned off",
                },
                {
                    "initial": False,
                    "target": True,
                    "action": "on",
                    "response": "Cellular service is turned on",
                },
            ),
            rng,
            episode_number,
        )
    )
    return {"state_template": template, "replacements": {}}


def _materialize_cellular_template(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    template = parameters["state_template"]
    _replace_initial_setting_state(
        materialized, {"cellular": bool(template["initial"])}
    )
    action = str(template["action"])
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=USER_INSTRUCTION + f"Turn {action} cellular service",
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=f"Turn {action} cellular",
    )
    _constraint(
        materialized, node_index=0, namespace=DatabaseNamespace.SETTING
    ).target_dataframe = pl.DataFrame({"cellular": bool(template["target"])})
    _constraint(
        materialized, node_index=1, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": str(template["response"]),
        }
    )
    return materialized


def _relationship_sampler(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del seed
    contacts = _contacts(scenario)
    rng.shuffle(contacts)
    target_contacts = contacts[:2]
    source, destination = _choose(
        (
            ("friend", "enemy"),
            ("coworker", "favorite"),
            ("neighbor", "emergency contact"),
        ),
        rng,
        episode_number,
    )
    return {
        "target_contacts": target_contacts,
        "source_relationship": source,
        "destination_relationship": destination,
        "replacements": {},
    }


def _plural(relationship: str) -> str:
    if relationship == "enemy":
        return "enemies"
    return relationship + "s"


def _set_contact_relationship_state(
    scenario: Scenario,
    target_contacts: Sequence[Mapping[str, Any]],
    source_relationship: str,
) -> None:
    ids = [contact["person_id"] for contact in target_contacts]
    database = scenario.starting_context._dbs[DatabaseNamespace.CONTACT]
    scenario.starting_context._dbs[DatabaseNamespace.CONTACT] = database.with_columns(
        pl.when(pl.col("is_self"))
        .then(pl.col("relationship"))
        .when(pl.col("person_id").is_in(ids))
        .then(pl.lit(source_relationship))
        .otherwise(pl.lit("acquaintance"))
        .alias("relationship")
    )


def _relationship_target(
    contacts: Sequence[Mapping[str, Any]], relationship: str
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "person_id": contact["person_id"],
                "name": contact["name"],
                "phone_number": contact["phone_number"],
                "relationship": relationship,
                "is_self": False,
            }
            for contact in contacts
        ]
    )


def _materialize_relationship_transition(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    contacts = [dict(contact) for contact in parameters["target_contacts"]]
    source = str(parameters["source_relationship"])
    destination = str(parameters["destination_relationship"])
    _set_contact_relationship_state(materialized, contacts, source)
    source_plural = _plural(source)
    destination_plural = _plural(destination)
    is_twice = "twice_multiple_user_turn" in parameters["scenario_kind"]
    is_multi_turn = "multiple_user_turn" in parameters["scenario_kind"]

    goal = (
        f"Ask User B to update all {source_plural} in your contact book as "
        f"your {destination_plural}."
    )
    if is_twice:
        goal += (
            f" After User B did so, ask User B to update them back to your "
            f"{source_plural}."
        )
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=USER_INSTRUCTION + goal + " You do not have more information.",
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=(
            f"Who are my {source_plural}?"
            if is_multi_turn
            else f"Make all of my {source_plural} my {destination_plural}"
        ),
    )
    _constraint(
        materialized, node_index=0, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.EXECUTION_ENVIRONMENT,
            "recipient": RoleType.AGENT,
            "tool_trace": json.dumps(
                {
                    "tool_name": "search_contacts",
                    "arguments": {"relationship": source},
                }
            ),
        }
    )
    _constraint(
        materialized, node_index=1, namespace=DatabaseNamespace.CONTACT
    ).target_dataframe = _relationship_target(contacts, destination)
    names = " and ".join(str(contact["name"]) for contact in contacts)
    response_node = 2
    response = (
        f"{names} are now your {destination_plural}"
        if is_multi_turn
        else f"All your {source_plural} are now your {destination_plural}"
    )
    _constraint(
        materialized, node_index=response_node, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": response,
        }
    )
    if is_twice:
        _constraint(
            materialized, node_index=3, namespace=DatabaseNamespace.CONTACT
        ).target_dataframe = _relationship_target(contacts, source)
        _constraint(
            materialized, node_index=4, namespace=DatabaseNamespace.SANDBOX
        ).target_dataframe = pl.DataFrame(
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": f"{names} are now your {source_plural} again.",
            }
        )
    return materialized


def _relationship_sampler_for(kind: str) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        values = dict(_relationship_sampler(scenario, rng, episode_number, seed))
        values["scenario_kind"] = kind
        return values

    return sample


def _set_message_order(
    scenario: Scenario, selected_message_id: str, *, oldest: bool
) -> None:
    database = scenario.starting_context._dbs[DatabaseNamespace.MESSAGING]
    message_ids = sorted(
        row["message_id"]
        for row in scenario.starting_context.get_database(
            DatabaseNamespace.MESSAGING
        ).to_dicts()
    )
    anchor = 1_700_000_000.0
    expression = pl.col("creation_timestamp")
    for index, message_id in enumerate(message_ids):
        if message_id == selected_message_id:
            timestamp = anchor - 100_000 if oldest else anchor + 100_000
        else:
            timestamp = anchor + index * 1_000 if oldest else anchor - index * 1_000
        expression = (
            pl.when(pl.col("message_id") == message_id)
            .then(pl.lit(timestamp))
            .otherwise(expression)
        )
    scenario.starting_context._dbs[DatabaseNamespace.MESSAGING] = database.with_columns(
        expression.alias("creation_timestamp")
    )


def _sample_oldest_message(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del seed
    messages = scenario.starting_context.get_database(
        DatabaseNamespace.MESSAGING
    ).to_dicts()
    message = dict(_choose(messages, rng, episode_number))
    return {"target_message": message, "replacements": {}}


def _materialize_oldest_message(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    message = dict(parameters["target_message"])
    _set_message_order(materialized, str(message["message_id"]), oldest=True)
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION + "Find the content of your (User A's) oldest message. "
            "You do not have more information."
        ),
    )
    responses = [
        f"Your first ever text says '{message['content']}'.",
        str(message["content"]),
    ]
    for occurrence, response in enumerate(responses):
        _constraint(
            materialized,
            node_index=2,
            namespace=DatabaseNamespace.SANDBOX,
            occurrence=occurrence,
        ).target_dataframe = pl.DataFrame(
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": response,
            }
        )
    return materialized


def _set_reminder_order(
    scenario: Scenario,
    selected_reminder_id: str,
    *,
    field: str,
    mode: str,
) -> None:
    database = scenario.starting_context._dbs[DatabaseNamespace.REMINDER]
    reminder_ids = sorted(
        row["reminder_id"]
        for row in scenario.starting_context.get_database(
            DatabaseNamespace.REMINDER
        ).to_dicts()
    )
    now = _local_now().replace(minute=0, second=0, microsecond=0)
    expression = pl.col(field)
    for index, reminder_id in enumerate(reminder_ids):
        if mode in {"created_yesterday", "due_yesterday"}:
            yesterday = (now - datetime.timedelta(days=1)).replace(hour=12)
            timestamp = (
                yesterday
                if reminder_id == selected_reminder_id
                else now - datetime.timedelta(days=5 + index)
            ).timestamp()
        elif mode == "upcoming":
            timestamp = (
                now + datetime.timedelta(hours=1)
                if reminder_id == selected_reminder_id
                else now + datetime.timedelta(days=2 + index)
            ).timestamp()
        elif mode == "later_today":
            base = now.replace(
                hour=min(23, max(now.hour + 1, 18)),
                minute=30 if now.hour == 23 else 0,
            )
            timestamp = (
                base.timestamp()
                if reminder_id == selected_reminder_id
                else (now + datetime.timedelta(days=3 + index)).timestamp()
            )
        elif mode == "latest_created":
            timestamp = (
                (now - datetime.timedelta(hours=1)).timestamp()
                if reminder_id == selected_reminder_id
                else (now - datetime.timedelta(days=3 + index)).timestamp()
            )
        else:
            raise ValueError(f"Unknown reminder ordering mode: {mode}")
        expression = (
            pl.when(pl.col("reminder_id") == reminder_id)
            .then(pl.lit(timestamp))
            .otherwise(expression)
        )
    scenario.starting_context._dbs[DatabaseNamespace.REMINDER] = database.with_columns(
        expression.alias(field)
    )


def _reminder_selection_sampler(mode: str) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        del seed
        reminder = dict(_choose(_reminders(scenario), rng, episode_number))
        wording = str(
            _choose(
                (
                    "What's the todo item I made yesterday?",
                    "Which reminder did I create yesterday?",
                    "What did I add to my reminders yesterday?",
                )
                if mode == "created_yesterday"
                else (
                    "What's my todo yesterday?",
                    "Which reminder was due yesterday?",
                    "What was on my reminder list yesterday?",
                ),
                rng,
                episode_number,
            )
        )
        return {
            "target_reminder": reminder,
            "mode": mode,
            "wording": wording,
            "replacements": {},
        }

    return sample


def _materialize_positive_reminder_search(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    reminder = dict(parameters["target_reminder"])
    mode = str(parameters["mode"])
    field = (
        "creation_timestamp" if mode == "created_yesterday" else "reminder_timestamp"
    )
    _set_reminder_order(
        materialized,
        str(reminder["reminder_id"]),
        field=field,
        mode=mode,
    )
    descriptor = (
        "created yesterday" if mode == "created_yesterday" else "from yesterday"
    )
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + f"Ask User B to find the content of your reminder {descriptor} in an "
            f"implicit manner. It should say {reminder['content']}. Do not leak this "
            "information. You do not have any more information."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(parameters["wording"]),
    )
    responses = [
        f"Your reminder {descriptor} says '{reminder['content']}'.",
        str(reminder["content"]),
    ]
    for occurrence, response in enumerate(responses):
        _constraint(
            materialized,
            node_index=2,
            namespace=DatabaseNamespace.SANDBOX,
            occurrence=occurrence,
        ).target_dataframe = pl.DataFrame(
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": response,
            }
        )
    return materialized


def _sample_remove_upcoming_reminder(
    scenario: Scenario, rng: random.Random, episode_number: int, seed: int
) -> ParameterValues:
    del seed
    reminder = dict(_choose(_reminders(scenario), rng, episode_number))
    return {"target_reminder": reminder, "replacements": {}}


def _materialize_remove_upcoming_reminder(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    reminder = dict(parameters["target_reminder"])
    _set_reminder_order(
        materialized,
        str(reminder["reminder_id"]),
        field="reminder_timestamp",
        mode="upcoming",
    )
    _constraint(
        materialized, node_index=2, namespace=DatabaseNamespace.REMINDER
    ).target_dataframe = pl.DataFrame({"reminder_id": reminder["reminder_id"]})
    return materialized


def _add_reminder_sampler(kind: str) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        del scenario, seed
        content = str(_choose(_REMINDER_CONTENTS, rng, episode_number))
        hour, time_text = _choose(
            ((9, "9AM"), (11, "11AM"), (14, "2PM"), (17, "5PM")),
            rng,
            episode_number,
        )
        now = _local_now()
        if kind == "weekday":
            weekday, day_text = _choose(
                (
                    (1, "next Monday"),
                    (2, "next Tuesday"),
                    (4, "next Thursday"),
                    (5, "next Friday"),
                ),
                rng,
                episode_number,
            )
            delta = (weekday - now.isoweekday()) % 7
            target = (now + datetime.timedelta(days=delta)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
        else:
            days, day_text = _choose(
                ((1, "tomorrow"), (2, "in two days"), (3, "in three days")),
                rng,
                episode_number,
            )
            target = (now + datetime.timedelta(days=days)).replace(
                hour=hour, minute=0, second=0, microsecond=0
            )
        return {
            "content": content,
            "day_text": day_text,
            "hour": hour,
            "time_text": time_text,
            "target_timestamp": target.timestamp(),
            "kind": kind,
            "replacements": {},
        }

    return sample


def _materialize_add_reminder(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    content = str(parameters["content"])
    day_text = str(parameters["day_text"])
    time_text = str(parameters["time_text"])
    is_multi_turn = "multiple_user_turn" in str(parameters["scenario_kind"])
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + f"Ask User B create a reminder to {content.lower()} {day_text} "
            f"{time_text}. You do not have any more information."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=(
            f"Remind me to {content.lower()}"
            if is_multi_turn
            else f"Remind me to {content.lower()} {day_text} {time_text}"
        ),
    )
    _constraint(
        materialized, node_index=1, namespace=DatabaseNamespace.REMINDER
    ).target_dataframe = pl.DataFrame(
        {
            "content": content,
            "reminder_timestamp": float(parameters["target_timestamp"]),
        }
    )
    return materialized


def _add_reminder_sampler_for(kind: str, scenario_kind: str) -> SamplerFactory:
    base_sampler = _add_reminder_sampler(kind)

    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        values = dict(base_sampler(scenario, rng, episode_number, seed))
        values["scenario_kind"] = scenario_kind
        return values

    return sample


def _apply_message_recency_state(
    scenario: Scenario, parameters: ParameterValues
) -> None:
    database = scenario.starting_context._dbs[DatabaseNamespace.MESSAGING]
    creation_timestamp = pl.col("creation_timestamp")
    recipient_person_id = pl.col("recipient_person_id")
    recipient_phone_number = pl.col("recipient_phone_number")
    for message_value in parameters["starting_outgoing_messages"]:
        message = dict(message_value)
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
    scenario.starting_context._dbs[DatabaseNamespace.MESSAGING] = database.with_columns(
        creation_timestamp.alias("creation_timestamp"),
        recipient_person_id.alias("recipient_person_id"),
        recipient_phone_number.alias("recipient_phone_number"),
    )


def _materialize_message_recency_multi_turn(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    _apply_message_recency_state(materialized, parameters)
    contact = dict(parameters["target_contact"])
    new_phone = str(parameters["new_phone_number"])
    selector = parameters["recency_selector"]
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + "Update the phone number of "
            + str(selector["system_text"])
            + f" to {new_phone}. You do not know who the person is."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(
            _choose(
                ("Who did I talk to last?", "Who did I message most recently?"),
                random.Random(new_phone),
                1,
            )
        ),
    )
    _constraint(
        materialized, node_index=3, namespace=DatabaseNamespace.CONTACT
    ).target_dataframe = pl.DataFrame(
        {"person_id": contact["person_id"], "phone_number": new_phone}
    )
    _constraint(
        materialized, node_index=4, namespace=DatabaseNamespace.SANDBOX
    ).target_dataframe = pl.DataFrame(
        {
            "sender": RoleType.AGENT,
            "recipient": RoleType.USER,
            "content": f"{contact['name']}'s phone number has been updated to {new_phone}.",
        }
    )
    return materialized


def _materialize_message_recency_insufficient(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    _apply_message_recency_state(materialized, parameters)
    selector = parameters["recency_selector"]
    new_phone = str(parameters["new_phone_number"])
    _replace_message_content(
        materialized,
        sender=RoleType.SYSTEM,
        recipient=RoleType.USER,
        content=(
            USER_INSTRUCTION
            + "Update the phone number of "
            + str(selector["system_text"])
            + f" to {new_phone}. You do not have any more information about that person."
        ),
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=(
            "Update the phone number of "
            + str(selector["user_text"])
            + f" to {new_phone}"
        ),
    )
    return materialized


def _sample_insufficient_reminder(mode: str, *, modify: bool = False) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        del seed
        reminder = dict(_choose(_reminders(scenario), rng, episode_number))
        hour, time_text = _choose(
            ((9, "9AM"), (14, "2PM"), (17, "5PM")), rng, episode_number
        )
        return {
            "latent_reminder": reminder,
            "mode": mode,
            "modify": modify,
            "time_text": time_text,
            "hour": hour,
            "replacements": {},
        }

    return sample


def _materialize_insufficient_reminder(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    reminder = dict(parameters["latent_reminder"])
    mode = str(parameters["mode"])
    field = (
        "creation_timestamp" if mode == "created_yesterday" else "reminder_timestamp"
    )
    _set_reminder_order(
        materialized,
        str(reminder["reminder_id"]),
        field=field,
        mode=mode,
    )
    if bool(parameters["modify"]):
        time_text = str(parameters["time_text"])
        _replace_message_content(
            materialized,
            sender=RoleType.SYSTEM,
            recipient=RoleType.USER,
            content=(
                USER_INSTRUCTION
                + f"Ask User B to postpone your upcoming reminder to tomorrow {time_text}. "
                "You do not have any more information."
            ),
        )
        _replace_message_content(
            materialized,
            sender=RoleType.USER,
            recipient=RoleType.AGENT,
            content=f"Push my upcoming reminder to tomorrow {time_text}.",
        )
    return materialized


def _insufficient_search_sampler(mode: str) -> SamplerFactory:
    def sample(
        scenario: Scenario, rng: random.Random, episode_number: int, seed: int
    ) -> ParameterValues:
        del seed
        reminder = dict(_choose(_reminders(scenario), rng, episode_number))
        wordings = {
            "created_yesterday": (
                "What's the todo item I made yesterday?",
                "Which reminder did I create yesterday?",
            ),
            "due_yesterday": (
                "What's my todo yesterday?",
                "Which reminder was due yesterday?",
            ),
            "later_today": (
                "What's on my todo later?",
                "Which reminder is coming up later today?",
            ),
        }
        return {
            "latent_reminder": reminder,
            "mode": mode,
            "wording": _choose(wordings[mode], rng, episode_number),
            "replacements": {},
        }

    return sample


def _materialize_insufficient_search(
    scenario: Scenario, parameters: ParameterValues
) -> Scenario:
    materialized = copy.deepcopy(scenario)
    reminder = dict(parameters["latent_reminder"])
    mode = str(parameters["mode"])
    field = (
        "creation_timestamp" if mode == "created_yesterday" else "reminder_timestamp"
    )
    _set_reminder_order(
        materialized,
        str(reminder["reminder_id"]),
        field=field,
        mode=mode,
    )
    _replace_message_content(
        materialized,
        sender=RoleType.USER,
        recipient=RoleType.AGENT,
        content=str(parameters["wording"]),
    )
    return materialized


def remaining_parameterizers() -> list[ScenarioParameterizer]:
    """Return parameterizers for every previously unchecked catalogue row."""
    fredrik_id = "9e137f06-916a-5310-8174-cf0b7e9f7054"
    parameterizers = [
        _parameterizer(
            "search_sender_phone_number_with_content",
            _sample_search_sender_message,
            _materialize_search_sender_message,
        ),
        _parameterizer(
            "send_message_with_phone_number_and_content", _sample_send_message_phone
        ),
        _parameterizer("add_contact_with_name_and_phone_number", _sample_add_contact),
        _parameterizer(
            "search_name_with_relationship",
            _sample_unique_relationship_search,
            _materialize_unique_relationship_search,
        ),
        _parameterizer(
            "find_days_till_holiday_wifi_off",
            _holiday_sampler("Christmas Day", next_occurrence=True),
            _materialize_holiday,
        ),
        _parameterizer(
            "send_message_with_contact_content_cellular_off",
            _contact_literal_sampler(
                original_name="Fredrik Thordendal",
                original_phone="+12453344098",
                include_content=True,
            ),
        ),
        _parameterizer(
            "update_contact_relationship_with_relationship",
            _relationship_sampler_for("single"),
            _materialize_relationship_transition,
        ),
        _parameterizer(
            "update_contact_relationship_with_relationship_twice_multiple_user_turn",
            _relationship_sampler_for("twice_multiple_user_turn"),
            _materialize_relationship_transition,
        ),
        _parameterizer(
            "modify_contact_with_message_recency_multiple_user_turn",
            _sample_modify_contact_with_message_recency,
            _materialize_message_recency_multi_turn,
        ),
        _parameterizer(
            "find_days_till_holiday_wifi_off_multiple_user_turn",
            _holiday_sampler("Christmas Day", next_occurrence=True),
            _materialize_holiday,
        ),
        _parameterizer(
            "send_message_with_contact_content_cellular_off_multiple_user_turn",
            _contact_literal_sampler(
                original_name="Fredrik Thordendal",
                original_phone="+12453344098",
                include_content=True,
            ),
        ),
        _parameterizer(
            "add_reminder_content_and_weekday_delta_and_time_multiple_user_turn",
            _add_reminder_sampler_for("weekday", "weekday_multiple_user_turn"),
            _materialize_add_reminder,
        ),
        _parameterizer(
            "remove_contact_with_id",
            _contact_literal_sampler(original_person_id=fredrik_id),
        ),
        _parameterizer(
            "search_relationship_with_phone_number",
            _contact_literal_sampler(
                original_phone="+10000000000", original_relationship="boss"
            ),
        ),
        _parameterizer(
            "search_phone_number_with_name",
            _contact_literal_sampler(
                original_name="Homer S", original_phone="+10000000000"
            ),
        ),
        _parameterizer(
            "find_thanksgiving_timestamp",
            _holiday_sampler("Thanksgiving"),
            _materialize_holiday,
        ),
        _parameterizer(
            "cellular_off", _sample_cellular_template, _materialize_cellular_template
        ),
        _parameterizer(
            "remove_reminder_with_recency_latest",
            _sample_remove_upcoming_reminder,
            _materialize_remove_upcoming_reminder,
        ),
        _parameterizer(
            "search_reminder_with_creation_recency_yesterday_implicit",
            _reminder_selection_sampler("created_yesterday"),
            _materialize_positive_reminder_search,
        ),
        _parameterizer(
            "search_reminder_with_recency_yesterday_implicit",
            _reminder_selection_sampler("due_yesterday"),
            _materialize_positive_reminder_search,
        ),
        _parameterizer(
            "add_reminder_content_and_weekday_delta_and_time",
            _add_reminder_sampler_for("weekday", "weekday_single_turn"),
            _materialize_add_reminder,
        ),
        _parameterizer(
            "search_message_with_recency_oldest_multiple_user_turn",
            _sample_oldest_message,
            _materialize_oldest_message,
        ),
        _parameterizer(
            "remove_contact_by_phone_multiple_user_turn",
            _contact_literal_sampler(
                original_phone="+12453344098", original_person_id=fredrik_id
            ),
        ),
        _parameterizer(
            "update_contact_relationship_with_relationship_multiple_user_turn",
            _relationship_sampler_for("multiple_user_turn"),
            _materialize_relationship_transition,
        ),
        _parameterizer(
            "find_days_till_holiday_multiple_user_turn",
            _holiday_sampler("Christmas Day", next_occurrence=True),
            _materialize_holiday,
        ),
        _parameterizer(
            "add_reminder_content_and_week_delta_and_time_multiple_user_turn",
            _add_reminder_sampler_for("week_delta", "week_delta_multiple_user_turn"),
            _materialize_add_reminder,
        ),
        _parameterizer(
            "modify_reminder_with_recency_latest_insufficient_information",
            _sample_insufficient_reminder("upcoming", modify=True),
            _materialize_insufficient_reminder,
        ),
        _parameterizer(
            "remove_reminder_with_recency_latest_insufficient_information",
            _sample_insufficient_reminder("upcoming"),
            _materialize_insufficient_reminder,
        ),
        _parameterizer(
            "modify_contact_with_message_recency_insufficient_information",
            _sample_modify_contact_with_message_recency,
            _materialize_message_recency_insufficient,
        ),
        _parameterizer(
            "send_message_with_contact_content_cellular_off_insufficient_information",
            _contact_literal_sampler(
                original_name="Fredrik Thordendal", include_content=True
            ),
        ),
        _parameterizer(
            "search_reminder_with_creation_recency_yesterday_insufficient_information_implicit",
            _insufficient_search_sampler("created_yesterday"),
            _materialize_insufficient_search,
        ),
        _parameterizer(
            "search_reminder_with_recency_yesterday_insufficient_information_implicit",
            _insufficient_search_sampler("due_yesterday"),
            _materialize_insufficient_search,
        ),
        _parameterizer(
            "search_reminder_with_recency_upcoming_insufficient_information_implicit",
            _insufficient_search_sampler("later_today"),
            _materialize_insufficient_search,
        ),
        _parameterizer(
            "remove_contact_by_phone_no_search_contacts_insufficient_information",
            _contact_literal_sampler(original_phone="+12453344098"),
        ),
        _parameterizer(
            "remove_contact_by_phone_no_remove_contact_insufficient_information",
            _contact_literal_sampler(original_phone="+12453344098"),
        ),
    ]
    return parameterizers
