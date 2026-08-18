import copy
import json
from typing import Any

import polars as pl
import pytest

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    RoleType,
    set_current_context,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.scenarios.episode_parameterization import (
    DEFAULT_EPISODE_RANDOMIZER,
    MaterializedEpisode,
    episode_setup_record,
)
from tool_sandbox.tools.contact import modify_contact
from tool_sandbox.tools.reminder import modify_reminder

SCENARIO_NAME = "update_contact_with_id_and_phone_number"
REMINDER_SCENARIO_NAME = "modify_reminder_with_recency_latest"
RECENCY_CONTACT_SCENARIO_NAME = "modify_contact_with_message_recency"


def _scenario() -> Scenario:
    return named_scenarios(ToolBackend.DEFAULT)[SCENARIO_NAME]


def _materialize(episode_number: int, seed: int = 17) -> MaterializedEpisode:
    return DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=SCENARIO_NAME,
        scenario=_scenario(),
        episode_number=episode_number,
        seed=seed,
    )


def _materialize_reminder(episode_number: int, seed: int = 17) -> MaterializedEpisode:
    return DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=REMINDER_SCENARIO_NAME,
        scenario=named_scenarios(ToolBackend.DEFAULT)[REMINDER_SCENARIO_NAME],
        episode_number=episode_number,
        seed=seed,
    )


def _materialize_recency_contact(
    episode_number: int, seed: int = 17
) -> MaterializedEpisode:
    return DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=RECENCY_CONTACT_SCENARIO_NAME,
        scenario=named_scenarios(ToolBackend.DEFAULT)[RECENCY_CONTACT_SCENARIO_NAME],
        episode_number=episode_number,
        seed=seed,
    )


def _evaluate_update(
    episode: MaterializedEpisode,
    *,
    target_contact: dict[str, Any],
    new_phone_number: str,
    response: str,
) -> float:
    execution_context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(execution_context)
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "modify_contact",
            }
        ],
    )
    modify_contact(
        person_id=target_contact["person_id"],
        phone_number=new_phone_number,
    )
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.EXECUTION_ENVIRONMENT,
                "recipient": RoleType.AGENT,
                "content": "None",
            },
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": response,
            },
        ],
    )
    return episode.scenario.evaluation.evaluate(
        execution_context=execution_context,
        max_turn_count=episode.scenario.max_messages,
    ).similarity


def _evaluate_reminder_update(
    episode: MaterializedEpisode,
    *,
    reminder_id: str,
    content: str,
    reminder_timestamp: float,
) -> float:
    execution_context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(execution_context)
    for tool_name in ("search_reminder", "get_current_timestamp"):
        execution_context.add_to_database(
            DatabaseNamespace.SANDBOX,
            [
                {
                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                    "recipient": RoleType.AGENT,
                    "content": "None",
                    "tool_trace": [
                        json.dumps(
                            {"tool_name": tool_name, "arguments": {}},
                            ensure_ascii=False,
                        )
                    ],
                }
            ],
        )
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "modify_reminder",
            }
        ],
    )
    modify_reminder(
        reminder_id=reminder_id,
        content=content,
        reminder_timestamp=reminder_timestamp,
    )
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.EXECUTION_ENVIRONMENT,
                "recipient": RoleType.AGENT,
                "content": "None",
            }
        ],
    )
    return episode.scenario.evaluation.evaluate(
        execution_context=execution_context,
        max_turn_count=episode.scenario.max_messages,
    ).similarity


def _evaluate_recency_contact_update(
    episode: MaterializedEpisode,
    *,
    target_contact: dict[str, Any],
    new_phone_number: str,
    response: str,
) -> float:
    execution_context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(execution_context)
    for tool_name in (
        "get_current_timestamp",
        "search_contacts",
        "search_messages",
    ):
        execution_context.add_to_database(
            DatabaseNamespace.SANDBOX,
            [
                {
                    "sender": RoleType.EXECUTION_ENVIRONMENT,
                    "recipient": RoleType.AGENT,
                    "content": "None",
                    "tool_trace": [
                        json.dumps(
                            {"tool_name": tool_name, "arguments": {}},
                            ensure_ascii=False,
                        )
                    ],
                }
            ],
        )
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.EXECUTION_ENVIRONMENT,
                "content": "modify_contact",
            }
        ],
    )
    modify_contact(
        person_id=target_contact["person_id"],
        phone_number=new_phone_number,
    )
    execution_context.add_to_database(
        DatabaseNamespace.SANDBOX,
        [
            {
                "sender": RoleType.EXECUTION_ENVIRONMENT,
                "recipient": RoleType.AGENT,
                "content": "None",
            },
            {
                "sender": RoleType.AGENT,
                "recipient": RoleType.USER,
                "content": response,
            },
        ],
    )
    return episode.scenario.evaluation.evaluate(
        execution_context=execution_context,
        max_turn_count=episode.scenario.max_messages,
    ).similarity


def test_parameterization_is_deterministic_and_varies_consecutive_episodes() -> None:
    episode_1 = _materialize(1)
    episode_1_repeated = _materialize(1)
    episode_2 = _materialize(2)

    assert episode_1.manifest.to_dict() == episode_1_repeated.manifest.to_dict()
    assert episode_1.manifest.manifest_id == episode_1_repeated.manifest.manifest_id
    with pytest.raises(TypeError):
        episode_1.manifest.parameters["target_contact"]["name"] = "mutated"
    assert (
        episode_1.manifest.parameters["target_contact"]["person_id"]
        != episode_2.manifest.parameters["target_contact"]["person_id"]
    )
    assert (
        episode_1.manifest.parameters["new_phone_number"]
        != episode_2.manifest.parameters["new_phone_number"]
    )


def test_task_state_and_metric_targets_share_one_manifest() -> None:
    episode = _materialize(1)
    parameters = episode.manifest.parameters
    target_contact = parameters["target_contact"]
    new_phone_number = parameters["new_phone_number"]

    sandbox = episode.scenario.starting_context.get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
    )
    assert sandbox["content"][-1] == parameters["task"]["user_to_agent"]

    starting_contact = episode.scenario.starting_context.get_database(
        DatabaseNamespace.CONTACT
    ).filter(pl.col("person_id") == target_contact["person_id"])
    assert starting_contact["phone_number"][0] == target_contact["phone_number"]
    assert starting_contact["phone_number"][0] != new_phone_number

    update_target = (
        episode.scenario.evaluation.milestone_matcher.milestones[0]
        .snapshot_constraints[0]
        .target_dataframe
    )
    assert update_target is not None
    assert update_target["person_id"][0] == target_contact["person_id"]
    assert update_target["phone_number"][0] == new_phone_number


def test_randomized_metric_accepts_current_target_and_rejects_previous_target() -> None:
    episode_1 = _materialize(1)
    episode_2 = _materialize(2)
    parameters_1 = episode_1.manifest.parameters
    parameters_2 = episode_2.manifest.parameters

    expected_response_2 = (
        episode_2.scenario.evaluation.milestone_matcher.milestones[1]
        .snapshot_constraints[0]
        .target_dataframe["content"][0]
    )
    assert (
        _evaluate_update(
            episode_2,
            target_contact=dict(parameters_2["target_contact"]),
            new_phone_number=str(parameters_2["new_phone_number"]),
            response=expected_response_2,
        )
        == 1.0
    )

    previous_response = (
        episode_1.scenario.evaluation.milestone_matcher.milestones[1]
        .snapshot_constraints[0]
        .target_dataframe["content"][0]
    )
    assert (
        _evaluate_update(
            episode_2,
            target_contact=dict(parameters_1["target_contact"]),
            new_phone_number=str(parameters_1["new_phone_number"]),
            response=previous_response,
        )
        < 1.0
    )


def test_identity_and_disabled_parameterization_are_backward_compatible() -> None:
    scenario = _scenario()
    with pytest.raises(ValueError, match="at least 1"):
        DEFAULT_EPISODE_RANDOMIZER.materialize(
            scenario_name=SCENARIO_NAME,
            scenario=scenario,
            episode_number=0,
            seed=17,
        )
    unsupported = DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name="unregistered_scenario",
        scenario=scenario,
        episode_number=3,
        seed=17,
    )
    disabled = DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=SCENARIO_NAME,
        scenario=scenario,
        episode_number=3,
        seed=17,
        enabled=False,
    )

    assert unsupported.scenario is scenario
    assert not unsupported.manifest.is_parameterized
    assert disabled.scenario is scenario
    assert not disabled.manifest.is_parameterized


def test_registered_parameterizer_applies_to_generated_tool_variants() -> None:
    variant_name = f"{SCENARIO_NAME}_3_distraction_tools"
    variant = named_scenarios(ToolBackend.DEFAULT)[variant_name]
    episode = DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=variant_name,
        scenario=variant,
        episode_number=2,
        seed=17,
    )

    assert episode.manifest.is_parameterized
    assert episode.manifest.base_scenario_name == SCENARIO_NAME
    assert episode.scenario.categories == variant.categories


def test_episode_setup_logs_task_and_pre_episode_state() -> None:
    episode = _materialize(1)
    setup = episode_setup_record(
        scenario=episode.scenario,
        manifest=episode.manifest,
    )

    assert setup["parameter_manifest"]["manifest_id"] == episode.manifest.manifest_id
    assert (
        setup["task_messages"][-1]["content"]
        == episode.manifest.parameters["task"]["user_to_agent"]
    )
    assert setup["starting_state"]["CONTACT"] == (
        episode.scenario.starting_context.get_database(
            DatabaseNamespace.CONTACT
        ).to_dicts()
    )
    metric_target = setup["evaluation_targets"]["milestones"]["nodes"][0][
        "constraints"
    ][0]["target_dataframe"][0]
    assert (
        metric_target["person_id"]
        == episode.manifest.parameters["target_contact"]["person_id"]
    )
    assert (
        metric_target["phone_number"] == episode.manifest.parameters["new_phone_number"]
    )
    json.dumps(setup)


def test_reminder_parameterization_randomizes_target_state_content_and_time() -> None:
    episode_1 = _materialize_reminder(1)
    episode_1_repeated = _materialize_reminder(1)
    episode_2 = _materialize_reminder(2)

    assert episode_1.manifest.to_dict() == episode_1_repeated.manifest.to_dict()
    parameters_1 = episode_1.manifest.parameters
    parameters_2 = episode_2.manifest.parameters
    assert (
        parameters_1["target_reminder"]["reminder_id"]
        != parameters_2["target_reminder"]["reminder_id"]
    )
    assert parameters_1["new_content"] != parameters_2["new_content"]
    assert parameters_1["time"] != parameters_2["time"]

    reminders = episode_1.scenario.starting_context.get_database(
        DatabaseNamespace.REMINDER
    )
    latest_reminder_id = reminders.sort("creation_timestamp")["reminder_id"][-1]
    assert latest_reminder_id == parameters_1["target_reminder"]["reminder_id"]


def test_reminder_task_state_and_metric_target_share_one_manifest() -> None:
    episode = _materialize_reminder(1)
    parameters = episode.manifest.parameters
    target_reminder = parameters["target_reminder"]

    sandbox = episode.scenario.starting_context.get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
    )
    assert sandbox["content"][-1] == parameters["task"]["user_to_agent"]
    assert parameters["new_content"] in sandbox["content"][-1]

    reminder_target = (
        episode.scenario.evaluation.milestone_matcher.milestones[2]
        .snapshot_constraints[0]
        .target_dataframe
    )
    assert reminder_target is not None
    assert reminder_target["reminder_id"][0] == target_reminder["reminder_id"]
    assert reminder_target["content"][0] == parameters["new_content"]
    assert (
        reminder_target["reminder_timestamp"][0] == parameters["new_reminder_timestamp"]
    )


def test_randomized_reminder_metric_requires_current_target_and_content() -> None:
    episode_1 = _materialize_reminder(1)
    episode_2 = _materialize_reminder(2)
    parameters_1 = episode_1.manifest.parameters
    parameters_2 = episode_2.manifest.parameters

    assert (
        _evaluate_reminder_update(
            episode_2,
            reminder_id=str(parameters_2["target_reminder"]["reminder_id"]),
            content=str(parameters_2["new_content"]),
            reminder_timestamp=float(parameters_2["new_reminder_timestamp"]),
        )
        == 1.0
    )
    assert (
        _evaluate_reminder_update(
            episode_2,
            reminder_id=str(parameters_2["target_reminder"]["reminder_id"]),
            content=str(parameters_1["new_content"]),
            reminder_timestamp=float(parameters_2["new_reminder_timestamp"]),
        )
        < 1.0
    )
    assert (
        _evaluate_reminder_update(
            episode_2,
            reminder_id=str(parameters_1["target_reminder"]["reminder_id"]),
            content=str(parameters_1["new_content"]),
            reminder_timestamp=float(parameters_1["new_reminder_timestamp"]),
        )
        < 1.0
    )


def test_recency_contact_parameterization_varies_selector_target_message_and_phone() -> (
    None
):
    episode_1 = _materialize_recency_contact(1)
    episode_1_repeated = _materialize_recency_contact(1)
    episode_2 = _materialize_recency_contact(2)

    assert episode_1.manifest.to_dict() == episode_1_repeated.manifest.to_dict()
    parameters_1 = episode_1.manifest.parameters
    parameters_2 = episode_2.manifest.parameters
    assert (
        parameters_1["target_contact"]["person_id"]
        != parameters_2["target_contact"]["person_id"]
    )
    assert parameters_1["new_phone_number"] != parameters_2["new_phone_number"]
    assert parameters_1["recency_selector"] != parameters_2["recency_selector"]

    contacts = episode_1.scenario.starting_context.get_database(
        DatabaseNamespace.CONTACT
    )
    self_id = contacts.filter(pl.col("is_self"))["person_id"][0]
    outgoing_messages = (
        episode_1.scenario.starting_context.get_database(DatabaseNamespace.MESSAGING)
        .filter(pl.col("sender_person_id") == self_id)
        .sort("creation_timestamp")
    )
    assert (
        outgoing_messages["recipient_person_id"][-1]
        == parameters_1["target_contact"]["person_id"]
    )
    assert (
        outgoing_messages["message_id"][-1]
        == parameters_1["target_outgoing_message_id"]
    )


def test_recency_contact_task_state_and_metric_targets_share_one_manifest() -> None:
    episode = _materialize_recency_contact(1)
    parameters = episode.manifest.parameters
    sandbox = episode.scenario.starting_context.get_database(
        DatabaseNamespace.SANDBOX,
        get_all_history_snapshots=True,
    )
    assert sandbox["content"][-1] == parameters["task"]["user_to_agent"]

    contact_target = (
        episode.scenario.evaluation.milestone_matcher.milestones[3]
        .snapshot_constraints[0]
        .target_dataframe
    )
    assert contact_target is not None
    assert contact_target["person_id"][0] == parameters["target_contact"]["person_id"]
    assert contact_target["phone_number"][0] == parameters["new_phone_number"]

    response_target = (
        episode.scenario.evaluation.milestone_matcher.milestones[4]
        .snapshot_constraints[0]
        .target_dataframe
    )
    assert response_target is not None
    assert (
        parameters["recency_selector"]["response_subject"]
        in response_target["content"][0]
    )
    assert parameters["new_phone_number"] in response_target["content"][0]


def test_recency_contact_metric_requires_current_message_resolved_target() -> None:
    episode_1 = _materialize_recency_contact(1)
    episode_2 = _materialize_recency_contact(2)
    parameters_1 = episode_1.manifest.parameters
    parameters_2 = episode_2.manifest.parameters

    response_2 = (
        episode_2.scenario.evaluation.milestone_matcher.milestones[4]
        .snapshot_constraints[0]
        .target_dataframe["content"][0]
    )
    assert (
        _evaluate_recency_contact_update(
            episode_2,
            target_contact=dict(parameters_2["target_contact"]),
            new_phone_number=str(parameters_2["new_phone_number"]),
            response=response_2,
        )
        == 1.0
    )

    response_1 = (
        episode_1.scenario.evaluation.milestone_matcher.milestones[4]
        .snapshot_constraints[0]
        .target_dataframe["content"][0]
    )
    assert (
        _evaluate_recency_contact_update(
            episode_2,
            target_contact=dict(parameters_1["target_contact"]),
            new_phone_number=str(parameters_1["new_phone_number"]),
            response=response_1,
        )
        < 1.0
    )
