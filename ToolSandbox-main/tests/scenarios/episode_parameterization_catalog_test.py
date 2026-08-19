from __future__ import annotations

import copy
import datetime
import json
import re
from pathlib import Path
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
from tool_sandbox.tools.contact import add_contact, modify_contact, remove_contact


def _catalogue_names() -> tuple[str, ...]:
    table = Path(__file__).parents[2] / "episode_parameterization_table.md"
    return tuple(re.findall(r"^\| `([^`]+)` \|", table.read_text(), flags=re.MULTILINE))


CATALOGUE_NAMES = _catalogue_names()
SCENARIOS = named_scenarios(ToolBackend.DEFAULT)


def _materialize(
    scenario_name: str, episode_number: int = 1, seed: int = 41
) -> MaterializedEpisode:
    return DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=scenario_name,
        scenario=SCENARIOS[scenario_name],
        episode_number=episode_number,
        seed=seed,
    )


def _task_text(scenario: Scenario) -> str:
    return "\n".join(
        str(value)
        for value in scenario.starting_context.get_database(
            DatabaseNamespace.SANDBOX, get_all_history_snapshots=True
        )["content"].to_list()
        if value is not None
    )


def _target_rows(
    episode: MaterializedEpisode,
    namespace: DatabaseNamespace,
    *,
    minefield: bool = False,
) -> list[dict[str, Any]]:
    matcher = (
        episode.scenario.evaluation.minefield_matcher
        if minefield
        else episode.scenario.evaluation.milestone_matcher
    )
    return [
        row
        for node in matcher.milestones
        for constraint in node.snapshot_constraints
        if constraint.database_namespace == namespace
        and constraint.target_dataframe is not None
        for row in constraint.target_dataframe.to_dicts()
    ]


def _add_sandbox_row(
    context: Any,
    *,
    sender: RoleType,
    recipient: RoleType,
    content: str | None = None,
    tool_trace: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {"sender": sender, "recipient": recipient}
    if content is not None:
        row["content"] = content
    if tool_trace is not None:
        row["tool_trace"] = [json.dumps(tool_trace, ensure_ascii=False)]
    context.add_to_database(DatabaseNamespace.SANDBOX, [row])


def test_catalogue_and_registry_cover_exactly_all_40_audited_scenarios() -> None:
    assert len(CATALOGUE_NAMES) == 40
    assert len(set(CATALOGUE_NAMES)) == 40
    assert set(DEFAULT_EPISODE_RANDOMIZER._parameterizers) == set(CATALOGUE_NAMES)


@pytest.mark.parametrize("scenario_name", CATALOGUE_NAMES)  # type: ignore[untyped-decorator]
def test_every_catalogue_scenario_is_deterministic_varied_and_replayable(
    scenario_name: str,
) -> None:
    original = SCENARIOS[scenario_name]
    original_setup = episode_setup_record(
        scenario=original,
        manifest=DEFAULT_EPISODE_RANDOMIZER.identity_manifest(
            scenario_name=scenario_name, episode_number=1, seed=41
        ),
    )
    episode_1 = _materialize(scenario_name, 1)
    repeated = _materialize(scenario_name, 1)
    episode_2 = _materialize(scenario_name, 2)

    assert episode_1.manifest.is_parameterized
    assert episode_1.manifest.to_dict() == repeated.manifest.to_dict()
    assert episode_1.manifest.parameters != episode_2.manifest.parameters
    assert episode_1.scenario is not original
    assert json.dumps(
        episode_setup_record(scenario=episode_1.scenario, manifest=episode_1.manifest)
    )

    unchanged_setup = episode_setup_record(
        scenario=original,
        manifest=DEFAULT_EPISODE_RANDOMIZER.identity_manifest(
            scenario_name=scenario_name, episode_number=1, seed=41
        ),
    )
    assert unchanged_setup == original_setup


@pytest.mark.parametrize("scenario_name", CATALOGUE_NAMES)  # type: ignore[untyped-decorator]
def test_generated_distraction_variant_uses_base_parameterizer(
    scenario_name: str,
) -> None:
    variant_name = f"{scenario_name}_3_distraction_tools"
    assert variant_name in SCENARIOS
    episode = DEFAULT_EPISODE_RANDOMIZER.materialize(
        scenario_name=variant_name,
        scenario=SCENARIOS[variant_name],
        episode_number=2,
        seed=41,
    )
    assert episode.manifest.is_parameterized
    assert episode.manifest.base_scenario_name == scenario_name


def test_literal_prompt_state_and_metric_targets_use_one_manifest() -> None:
    send = _materialize("send_message_with_phone_number_and_content")
    phone = str(send.manifest.parameters["recipient_phone_number"])
    content = str(send.manifest.parameters["content"])
    assert phone in _task_text(send.scenario)
    assert content in _task_text(send.scenario)
    message_targets = _target_rows(send, DatabaseNamespace.MESSAGING)
    assert {"recipient_phone_number": phone, "content": content} in message_targets
    assert any(
        phone in row["content"] and content in row["content"]
        for row in _target_rows(send, DatabaseNamespace.SANDBOX)
        if row.get("content")
    )

    contact_search = _materialize("search_name_with_relationship")
    target = dict(contact_search.manifest.parameters["target_contact"])
    relationship = str(contact_search.manifest.parameters["relationship"])
    matches = contact_search.scenario.starting_context.get_database(
        DatabaseNamespace.CONTACT
    ).filter(pl.col("relationship") == relationship)
    assert matches["person_id"].to_list() == [target["person_id"]]
    traces = [
        json.loads(row["tool_trace"])
        for row in _target_rows(contact_search, DatabaseNamespace.SANDBOX)
        if row.get("tool_trace")
    ]
    assert {
        "tool_name": "search_contacts",
        "arguments": {"relationship": relationship},
    } in traces


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "scenario_name",
    (
        "update_contact_relationship_with_relationship",
        "update_contact_relationship_with_relationship_multiple_user_turn",
        "update_contact_relationship_with_relationship_twice_multiple_user_turn",
    ),
)
def test_relationship_templates_synchronize_complete_contact_set(
    scenario_name: str,
) -> None:
    episode = _materialize(scenario_name)
    contacts = [dict(value) for value in episode.manifest.parameters["target_contacts"]]
    source = str(episode.manifest.parameters["source_relationship"])
    destination = str(episode.manifest.parameters["destination_relationship"])
    ids = {contact["person_id"] for contact in contacts}
    starting = episode.scenario.starting_context.get_database(DatabaseNamespace.CONTACT)
    assert (
        set(starting.filter(pl.col("relationship") == source)["person_id"].to_list())
        == ids
    )

    targets = _target_rows(episode, DatabaseNamespace.CONTACT)
    destination_rows = [row for row in targets if row["relationship"] == destination]
    assert {row["person_id"] for row in destination_rows} == ids
    assert len(destination_rows) == len(ids)


def test_holiday_manifest_uses_future_date_and_same_explicit_tool_year() -> None:
    episode = _materialize("find_days_till_holiday_multiple_user_turn", 2)
    holiday = str(episode.manifest.parameters["holiday_name"])
    year = int(episode.manifest.parameters["year"])
    date = datetime.date.fromisoformat(str(episode.manifest.parameters["date"]))
    assert date >= datetime.datetime.now().astimezone().date()
    assert holiday in _task_text(episode.scenario)
    assert str(year) in _task_text(episode.scenario)

    search_calls: list[dict[str, Any]] = []
    for row in _target_rows(episode, DatabaseNamespace.SANDBOX):
        if not row.get("tool_trace"):
            continue
        trace = json.loads(row["tool_trace"])
        values = trace if isinstance(trace, list) else [trace]
        search_calls.extend(
            value for value in values if value.get("tool_name") == "search_holiday"
        )
    assert search_calls
    assert all(
        call["arguments"] == {"holiday_name": holiday, "year": year}
        for call in search_calls
    )


def test_randomized_addition_metric_accepts_materialized_target() -> None:
    episode = _materialize("add_contact_with_name_and_phone_number")
    parameters = episode.manifest.parameters
    context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(context)
    add_contact(
        name=str(parameters["name"]),
        phone_number=str(parameters["phone_number"]),
        relationship=None,
    )
    response = _target_rows(episode, DatabaseNamespace.SANDBOX)[0]["content"]
    _add_sandbox_row(
        context,
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content=response,
    )
    assert (
        episode.scenario.evaluation.evaluate(
            execution_context=context,
            max_turn_count=episode.scenario.max_messages,
        ).similarity
        == 1.0
    )


def test_randomized_removal_metric_uses_materialized_pre_episode_reference() -> None:
    episode = _materialize("remove_contact_with_id")
    target = dict(episode.manifest.parameters["target_contact"])
    context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(context)
    remove_contact(person_id=str(target["person_id"]))
    response = _target_rows(episode, DatabaseNamespace.SANDBOX)[0]["content"]
    _add_sandbox_row(
        context,
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content=response,
    )
    assert (
        episode.scenario.evaluation.evaluate(
            execution_context=context,
            max_turn_count=episode.scenario.max_messages,
        ).similarity
        == 1.0
    )


def test_randomized_multi_contact_metric_accepts_only_complete_target_set() -> None:
    episode = _materialize("update_contact_relationship_with_relationship")
    contacts = [dict(value) for value in episode.manifest.parameters["target_contacts"]]
    source = str(episode.manifest.parameters["source_relationship"])
    destination = str(episode.manifest.parameters["destination_relationship"])
    context = copy.deepcopy(episode.scenario.starting_context)
    set_current_context(context)
    _add_sandbox_row(
        context,
        sender=RoleType.EXECUTION_ENVIRONMENT,
        recipient=RoleType.AGENT,
        tool_trace={
            "tool_name": "search_contacts",
            "arguments": {"relationship": source},
        },
    )
    for contact in contacts:
        modify_contact(person_id=str(contact["person_id"]), relationship=destination)
    response = _target_rows(episode, DatabaseNamespace.SANDBOX)[1]["content"]
    _add_sandbox_row(
        context,
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content=response,
    )
    assert (
        episode.scenario.evaluation.evaluate(
            execution_context=context,
            max_turn_count=episode.scenario.max_messages,
        ).similarity
        == 1.0
    )


def test_insufficient_information_keeps_capability_boundary_and_minefield() -> None:
    episode = _materialize(
        "remove_contact_by_phone_no_search_contacts_insufficient_information", 2
    )
    configuration = episode.scenario.starting_context
    assert set(configuration.tool_allow_list or ()) == {
        "end_conversation",
        "remove_contact",
    }
    assert set(configuration.tool_deny_list or ()) == {
        "search_contacts",
        "search_messages",
    }
    minefield_rows = _target_rows(episode, DatabaseNamespace.SANDBOX, minefield=True)
    assert any("remove_contact" in row["content"] for row in minefield_rows)

    safe_context = copy.deepcopy(episode.scenario.starting_context)
    _add_sandbox_row(
        safe_context,
        sender=RoleType.AGENT,
        recipient=RoleType.USER,
        content="I cannot identify the contact without a lookup tool.",
    )
    safe_result = episode.scenario.evaluation.evaluate(
        execution_context=safe_context,
        max_turn_count=episode.scenario.max_messages,
    )

    unsafe_context = copy.deepcopy(episode.scenario.starting_context)
    _add_sandbox_row(
        unsafe_context,
        sender=RoleType.AGENT,
        recipient=RoleType.EXECUTION_ENVIRONMENT,
        content="remove_contact",
    )
    unsafe_result = episode.scenario.evaluation.evaluate(
        execution_context=unsafe_context,
        max_turn_count=episode.scenario.max_messages,
    )
    assert safe_result.similarity == 1.0
    assert unsafe_result.minefield_similarity > 0
    assert unsafe_result.similarity == 0.0
