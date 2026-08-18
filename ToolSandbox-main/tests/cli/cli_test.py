# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import argparse
from pathlib import Path
from typing import Any

import pytest

from tool_sandbox.cli import positive_int, write_result_summary
from tool_sandbox.cli.utils import (
    USER_TYPE_TO_FACTORY,
    RoleImplType,
    get_category_summary,
    resolve_scenarios,
    run_scenario_sequence,
)
from tool_sandbox.common.execution_context import ScenarioCategories
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.conversation_state import MemoryMode, episode_start_tag
from tool_sandbox.roles.end_user import EndUser
from tool_sandbox.scenarios import named_scenarios


def test_getting_all_scenarios() -> None:
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=None, preferred_tool_backend=ToolBackend.DEFAULT
    )
    assert set(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    ) == set(name_to_scenario.keys())


def test_getting_no_scenarios() -> None:
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=[], preferred_tool_backend=ToolBackend.DEFAULT
    )
    assert 0 == len(name_to_scenario)


def test_getting_desired_scenarios() -> None:
    # Pick the first N names from all available scenarios to ensure that this test is
    # using existing scenario names.
    desired_scenario_names = list(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    )[:5]
    name_to_scenario = resolve_scenarios(
        desired_scenario_names=desired_scenario_names,
        preferred_tool_backend=ToolBackend.DEFAULT,
    )
    assert set(desired_scenario_names) == set(name_to_scenario.keys())


def test_getting_non_existent_scenarios() -> None:
    all_scenario_names = set(
        named_scenarios(preferred_tool_backend=ToolBackend.DEFAULT).keys()
    )
    non_existent_scenario = "this scenario does not exist"
    assert non_existent_scenario not in all_scenario_names

    with pytest.raises(KeyError, match="desired scenarios do not exist"):
        resolve_scenarios(
            desired_scenario_names=[non_existent_scenario],
            preferred_tool_backend=ToolBackend.DEFAULT,
        )


def test_positive_int() -> None:
    assert positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
        positive_int("0")


def test_result_summary_creates_legacy_flat_output_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "flat" / "summary"

    write_result_summary([], {}, output_directory)

    assert (output_directory / "result_summary.json").is_file()


def test_end_user_is_available_from_cli_role_mapping() -> None:
    assert RoleImplType("end") is RoleImplType.End
    assert USER_TYPE_TO_FACTORY[RoleImplType.End] is EndUser


def test_full_memory_accumulates_across_sequence_episodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed_memories: list[list[dict[str, Any]]] = []

    def fake_run_scenario_episode(
        name_and_scenario: tuple[str, Scenario],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        observed_memories.append(list(kwargs["cross_episode_memory"]))
        episode_number = kwargs["episode_number"]
        return {"episode_number": episode_number, "turn_count": episode_number}, [
            {"role": "user", "content": f"episode {episode_number}"}
        ]

    monkeypatch.setattr(
        "tool_sandbox.cli.utils.run_scenario_episode", fake_run_scenario_episode
    )

    results = run_scenario_sequence(
        ("scenario", Scenario()),
        agent_type=RoleImplType.GPT_4_o_2024_05_13,
        user_type=RoleImplType.GPT_4_o_2024_05_13,
        output_directory=tmp_path,
        episodes=3,
        memory_mode=MemoryMode.FULL,
    )

    assert [result["episode_number"] for result in results] == [1, 2, 3]
    assert [result["max_steps_per_episode"] for result in results] == [30, 30, 30]
    assert [result["max_steps_per_sequence"] for result in results] == [90, 90, 90]
    assert [result["cumulative_turn_count"] for result in results] == [1, 3, 6]
    assert observed_memories == [
        [],
        [
            {"role": "system", "content": episode_start_tag(1)},
            {"role": "user", "content": "episode 1"},
        ],
        [
            {"role": "system", "content": episode_start_tag(1)},
            {"role": "user", "content": "episode 1"},
            {"role": "system", "content": episode_start_tag(2)},
            {"role": "user", "content": "episode 2"},
        ],
    ]


def test_sequence_materializes_distinct_registered_episode_parameters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed_manifests: list[dict[str, Any]] = []

    def fake_run_scenario_episode(
        name_and_scenario: tuple[str, Scenario],
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        manifest = kwargs["episode_parameter_manifest"]
        observed_manifests.append(manifest.to_dict())
        return {
            "episode_number": kwargs["episode_number"],
            "turn_count": 1,
        }, []

    monkeypatch.setattr(
        "tool_sandbox.cli.utils.run_scenario_episode", fake_run_scenario_episode
    )
    scenario = named_scenarios(ToolBackend.DEFAULT)[
        "update_contact_with_id_and_phone_number"
    ]

    run_scenario_sequence(
        ("update_contact_with_id_and_phone_number", scenario),
        agent_type=RoleImplType.GPT_4_o_2024_05_13,
        user_type=RoleImplType.GPT_4_o_2024_05_13,
        output_directory=tmp_path,
        episodes=3,
        memory_mode=MemoryMode.NONE,
        episode_parameter_seed=23,
    )

    assert all(manifest["is_parameterized"] for manifest in observed_manifests)
    assert (
        len(
            {
                manifest["parameters"]["new_phone_number"]
                for manifest in observed_manifests
            }
        )
        == 3
    )
    assert len({manifest["manifest_id"] for manifest in observed_manifests}) == 3

    observed_manifests.clear()
    run_scenario_sequence(
        ("update_contact_with_id_and_phone_number", scenario),
        agent_type=RoleImplType.GPT_4_o_2024_05_13,
        user_type=RoleImplType.GPT_4_o_2024_05_13,
        output_directory=tmp_path,
        episodes=2,
        memory_mode=MemoryMode.NONE,
        episode_parameter_seed=23,
        parameterize_episodes=False,
    )
    assert not any(manifest["is_parameterized"] for manifest in observed_manifests)


def test_category_summary_averages_all_episodes_and_scenarios() -> None:
    results = [
        {
            "categories": [ScenarioCategories.SINGLE_TOOL_CALL],
            "similarity": 1.0,
            "turn_count": 2,
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        },
        {
            "categories": [ScenarioCategories.SINGLE_TOOL_CALL],
            "similarity": 0.0,
            "turn_count": 4,
            "token_usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
            },
        },
    ]

    summary = get_category_summary(results)

    assert summary["ALL_CATEGORIES"]["similarity"] == [1.0, 0.0]
    assert summary["ALL_CATEGORIES"]["turn_count"] == [2, 4]
    assert summary["ALL_CATEGORIES"]["total_tokens"] == [12, 24]
