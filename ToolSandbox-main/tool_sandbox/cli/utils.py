# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
import json
import traceback
from collections import Counter, defaultdict
from enum import auto
from pathlib import Path
from typing import Any, Callable, Optional, Union

from strenum import StrEnum

from tool_sandbox.common.execution_context import (
    RoleType,
    ScenarioCategories,
    get_current_context,
)
from tool_sandbox.common.scenario import Scenario
from tool_sandbox.common.tool_discovery import ToolBackend
from tool_sandbox.roles.anthropic_api_agent import (
    ClaudeHaikuAgent,
    ClaudeOpusAgent,
    ClaudeSonnetAgent,
)
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.cli_role import CliAgent, CliUser
from tool_sandbox.roles.cohere_agent import CohereAgent
from tool_sandbox.roles.conversation_state import (
    MemoryMode,
    episode_start_tag,
    sum_token_usage,
)
from tool_sandbox.roles.deepseek_openrouter_agent import DeepSeekOpenRouterAgent
from tool_sandbox.roles.end_user import EndUser
from tool_sandbox.roles.execution_environment import ExecutionEnvironment
from tool_sandbox.roles.gemini_agent import GeminiAgent
from tool_sandbox.roles.gorilla_api_agent import GorillaAPIAgent
from tool_sandbox.roles.hermes_api_agent import HermesAPIAgent
from tool_sandbox.roles.mistral_api_agent import MistralOpenAIServerAgent
from tool_sandbox.roles.openai_api_agent import (
    GPT_3_5_0125_Agent,
    GPT_4_0125_Agent,
    GPT_4_o_2024_05_13_Agent,
)
from tool_sandbox.roles.openai_api_user import (
    GPT_3_5_0125_User,
    GPT_4_0125_User,
    GPT_4_o_2024_05_13_User,
)
from tool_sandbox.roles.unhelpful_agent import UnhelpfulAgent
from tool_sandbox.scenarios import named_scenarios
from tool_sandbox.scenarios.episode_parameterization import (
    DEFAULT_EPISODE_RANDOMIZER,
    EpisodeParameterManifest,
    base_scenario_name,
    episode_setup_record,
)


class RoleImplType(StrEnum):
    Hermes = auto()
    Gorilla = auto()
    GPT_3_5_0125 = auto()
    GPT_4_0125 = auto()
    GPT_4_o_2024_05_13 = auto()
    Claude_3_Opus = auto()
    Claude_3_Sonnet = auto()
    Claude_3_Haiku = auto()
    Gemini_1_0 = auto()
    Gemini_1_5 = auto()
    Gemini_1_5_Flash = auto()
    Cli = auto()
    Deterministic = auto()
    MistralOpenAIServer = auto()
    Cohere_Command_R = auto()
    Cohere_Command_R_Plus = auto()
    DeepSeek_V4_Flash_0731 = auto()
    Unhelpful = auto()
    End = "end"


AGENT_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.Hermes: lambda: HermesAPIAgent(
        model_name="NousResearch/Hermes-2-Pro-Mistral-7B"
    ),
    RoleImplType.Gorilla: lambda: GorillaAPIAgent(
        model_name="gorilla-llm/gorilla-openfunctions-v2"
    ),
    RoleImplType.MistralOpenAIServer: lambda: MistralOpenAIServerAgent(
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    ),
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_Agent,
    RoleImplType.GPT_4_0125: GPT_4_0125_Agent,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_Agent,
    RoleImplType.Claude_3_Opus: ClaudeOpusAgent,
    RoleImplType.Claude_3_Sonnet: ClaudeSonnetAgent,
    RoleImplType.Claude_3_Haiku: ClaudeHaikuAgent,
    RoleImplType.Gemini_1_0: lambda: GeminiAgent(model_name="gemini-1.0-pro"),
    RoleImplType.Gemini_1_5: lambda: GeminiAgent(model_name="gemini-1.5-pro-001"),
    RoleImplType.Gemini_1_5_Flash: lambda: GeminiAgent(
        model_name="gemini-1.5-flash-001"
    ),
    RoleImplType.Cli: CliAgent,
    RoleImplType.Cohere_Command_R: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-v01"
    ),
    RoleImplType.Cohere_Command_R_Plus: lambda: CohereAgent(
        model_name="CohereForAI/c4ai-command-r-plus"
    ),
    RoleImplType.DeepSeek_V4_Flash_0731: DeepSeekOpenRouterAgent,
    RoleImplType.Unhelpful: UnhelpfulAgent,
}

USER_TYPE_TO_FACTORY: dict[RoleImplType, Callable[..., BaseRole]] = {
    RoleImplType.GPT_3_5_0125: GPT_3_5_0125_User,
    RoleImplType.GPT_4_0125: GPT_4_0125_User,
    RoleImplType.GPT_4_o_2024_05_13: GPT_4_o_2024_05_13_User,
    RoleImplType.Cli: CliUser,
    RoleImplType.End: EndUser,
}

# The scenarios to play back when the `--test_mode` flag is set.
TEST_SCENARIO_NAMES = [
    "send_message_with_contact_content_cellular_off_multiple_user_turn",
    "send_message_with_contact_content_cellular_off_multiple_user_turn_10_distraction_tools",
    "send_message_with_contact_content_cellular_off_3_distraction_tools_arg_description_scrambled",
    # "remove_contact_by_phone_multiple_user_turn",
    # "find_temperature_f_with_location_and_time_diff_multiple_user_turn",
]


def generate_hierarchical_output_path(
    base_output_dir: Path,
    agent_name: str,
    scenario_name: str,
    episodes: int,
    model_name: str,
    memory_mode: MemoryMode,
    timestamp_str: str,
) -> Path:
    """Generate hierarchical output path for experiment logs.

    Structure:
    base_output_dir/agent_name/base_scenario_name/modification/episode_N/model_memory_timestamp/

    Args:
        base_output_dir:  Base output directory (typically 'data')
        agent_name:       Agent type name (e.g., 'deepseek')
        scenario_name:    Full scenario name (e.g., 'modify_contact_with_message_recency_3_distraction_tools')
        episodes:         Number of episodes
        model_name:       Model name (e.g., 'deepseek-v4-flash-0731')
        memory_mode:      Memory mode (FULL or NONE)
        timestamp_str:    Timestamp in format YYMMDDHHMMSS

    Returns:
        Path object for the hierarchical output directory
    """
    base_name = base_scenario_name(scenario_name)
    modification = scenario_name.replace(base_name, "").lstrip("_")
    if not modification:
        modification = "base"

    # Format: model_memory_timestamp
    memory_mode_str = memory_mode.value if hasattr(memory_mode, 'value') else str(memory_mode)
    model_memory_timestamp = f"{model_name}_memory_{memory_mode_str}_{timestamp_str}"

    # Build path components explicitly to avoid any Path construction issues
    path_components = [
        str(base_output_dir),
        f"agent_{agent_name}",
        base_name,
        modification,
        f"episode_{episodes}",
        model_memory_timestamp,
    ]
    
    # Construct path by joining all components
    output_path = Path(*path_components)

    return output_path


def resolve_scenarios(
    desired_scenario_names: Optional[list[str]],
    preferred_tool_backend: ToolBackend,
) -> dict[str, Scenario]:
    """Resolve the scenarios to run.

    Args:
        desired_scenario_names: Name of scenarios to run. If empty all scenarios will be
                                returned.
        preferred_tool_backend: Which backend should be chosen in face of conflicting tool names.

    Returns:
        Dictionary from scenario name to definition.
    """
    if desired_scenario_names is None:
        # No filtering needed. Return all scenarios.
        return named_scenarios(preferred_tool_backend=preferred_tool_backend)

    name_to_scenario = {
        name: scenario
        for name, scenario in named_scenarios(
            preferred_tool_backend=preferred_tool_backend
        ).items()
        if name in desired_scenario_names
    }

    # Raise an exception if not all desired scenarios exist, e.g. to fail if there was a
    # typo in the scenario names of the CLI command.
    if len(desired_scenario_names) != len(name_to_scenario):
        missing_scenarios = set(desired_scenario_names) - set(name_to_scenario.keys())
        raise KeyError(
            "The following desired scenarios do not exist: "
            f"{sorted(list(missing_scenarios))}"
        )
    return name_to_scenario


def run_scenario(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
) -> dict[str, Any]:
    """Play and evaluate one episode with the original no-memory behavior.

    This is a necessary utility function to make multiprocessing work.

    Args:
        name_and_scenario:              Scenario name and Scenario object.
        agent_type:                     Agent type.
        user_type:                      User type.
        output_directory:               Directory to write output into.

    Returns:
        Evaluation info
    """
    summary, _ = run_scenario_episode(
        name_and_scenario,
        agent_type=agent_type,
        user_type=user_type,
        output_directory=output_directory,
        episode_number=1,
        memory_mode=MemoryMode.NONE,
        cross_episode_memory=[],
    )
    return summary


def run_scenario_episode(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
    episode_number: int,
    memory_mode: MemoryMode,
    cross_episode_memory: list[dict[str, Any]],
    episode_parameter_manifest: Optional[EpisodeParameterManifest] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Play one reset episode and return its score plus newly produced memory."""
    name, scenario = name_and_scenario
    if episode_parameter_manifest is None:
        episode_parameter_manifest = DEFAULT_EPISODE_RANDOMIZER.identity_manifest(
            scenario_name=name,
            episode_number=episode_number,
            seed=0,
        )
    roles = {
        RoleType.USER: USER_TYPE_TO_FACTORY[user_type](),
        RoleType.EXECUTION_ENVIRONMENT: ExecutionEnvironment(),
        RoleType.AGENT: AGENT_TYPE_TO_FACTORY[agent_type](),
    }
    agent = roles[RoleType.AGENT]
    try:
        output_directory.mkdir(parents=True, exist_ok=True)
        agent.set_episode_number(episode_number)
        if memory_mode == MemoryMode.FULL:
            if not agent.supports_full_memory:
                raise ValueError(
                    f"Agent {agent_type} does not support --memory full. "
                    "Use an OpenAI or OpenRouter agent."
                )
            agent.set_cross_episode_memory(cross_episode_memory)

        trajectory_path = Path("trajectories") / name / f"episode_{episode_number:04d}"
        episode_output_directory = output_directory / trajectory_path
        episode_output_directory.mkdir(parents=True, exist_ok=True)
        episode_setup_path = trajectory_path / "episode_setup.json"
        with open(
            output_directory / episode_setup_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                episode_setup_record(
                    scenario=scenario,
                    manifest=episode_parameter_manifest,
                ),
                f,
                indent=2,
                ensure_ascii=False,
            )
        with open(
            episode_output_directory / "memory_before_episode.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cross_episode_memory, f, indent=2, ensure_ascii=False)

        try:
            result = scenario.play_and_evaluate(
                roles=roles,
                output_directory=output_directory,
                scenario_name=str(Path(name) / f"episode_{episode_number:04d}"),
                episode_number=episode_number,
            )
            summary = {
                "name": name,
                "episode_number": episode_number,
                "memory_mode": str(memory_mode),
                "memory_message_count": len(cross_episode_memory),
                "trajectory_path": str(trajectory_path),
                "episode_setup_path": str(episode_setup_path),
                "parameter_manifest_id": episode_parameter_manifest.manifest_id,
                "parameterized": episode_parameter_manifest.is_parameterized,
                "parameter_manifest": episode_parameter_manifest.to_dict(),
                "categories": scenario.categories,
                "traceback": None,
                "exception_type": None,
                "milestone_similarity": result.evaluation_result.milestone_similarity,
                "minefield_similarity": result.evaluation_result.minefield_similarity,
                "similarity": result.evaluation_result.similarity,
                "turn_count": result.evaluation_result.turn_count,
                "milestone_mapping": result.evaluation_result.milestone_mapping,
                "minefield_mapping": result.evaluation_result.minefield_mapping,
            }
        except Exception as e:
            summary = {
                "name": name,
                "episode_number": episode_number,
                "memory_mode": str(memory_mode),
                "memory_message_count": len(cross_episode_memory),
                "trajectory_path": str(trajectory_path),
                "episode_setup_path": str(episode_setup_path),
                "parameter_manifest_id": episode_parameter_manifest.manifest_id,
                "parameterized": episode_parameter_manifest.is_parameterized,
                "parameter_manifest": episode_parameter_manifest.to_dict(),
                "categories": scenario.categories,
                "traceback": traceback.format_exc(),
                "exception_type": type(e).__name__,
                "milestone_similarity": 0,
                "minefield_similarity": 0,
                "similarity": 0,
                "turn_count": scenario.max_messages,
                "milestone_mapping": {},
                "minefield_mapping": {},
            }
        token_usage_records = agent.get_token_usage_records()
        summary["token_usage"] = sum_token_usage(token_usage_records)

        episode_memory = (
            agent.export_episode_memory() if memory_mode == MemoryMode.FULL else []
        )
        return summary, episode_memory
    finally:
        for role in roles.values():
            role.teardown()


def run_scenario_sequence(
    name_and_scenario: tuple[str, Scenario],
    *,
    agent_type: RoleImplType,
    user_type: RoleImplType,
    output_directory: Path,
    episodes: int,
    memory_mode: MemoryMode,
    episode_parameter_seed: int = 0,
    parameterize_episodes: bool = True,
    agent_name: Optional[str] = None,
    model_name: Optional[str] = None,
    timestamp_str: Optional[str] = None,
    base_output_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Run one scenario repeatedly with reset state and optional full memory.

    Args:
        name_and_scenario: Scenario name and Scenario object
        agent_type: Agent type
        user_type: User type
        output_directory: Directory to write output into (legacy parameter, used if hierarchical params not provided)
        episodes: Number of episodes
        memory_mode: Memory mode
        episode_parameter_seed: Seed for episode parameterization
        parameterize_episodes: Whether to parameterize episodes
        agent_name: Agent name for hierarchical path (e.g., 'deepseek')
        model_name: Model name for hierarchical path (e.g., 'deepseek-v4-flash-0731')
        timestamp_str: Timestamp for hierarchical path (format: YYMMDDHHMMSS)
        base_output_dir: Base output directory for hierarchical structure
    """
    _, scenario = name_and_scenario
    cross_episode_memory: list[dict[str, Any]] = []
    sequence_results: list[dict[str, Any]] = []
    max_steps_per_sequence = scenario.max_messages * episodes
    cumulative_turn_count = 0

    # Generate hierarchical output directory if metadata is provided
    scenario_output_directory = output_directory
    if all([agent_name, model_name, timestamp_str, base_output_dir]):
        scenario_output_directory = generate_hierarchical_output_path(
            base_output_dir=base_output_dir,
            agent_name=agent_name,
            scenario_name=name_and_scenario[0],
            episodes=episodes,
            model_name=model_name,
            memory_mode=memory_mode,
            timestamp_str=timestamp_str,
        )
        # Log the generated path for debugging
        print(f"[DEBUG] Using hierarchical path: {scenario_output_directory}", flush=True)
    else:
        # Log when hierarchical path is NOT used
        print(
            f"[WARNING] Hierarchical path parameters incomplete. Using legacy path instead.\n"
            f"  agent_name={agent_name}, model_name={model_name}, "
            f"timestamp_str={timestamp_str}, base_output_dir={base_output_dir}",
            flush=True
        )

    for episode_number in range(1, episodes + 1):
        materialized_episode = DEFAULT_EPISODE_RANDOMIZER.materialize(
            scenario_name=name_and_scenario[0],
            scenario=scenario,
            episode_number=episode_number,
            seed=episode_parameter_seed,
            # Preserve the historical single-episode task exactly. Multi-episode
            # runs parameterize only registered scenarios; all others are identity.
            enabled=parameterize_episodes and episodes > 1,
        )
        summary, episode_memory = run_scenario_episode(
            (name_and_scenario[0], materialized_episode.scenario),
            agent_type=agent_type,
            user_type=user_type,
            output_directory=scenario_output_directory,
            episode_number=episode_number,
            memory_mode=memory_mode,
            cross_episode_memory=cross_episode_memory,
            episode_parameter_manifest=materialized_episode.manifest,
        )
        cumulative_turn_count += summary["turn_count"]
        summary["max_steps_per_episode"] = scenario.max_messages
        summary["max_steps_per_sequence"] = max_steps_per_sequence
        summary["cumulative_turn_count"] = cumulative_turn_count
        sequence_results.append(summary)
        if memory_mode == MemoryMode.FULL:
            cross_episode_memory.extend(
                [
                    {
                        "role": "system",
                        "content": episode_start_tag(episode_number),
                    },
                    *episode_memory,
                ]
            )
    return sequence_results


def get_category_summary(
    result_summary: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per test case result summary into category wise summary.

    Args:
        result_summary:     A list of results for each test case.

    Returns:
        Category wise summary.
    """
    # Aggregate results by category
    category_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for current_summary in result_summary:
        for category in current_summary["categories"]:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                current_summary["categories"]
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_summary[category]["similarity"].append(
                current_summary["similarity"]
            )
            category_summary[category]["turn_count"].append(
                current_summary["turn_count"]
            )
            for token_key, value in current_summary["token_usage"].items():
                category_summary[category][token_key].append(value)
        category_summary["ALL_CATEGORIES"]["similarity"].append(
            current_summary["similarity"]
        )
        category_summary["ALL_CATEGORIES"]["turn_count"].append(
            current_summary["turn_count"]
        )
        for token_key, value in current_summary["token_usage"].items():
            category_summary["ALL_CATEGORIES"][token_key].append(value)
    return category_summary


def get_category_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on ScenarioCategories.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each category.
    """
    category_counter: Counter[Union[ScenarioCategories, str]] = Counter()
    for scenario in name_to_scenario.values():
        for category in scenario.categories:
            # The augmented scenarios are based on top of the `THREE_DISTRACTION_TOOLS`,
            # but we do not want to double count the stats for `THREE_DISTRACTION_TOOLS`.
            # Otherwise it would not be comparable to e.g. `TEN_DISTRACTION_TOOLS`.
            if category == ScenarioCategories.THREE_DISTRACTION_TOOLS and set(
                scenario.categories
            ) & {
                ScenarioCategories.TOOL_NAME_SCRAMBLED,
                ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
                ScenarioCategories.ARG_TYPE_SCRAMBLED,
                ScenarioCategories.ARG_NAME_SCRAMBLED,
            }:
                continue
            category_counter[category] += 1
        category_counter["ALL_CATEGORIES"] += 1
    return category_counter


def get_necessary_tool_name_to_scenario_count(
    name_to_scenario: dict[str, Scenario],
) -> Counter[Union[ScenarioCategories, str]]:
    """Count number of scenarios based on necessary tool names.

    Args:
        name_to_scenario:   A dict with scenario name as keys, scenario objects as values.

    Returns:
        A counter object containing counts for each necessary tool names.
    """
    tool_name_counter: Counter[Union[ScenarioCategories, str]] = Counter(
        {
            tool_name: 0
            for tool_name in get_current_context().get_available_tools(
                scrambling_allowed=False
            )
        }
    )
    # Necessary tool names can be deducted from allowed tools in NO_DISTRACTION_TOOLS category
    # Then the total count equals the count from this category * number of augmentations.
    augmentation_categories: set[Union[ScenarioCategories, str]] = set()
    for scenario in name_to_scenario.values():
        if ScenarioCategories.NO_DISTRACTION_TOOLS in scenario.categories:
            assert scenario.starting_context.tool_allow_list is not None
            for necessary_tool in scenario.starting_context.tool_allow_list:
                tool_name_counter[necessary_tool] += 1
        augmentation_categories |= {
            ScenarioCategories.NO_DISTRACTION_TOOLS,
            ScenarioCategories.THREE_DISTRACTION_TOOLS,
            ScenarioCategories.TEN_DISTRACTION_TOOLS,
            ScenarioCategories.ALL_TOOLS_AVAILABLE,
            ScenarioCategories.TOOL_NAME_SCRAMBLED,
            ScenarioCategories.TOOL_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_DESCRIPTION_SCRAMBLED,
            ScenarioCategories.ARG_TYPE_SCRAMBLED,
            ScenarioCategories.ARG_NAME_SCRAMBLED,
        } & set(scenario.categories)
    for necessary_tool in tool_name_counter:
        tool_name_counter[necessary_tool] *= len(augmentation_categories)
    return tool_name_counter
