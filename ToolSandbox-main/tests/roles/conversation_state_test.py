# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

from tool_sandbox.roles.conversation_state import (
    INDEPENDENT_EPISODE_SYSTEM_PROMPT,
    MemoryMode,
    compose_with_cross_episode_memory,
    episode_start_tag,
    episode_messages_for_memory,
    normalize_token_usage,
    sum_token_usage,
)
from tool_sandbox.roles.deepseek_openrouter_agent import DeepSeekOpenRouterAgent
from tool_sandbox.roles.openai_api_agent import GPT_4_o_2024_05_13_Agent


def test_memory_is_inserted_after_the_independent_episode_boundary() -> None:
    current = [
        {"role": "system", "content": "current system"},
        {"role": "user", "content": "current task"},
    ]
    memory = [
        {"role": "user", "content": "previous task"},
        {"role": "assistant", "content": "previous answer"},
    ]

    assert compose_with_cross_episode_memory(current, memory, episode_number=3) == [
        {
            "role": "system",
            "content": f"{INDEPENDENT_EPISODE_SYSTEM_PROMPT}\n\n{episode_start_tag(3)}",
        },
        current[0],
        *memory,
        current[1],
    ]


def test_system_messages_are_not_persisted_between_episodes() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "answer"},
    ]

    assert episode_messages_for_memory(messages) == messages[1:]


def test_token_usage_normalizes_and_sums_provider_formats() -> None:
    openai_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    openrouter_usage = {"input_tokens": 20, "output_tokens": 5}

    records = [
        normalize_token_usage(openai_usage),
        normalize_token_usage(openrouter_usage),
    ]

    assert sum_token_usage(records) == {
        "prompt_tokens": 30,
        "completion_tokens": 9,
        "total_tokens": 39,
    }


def test_memory_modes_and_supported_agents() -> None:
    assert [mode.value for mode in MemoryMode] == ["none", "full"]
    assert GPT_4_o_2024_05_13_Agent.supports_full_memory
    assert DeepSeekOpenRouterAgent.supports_full_memory
