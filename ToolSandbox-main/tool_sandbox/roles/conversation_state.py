# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Shared helpers for cross-episode conversation memory and token accounting."""

from __future__ import annotations

import copy
from enum import Enum
from typing import Any


class MemoryMode(str, Enum):
    """Cross-episode memory policies supported by the experiment runner."""

    NONE = "none"
    FULL = "full"

    def __str__(self) -> str:
        return self.value


ZERO_TOKEN_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
}


INDEPENDENT_EPISODE_SYSTEM_PROMPT = """This is a new independent episode.

The tool environment and all underlying state have been reset for this episode.
Any prior-episode conversation included below is archived history, not current
state.

Do not assume that prior actions, tool outputs, contact details, IDs, messages,
settings, or completion claims remain true. Do not claim a task is complete
unless you performed or verified the required action in this episode.

You may use prior history only for general procedural lessons, such as tool
usage patterns or recovery strategies."""


def episode_start_tag(episode_number: int) -> str:
    """Return the model-visible delimiter for an episode."""
    return f"-- Start of Episode {episode_number} --"


def independent_episode_system_prompt(episode_number: int | None = None) -> str:
    """Return the reset instruction, optionally labeled with its episode number."""
    if episode_number is None:
        return INDEPENDENT_EPISODE_SYSTEM_PROMPT
    return f"{INDEPENDENT_EPISODE_SYSTEM_PROMPT}\n\n{episode_start_tag(episode_number)}"


def compose_with_cross_episode_memory(
    current_episode_messages: list[dict[str, Any]],
    cross_episode_memory: list[dict[str, Any]],
    *,
    episode_number: int | None = None,
) -> list[dict[str, Any]]:
    """Compose a request with an independent-episode boundary and prior memory.

    The boundary instruction is present for every memory mode. Current episode
    system messages remain at the front of the request after that boundary. Prior
    episode system messages are not persisted because they are identical runner
    instructions rather than conversation history.
    """
    system_messages = [
        copy.deepcopy(message)
        for message in current_episode_messages
        if message.get("role") == "system"
    ]
    episode_messages = [
        copy.deepcopy(message)
        for message in current_episode_messages
        if message.get("role") != "system"
    ]
    return [
        {
            "role": "system",
            "content": independent_episode_system_prompt(episode_number),
        },
        *system_messages,
        *copy.deepcopy(cross_episode_memory),
        *episode_messages,
    ]


def episode_messages_for_memory(
    current_episode_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the agent-visible conversation to retain after one episode."""
    return [
        copy.deepcopy(message)
        for message in current_episode_messages
        if message.get("role") != "system"
    ]


def normalize_token_usage(raw_usage: Any) -> dict[str, int]:
    """Normalize OpenAI- and OpenRouter-style usage objects."""
    if raw_usage is None:
        return dict(ZERO_TOKEN_USAGE)
    if hasattr(raw_usage, "model_dump"):
        raw_usage = raw_usage.model_dump()
    if not isinstance(raw_usage, dict):
        return dict(ZERO_TOKEN_USAGE)

    prompt_tokens = int(
        raw_usage.get("prompt_tokens", raw_usage.get("input_tokens", 0)) or 0
    )
    completion_tokens = int(
        raw_usage.get("completion_tokens", raw_usage.get("output_tokens", 0)) or 0
    )
    total_tokens_value = raw_usage.get("total_tokens")
    total_tokens = (
        int(total_tokens_value)
        if total_tokens_value is not None
        else prompt_tokens + completion_tokens
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def sum_token_usage(records: list[dict[str, int]]) -> dict[str, int]:
    """Sum normalized usage records across all model calls in an episode."""
    return {
        key: sum(record.get(key, 0) for record in records) for key in ZERO_TOKEN_USAGE
    }
