# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""DeepSeek agent role backed by the OpenRouter chat completions API."""

from __future__ import annotations

import copy
import json
import os
from typing import Any, cast

import requests
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from requests.exceptions import RequestException
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from tool_sandbox.common.execution_context import RoleType, get_current_context
from tool_sandbox.common.message_conversion import (
    Message,
    openai_tool_call_to_python_code,
    to_openai_messages,
)
from tool_sandbox.common.tool_conversion import convert_to_openai_tools
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.conversation_state import (
    compose_with_cross_episode_memory,
    episode_messages_for_memory,
    normalize_token_usage,
)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash-0731"
OPENROUTER_REQUEST_TIMEOUT_SECONDS = 120


@retry(
    wait=wait_random_exponential(multiplier=1, max=40),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(RequestException),
)
def request_openrouter_chat_completion(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model_name: str = DEFAULT_DEEPSEEK_MODEL,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Send a reasoning-enabled chat completion request to OpenRouter.

    Args:
        messages: OpenAI-compatible chat messages.
        tools: Optional OpenAI-compatible function tool definitions.
        model_name: OpenRouter model identifier.
        api_key: Optional explicit OpenRouter API key. If omitted, the
                 ``OPENROUTER_API_KEY`` environment variable is used.

    Returns:
        The decoded OpenRouter response body.

    Raises:
        ValueError: If no OpenRouter API key is configured or the response does not
                    contain an assistant message.
        TypeError: If the assistant message has an invalid type.
        requests.exceptions.RequestException: If the HTTP request fails.
    """
    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "The `OPENROUTER_API_KEY` environment variable must be set to use "
            "the DeepSeek OpenRouter agent."
        )

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "reasoning": {"enabled": True},
    }
    if tools is not None:
        payload["tools"] = tools

    response = requests.post(
        url=OPENROUTER_CHAT_COMPLETIONS_URL,
        headers={
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    response_body = response.json()
    try:
        assistant_message = response_body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "OpenRouter response did not contain choices[0].message."
        ) from error
    if not isinstance(assistant_message, dict):
        raise TypeError("OpenRouter choices[0].message must be an object.")
    return cast(dict[str, Any], response_body)


def restore_assistant_messages(
    messages: list[dict[str, Any]],
    assistant_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore prior raw assistant messages, including ``reasoning_details``.

    ToolSandbox stores assistant tool calls in its own message representation and
    reconstructs them before each API call. OpenRouter reasoning models require the
    original assistant ``reasoning_details`` blocks to be passed back unmodified. The
    saved messages correspond to the most recent assistant messages, so suffix
    alignment also supports contexts that contained assistant messages before this
    agent instance was created.
    """
    restored_messages = copy.deepcopy(messages)
    if not assistant_messages:
        return restored_messages

    assistant_indices = [
        index
        for index, message in enumerate(restored_messages)
        if message.get("role") == "assistant"
    ]
    if len(assistant_indices) < len(assistant_messages):
        raise ValueError(
            "Cannot restore OpenRouter assistant history: more saved responses than "
            "assistant messages are present in the ToolSandbox context."
        )

    for index, assistant_message in zip(
        assistant_indices[-len(assistant_messages) :],
        assistant_messages,
    ):
        restored_messages[index] = copy.deepcopy(assistant_message)
    return restored_messages


def response_message_to_tool_sandbox_messages(
    response_message: dict[str, Any],
    *,
    available_tool_names: set[str],
    agent_to_execution_facing_tool_name: dict[str, str],
) -> list[Message]:
    """Convert an OpenRouter assistant message to ToolSandbox messages."""
    tool_calls = response_message.get("tool_calls")
    if not tool_calls:
        content = response_message.get("content")
        if not isinstance(content, str):
            raise ValueError(
                "OpenRouter assistant response without tool calls must contain text."
            )
        return [
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.USER,
                content=content,
            )
        ]

    if not isinstance(tool_calls, list):
        raise TypeError("OpenRouter assistant message tool_calls must be a list.")

    response_messages: list[Message] = []
    for raw_tool_call in tool_calls:
        tool_call = ChatCompletionMessageToolCall.model_validate(raw_tool_call)
        agent_facing_tool_name = tool_call.function.name
        try:
            execution_facing_tool_name = agent_to_execution_facing_tool_name[
                agent_facing_tool_name
            ]
        except KeyError as error:
            raise KeyError(
                f"OpenRouter requested unknown tool {agent_facing_tool_name!r}."
            ) from error
        response_messages.append(
            Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=openai_tool_call_to_python_code(
                    tool_call,
                    available_tool_names,
                    execution_facing_tool_name=execution_facing_tool_name,
                ),
                openai_tool_call_id=tool_call.id,
                openai_function_name=agent_facing_tool_name,
            )
        )
    return response_messages


class DeepSeekOpenRouterAgent(BaseRole):
    """DeepSeek tool-use agent accessed through OpenRouter's HTTP API."""

    role_type: RoleType = RoleType.AGENT
    supports_full_memory = True

    def __init__(self, model_name: str = DEFAULT_DEEPSEEK_MODEL) -> None:
        self.model_name = model_name
        self.reset()

    def reset(self) -> None:
        """Clear API-only message fields that ToolSandbox does not persist."""
        self._assistant_messages: list[dict[str, Any]] = []
        self._cross_episode_memory: list[dict[str, Any]] = []
        self._episode_number: int | None = None
        self._token_usage_records: list[dict[str, int]] = []

    def set_episode_number(self, episode_number: int) -> None:
        self._episode_number = episode_number

    def set_cross_episode_memory(self, messages: list[dict[str, Any]]) -> None:
        self._cross_episode_memory = copy.deepcopy(messages)

    def export_episode_memory(self) -> list[dict[str, Any]]:
        messages = self.filter_messages(self.get_messages())
        converted_messages, _ = to_openai_messages(messages)
        restored_messages = restore_assistant_messages(
            cast(list[dict[str, Any]], converted_messages),
            self._assistant_messages,
        )
        return episode_messages_for_memory(restored_messages)

    def get_token_usage_records(self) -> list[dict[str, int]]:
        return copy.deepcopy(self._token_usage_records)

    def respond(self, ending_index: int | None = None) -> None:
        """Read the current context, call OpenRouter, and add the model response."""
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        messages = self.filter_messages(messages=messages)
        if messages[-1].sender == RoleType.SYSTEM:
            return

        available_tools = self.get_available_tools()
        openrouter_tools = convert_to_openai_tools(available_tools)
        converted_messages, _ = to_openai_messages(messages)
        current_episode_messages = restore_assistant_messages(
            cast(list[dict[str, Any]], converted_messages),
            self._assistant_messages,
        )
        openrouter_messages = compose_with_cross_episode_memory(
            current_episode_messages,
            self._cross_episode_memory,
            episode_number=self._episode_number,
        )

        response = request_openrouter_chat_completion(
            model_name=self.model_name,
            messages=openrouter_messages,
            tools=openrouter_tools,
        )
        self._token_usage_records.append(normalize_token_usage(response.get("usage")))
        response_message = cast(dict[str, Any], response["choices"][0]["message"])

        current_context = get_current_context()
        response_messages = response_message_to_tool_sandbox_messages(
            response_message,
            available_tool_names=set(available_tools),
            agent_to_execution_facing_tool_name=(
                current_context.get_agent_to_execution_facing_tool_name()
            ),
        )
        self.add_messages(response_messages)

        # Keep the original message intact. In particular, OpenRouter requires the
        # ordered reasoning_details blocks to be returned without modification.
        self._assistant_messages.append(copy.deepcopy(response_message))
