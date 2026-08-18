# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Unit tests for the OpenRouter-backed DeepSeek agent."""

import json
from unittest.mock import Mock, patch

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    new_context,
)
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.deepseek_openrouter_agent import (
    DEFAULT_DEEPSEEK_MODEL,
    OPENROUTER_CHAT_COMPLETIONS_URL,
    OPENROUTER_REQUEST_TIMEOUT_SECONDS,
    request_openrouter_chat_completion,
    response_message_to_tool_sandbox_messages,
    restore_assistant_messages,
)
from tool_sandbox.roles.execution_environment import ExecutionEnvironment


@patch("tool_sandbox.roles.deepseek_openrouter_agent.requests.post")
def test_request_openrouter_chat_completion(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "done"}}]
    }
    mock_post.return_value = response
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "test"}}]

    result = request_openrouter_chat_completion(
        messages=messages,
        tools=tools,
        api_key="test-key",
    )

    assert result == response.json.return_value
    response.raise_for_status.assert_called_once_with()
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == OPENROUTER_CHAT_COMPLETIONS_URL
    assert call_kwargs["headers"] == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert call_kwargs["timeout"] == OPENROUTER_REQUEST_TIMEOUT_SECONDS
    assert json.loads(call_kwargs["data"]) == {
        "model": DEFAULT_DEEPSEEK_MODEL,
        "messages": messages,
        "reasoning": {"enabled": True},
        "tools": tools,
    }


def test_restore_assistant_messages_preserves_reasoning_details() -> None:
    reasoning_details = [
        {
            "type": "reasoning.text",
            "text": "encrypted-or-provider-specific-content",
            "id": "reasoning-1",
            "format": "unknown",
        }
    ]
    reconstructed_messages = [
        {"role": "user", "content": "Use a tool"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
    ]
    original_assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": reconstructed_messages[1]["tool_calls"],
        "reasoning_details": reasoning_details,
    }

    restored = restore_assistant_messages(
        reconstructed_messages,
        [original_assistant_message],
    )

    assert restored[1] == original_assistant_message
    assert restored[1]["reasoning_details"] == reasoning_details
    assert reconstructed_messages[1].get("reasoning_details") is None


def test_response_message_converts_tool_call() -> None:
    converted_messages = response_message_to_tool_sandbox_messages(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "chatcmpl-tool-baf359e53cef95ed",
                    "type": "function",
                    "function": {
                        "name": "agent_facing_name",
                        "arguments": '{"on": true}',
                    },
                }
            ],
            "reasoning_details": [{"type": "reasoning.text", "text": "reason"}],
        },
        available_tool_names={"agent_facing_name"},
        agent_to_execution_facing_tool_name={
            "agent_facing_name": "execution_facing_name"
        },
    )

    assert len(converted_messages) == 1
    message = converted_messages[0]
    assert message.openai_tool_call_id == "chatcmpl-tool-baf359e53cef95ed"
    assert message.openai_function_name == "agent_facing_name"
    assert (
        "execution_facing_name(**chatcmpl_tool_baf359e53cef95ed_parameters)"
        in message.content
    )
    compile(message.content, "<tool-call>", "exec")


def test_hyphenated_openrouter_tool_call_executes_messaging_tool() -> None:
    context = ExecutionContext(tool_allow_list=["send_message_with_phone_number"])
    with new_context(context):
        context.add_to_database(
            DatabaseNamespace.CONTACT,
            rows=[
                {
                    "person_id": "self-person",
                    "name": "Test User",
                    "phone_number": "+11234567890",
                    "relationship": "self",
                    "is_self": True,
                }
            ],
        )
        execution_environment = ExecutionEnvironment()
        execution_environment.add_messages(
            [
                Message(
                    sender=RoleType.AGENT,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content=(
                        "from tool_sandbox.tools.messaging import "
                        "send_message_with_phone_number"
                    ),
                    conversation_active=True,
                )
            ]
        )
        execution_environment.respond()

        tool_messages = response_message_to_tool_sandbox_messages(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-baf359e53cef95ed",
                        "type": "function",
                        "function": {
                            "name": "send_message_with_phone_number",
                            "arguments": json.dumps(
                                {
                                    "phone_number": "+12453344098",
                                    "content": "How's the new album coming along",
                                }
                            ),
                        },
                    }
                ],
            },
            available_tool_names={"send_message_with_phone_number"},
            agent_to_execution_facing_tool_name={
                "send_message_with_phone_number": "send_message_with_phone_number"
            },
        )
        execution_environment.add_messages(tool_messages)
        execution_environment.respond()

        sent_messages = context.get_database(DatabaseNamespace.MESSAGING).to_dicts()
        assert len(sent_messages) == 1
        assert sent_messages[0]["recipient_phone_number"] == "+12453344098"
        assert sent_messages[0]["content"] == "How's the new album coming along"
        tool_result = execution_environment.get_messages()[-1]
        assert tool_result.openai_tool_call_id == "chatcmpl-tool-baf359e53cef95ed"
        assert tool_result.tool_call_exception is None
