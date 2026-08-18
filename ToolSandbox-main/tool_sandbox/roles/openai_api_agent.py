# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Agent role for any model that conforms to OpenAI tool use API"""

import copy
from typing import Any, Iterable, List, Optional, Union, cast

from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from requests.exceptions import HTTPError
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
from tool_sandbox.common.utils import all_logging_disabled
from tool_sandbox.roles.base_role import BaseRole
from tool_sandbox.roles.conversation_state import (
    compose_with_cross_episode_memory,
    episode_messages_for_memory,
    normalize_token_usage,
)


class OpenAIAPIAgent(BaseRole):
    """Agent role for any model that conforms to OpenAI tool use API"""

    role_type: RoleType = RoleType.AGENT
    model_name: str
    supports_full_memory = True

    def __init__(self) -> None:
        # We set the `base_url` explicitly here to avoid picking up the
        # `OPENAI_BASE_URL` environment variable that may be set for serving models as
        # OpenAI API compatible servers.
        self.openai_client: OpenAI = OpenAI(base_url="https://api.openai.com/v1")
        self._cross_episode_memory: list[dict[str, Any]] = []
        self._episode_number: int | None = None
        self._token_usage_records: list[dict[str, int]] = []

    def set_episode_number(self, episode_number: int) -> None:
        self._episode_number = episode_number

    def set_cross_episode_memory(self, messages: list[dict[str, Any]]) -> None:
        self._cross_episode_memory = copy.deepcopy(messages)

    def export_episode_memory(self) -> list[dict[str, Any]]:
        messages = self.filter_messages(self.get_messages())
        openai_messages, _ = to_openai_messages(messages)
        return episode_messages_for_memory(cast(list[dict[str, Any]], openai_messages))

    def get_token_usage_records(self) -> list[dict[str, int]]:
        return copy.deepcopy(self._token_usage_records)

    def respond(self, ending_index: Optional[int] = None) -> None:
        """Reads a List of messages and attempt to respond with a Message

        Specifically, interprets system, user, execution environment messages and sends out NL response to user, or
        code snippet to execution environment.

        Message comes from current context, the last k messages should be directed to this role type
        Response are written to current context as well. n new messages, addressed to appropriate recipient
        k != n when dealing with parallel function call and responses. Parallel function call are expanded into
        individual messages, parallel function call responses are combined as 1 OpenAI API request

        Args:
            ending_index:   Optional index. Will respond to message located at ending_index instead of most recent one
                            if provided. Utility for processing system message, which could contain multiple entries
                            before each was responded to

        Raises:
            KeyError:   When the last message is not directed to this role
        """
        messages: List[Message] = self.get_messages(ending_index=ending_index)
        response_messages: List[Message] = []
        self.messages_validation(messages=messages)
        # Keeps only relevant messages
        messages = self.filter_messages(messages=messages)
        # Does not respond to System
        if messages[-1].sender == RoleType.SYSTEM:
            return
        # Get OpenAI tools if most recent message is from user
        available_tools = self.get_available_tools()
        available_tool_names = set(available_tools.keys())
        # We need a cast here since `convert_to_openai_tool` returns a plain dict, but
        # `ChatCompletionToolParam` is a `TypedDict`.
        openai_tools = cast(
            Union[Iterable[ChatCompletionToolParam], NotGiven],
            convert_to_openai_tools(available_tools)
            if messages[-1].sender == RoleType.USER
            or messages[-1].sender == RoleType.EXECUTION_ENVIRONMENT
            else NOT_GIVEN,
        )
        # Convert to OpenAI messages.
        current_context = get_current_context()
        current_episode_messages, _ = to_openai_messages(messages)
        openai_messages = compose_with_cross_episode_memory(
            cast(list[dict[str, Any]], current_episode_messages),
            self._cross_episode_memory,
            episode_number=self._episode_number,
        )
        # Call model
        response = self.model_inference(
            openai_messages=openai_messages, openai_tools=openai_tools
        )
        self._token_usage_records.append(normalize_token_usage(response.usage))
        # Parse response
        openai_response_message = response.choices[0].message
        # Message contains no tool call, aka addressed to user
        if openai_response_message.tool_calls is None:
            assert openai_response_message.content is not None
            response_messages = [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.USER,
                    content=openai_response_message.content,
                )
            ]
        else:
            assert openai_tools is not NOT_GIVEN
            for tool_call in openai_response_message.tool_calls:
                # The response contains the agent facing tool name so we need to get
                # the execution facing tool name when creating the Python code.
                execution_facing_tool_name = (
                    current_context.get_execution_facing_tool_name(
                        tool_call.function.name
                    )
                )
                response_messages.append(
                    Message(
                        sender=self.role_type,
                        recipient=RoleType.EXECUTION_ENVIRONMENT,
                        content=openai_tool_call_to_python_code(
                            tool_call,
                            available_tool_names,
                            execution_facing_tool_name=execution_facing_tool_name,
                        ),
                        openai_tool_call_id=tool_call.id,
                        openai_function_name=tool_call.function.name,
                    )
                )
        self.add_messages(response_messages)

    @retry(
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(HTTPError),
    )
    def model_inference(
        self,
        openai_messages: list[dict[str, Any]],
        openai_tools: Union[Iterable[ChatCompletionToolParam], NotGiven],
    ) -> ChatCompletion:
        """Run OpenAI model inference

        Args:
            openai_messages:    List of OpenAI API format messages
            openai_tools:       List of OpenAI API format tools definition

        Returns:
            OpenAI API chat completion object
        """
        with all_logging_disabled():
            return self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=cast(list[ChatCompletionMessageParam], openai_messages),
                tools=openai_tools,
            )


class GPT_4_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-4-0125-preview"


class GPT_3_5_0125_Agent(OpenAIAPIAgent):
    model_name = "gpt-3.5-turbo-0125"


class GPT_4_o_2024_05_13_Agent(OpenAIAPIAgent):
    model_name = "gpt-4o-2024-05-13"
