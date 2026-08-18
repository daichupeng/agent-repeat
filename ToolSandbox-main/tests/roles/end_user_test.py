# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Tests for the deterministic end user."""

from tool_sandbox.common.execution_context import (
    DatabaseNamespace,
    ExecutionContext,
    RoleType,
    new_context,
)
from tool_sandbox.roles.end_user import EndUser
from tool_sandbox.roles.execution_environment import ExecutionEnvironment


def test_end_user_ends_conversation_without_input_or_model_call() -> None:
    context = ExecutionContext(tool_allow_list=["end_conversation"])
    with new_context(context):
        execution_environment = ExecutionEnvironment()
        context.add_to_database(
            DatabaseNamespace.SANDBOX,
            rows=[
                {
                    "sender": RoleType.SYSTEM,
                    "recipient": RoleType.EXECUTION_ENVIRONMENT,
                    "content": (
                        "from tool_sandbox.tools.user_tools import end_conversation"
                    ),
                }
            ],
        )
        execution_environment.respond()
        context.add_to_database(
            DatabaseNamespace.SANDBOX,
            rows=[
                {
                    "sender": RoleType.AGENT,
                    "recipient": RoleType.USER,
                    "content": "The task is complete.",
                }
            ],
        )

        EndUser().respond()
        end_message = EndUser.get_messages()[-1]
        assert end_message.sender == RoleType.USER
        assert end_message.recipient == RoleType.EXECUTION_ENVIRONMENT
        assert end_message.content == "print(repr(end_conversation()))"

        execution_environment.respond()
        sandbox = context.get_database(DatabaseNamespace.SANDBOX)
        assert not sandbox["conversation_active"][-1]
