# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""Deterministic user role that ends every episode after the agent responds."""

from typing import Optional

from tool_sandbox.common.execution_context import RoleType
from tool_sandbox.common.message_conversion import Message
from tool_sandbox.roles.base_role import BaseRole


class EndUser(BaseRole):
    """Respond to every agent request by invoking ``end_conversation``."""

    role_type: RoleType = RoleType.USER
    model_name = "end"

    def respond(self, ending_index: Optional[int] = None) -> None:
        """End the current conversation without prompting or model inference."""
        messages = self.get_messages(ending_index=ending_index)
        self.messages_validation(messages=messages)
        if "end_conversation" not in self.get_available_tools():
            raise RuntimeError("The deterministic end user requires end_conversation.")
        self.add_messages(
            [
                Message(
                    sender=self.role_type,
                    recipient=RoleType.EXECUTION_ENVIRONMENT,
                    content="print(repr(end_conversation()))",
                )
            ]
        )
