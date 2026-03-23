"""
Memory management for RAG conversations.

Provides conversation history tracking for multi-turn interactions.
"""

import logging
from dataclasses import dataclass
from typing import Dict
from uuid import uuid4

import adalflow as adal

logger = logging.getLogger(__name__)


@dataclass
class UserQuery:
    """Represents a user query in the conversation."""
    query_str: str


@dataclass
class AssistantResponse:
    """Represents an assistant response in the conversation."""
    response_str: str


@dataclass
class DialogTurn:
    """Represents a single turn in the conversation (user query + assistant response)."""
    id: str
    user_query: UserQuery
    assistant_response: AssistantResponse


class CustomConversation:
    """Custom implementation of Conversation to fix the list assignment index out of range error."""

    def __init__(self):
        self.dialog_turns = []

    def append_dialog_turn(self, dialog_turn: DialogTurn):
        """Safely append a dialog turn to the conversation."""
        if not hasattr(self, 'dialog_turns'):
            self.dialog_turns = []
        self.dialog_turns.append(dialog_turn)


class Memory(adal.core.component.DataComponent):
    """Simple conversation management with a list of dialog turns."""

    def __init__(self):
        super().__init__()
        # Use our custom implementation instead of the original Conversation class
        self.current_conversation = CustomConversation()

    def call(self) -> Dict:
        """Return the conversation history as a dictionary."""
        all_dialog_turns = {}
        try:
            # Check if dialog_turns exists and is a list
            if hasattr(self.current_conversation, 'dialog_turns'):
                if self.current_conversation.dialog_turns:
                    logger.info(f"Memory content: {len(self.current_conversation.dialog_turns)} turns")
                    for i, turn in enumerate(self.current_conversation.dialog_turns):
                        if hasattr(turn, 'id') and turn.id is not None:
                            all_dialog_turns[turn.id] = turn
                            logger.info(f"Added turn {i+1} with ID {turn.id} to memory")
                        else:
                            logger.warning(f"Skipping invalid turn object in memory: {turn}")
                else:
                    logger.info("Dialog turns list exists but is empty")
            else:
                logger.info("No dialog_turns attribute in current_conversation")
                # Try to initialize it
                self.current_conversation.dialog_turns = []
        except Exception as e:
            logger.error(f"Error accessing dialog turns: {str(e)}")
            # Try to recover
            try:
                self.current_conversation = CustomConversation()
                logger.info("Recovered by creating new conversation")
            except Exception as e2:
                logger.error(f"Failed to recover: {str(e2)}")

        logger.info(f"Returning {len(all_dialog_turns)} dialog turns from memory")
        return all_dialog_turns

    def add_dialog_turn(self, user_query: str, assistant_response: str) -> bool:
        """
        Add a dialog turn to the conversation history.

        Args:
            user_query: The user's query
            assistant_response: The assistant's response

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create a new dialog turn using our custom implementation
            dialog_turn = DialogTurn(
                id=str(uuid4()),
                user_query=UserQuery(query_str=user_query),
                assistant_response=AssistantResponse(response_str=assistant_response),
            )

            # Make sure the current_conversation has the append_dialog_turn method
            if not hasattr(self.current_conversation, 'append_dialog_turn'):
                logger.warning("current_conversation does not have append_dialog_turn method, creating new one")
                # Initialize a new conversation if needed
                self.current_conversation = CustomConversation()

            # Ensure dialog_turns exists
            if not hasattr(self.current_conversation, 'dialog_turns'):
                logger.warning("dialog_turns not found, initializing empty list")
                self.current_conversation.dialog_turns = []

            # Safely append the dialog turn
            self.current_conversation.dialog_turns.append(dialog_turn)
            logger.info(f"Successfully added dialog turn, now have {len(self.current_conversation.dialog_turns)} turns")
            return True

        except Exception as e:
            logger.error(f"Error adding dialog turn: {str(e)}")
            # Try to recover by creating a new conversation
            try:
                self.current_conversation = CustomConversation()
                dialog_turn = DialogTurn(
                    id=str(uuid4()),
                    user_query=UserQuery(query_str=user_query),
                    assistant_response=AssistantResponse(response_str=assistant_response),
                )
                self.current_conversation.dialog_turns.append(dialog_turn)
                logger.info("Recovered from error by creating new conversation")
                return True
            except Exception as e2:
                logger.error(f"Failed to recover from error: {str(e2)}")
                return False
