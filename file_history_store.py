"""基于本地文件的多轮对话历史存储"""

import os
import json
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict


def get_history(session_id):
    """获取多轮对话历史"""
    return FileChatMessageHistory(session_id, storage_path="./history")


class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id, storage_path):
        self.session_id = session_id
        self.storage_path = storage_path
        self.file_path = os.path.join(self.storage_path, self.session_id)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        all_messages = [*self.messages, *messages]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(message) for message in all_messages], f)

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except FileNotFoundError:
            return []

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
