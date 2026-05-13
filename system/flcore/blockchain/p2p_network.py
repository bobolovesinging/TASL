"""
P2P通信层：提供节点之间的消息广播与接收能力
"""
from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Dict, List, Optional, Any


class P2PMessage:
    """统一的P2P消息结构"""

    def __init__(
        self,
        sender_id: int,
        message_type: str,
        round_number: int,
        payload: Dict[str, Any],
        target_id: Optional[int] = None,
    ):
        self.sender_id = sender_id
        self.target_id = target_id
        self.type = message_type
        self.round = round_number
        self.payload = payload
        self.timestamp = time.time()


class P2PNetwork:
    """基于内存队列的简易P2P网络（用于本地模拟）"""

    def __init__(self):
        self._channels: Dict[int, Queue] = {}
        self._lock = threading.Lock()

    def register_node(self, node_id: int):
        with self._lock:
            if node_id not in self._channels:
                self._channels[node_id] = Queue()

    def unregister_node(self, node_id: int):
        with self._lock:
            if node_id in self._channels:
                self._channels.pop(node_id)

    def list_nodes(self) -> List[int]:
        with self._lock:
            return list(self._channels.keys())

    def send(
        self,
        sender_id: int,
        target_id: int,
        payload: Dict[str, Any],
        message_type: str = "model_update",
        round_number: int = 0,
    ):
        if target_id == sender_id:
            return
        with self._lock:
            if target_id not in self._channels:
                return
            message = P2PMessage(
                sender_id=sender_id,
                target_id=target_id,
                message_type=message_type,
                round_number=round_number,
                payload=payload,
            )
            self._channels[target_id].put(message)

    def broadcast(
        self,
        sender_id: int,
        payload: Dict[str, Any],
        message_type: str = "model_update",
        round_number: int = 0,
        exclude_self: bool = False,
    ):
        targets = []
        with self._lock:
            for node_id in self._channels.keys():
                if exclude_self and node_id == sender_id:
                    continue
                targets.append(node_id)
        for target_id in targets:
            self.send(
                sender_id=sender_id,
                target_id=target_id,
                payload=payload,
                message_type=message_type,
                round_number=round_number,
            )

    def receive(
        self,
        node_id: int,
        max_messages: Optional[int] = None,
    ) -> List[P2PMessage]:
        messages: List[P2PMessage] = []
        queue_ref = self._channels.get(node_id)
        if queue_ref is None:
            return messages
        while not queue_ref.empty():
            if max_messages is not None and len(messages) >= max_messages:
                break
            messages.append(queue_ref.get())
        return messages

    def clear(self, node_id: int):
        queue_ref = self._channels.get(node_id)
        if queue_ref is None:
            return
        while not queue_ref.empty():
            queue_ref.get()


class GlobalP2PNetwork:
    """P2P网络的单例入口，方便在不同模块之间共享"""

    _instance: Optional[P2PNetwork] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> P2PNetwork:
        with cls._lock:
            if cls._instance is None:
                cls._instance = P2PNetwork()
            return cls._instance

