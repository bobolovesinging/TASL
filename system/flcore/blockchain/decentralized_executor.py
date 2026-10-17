"""
去中心化智能合约执行器
每个节点（客户端/服务器）都可以通过该执行器独立执行智能合约聚合
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Any

import torch

from .smart_contract import SmartContract, ModelUpdateSubmission
from .p2p_network import P2PNetwork


class DecentralizedContractExecutor:
    """封装节点本地的智能合约执行逻辑"""

    def __init__(
        self,
        node_id: int,
        smart_contract: SmartContract,
        p2p_network: Optional[P2PNetwork] = None,
        min_required: int = 1,
        total_nodes: Optional[int] = None,
    ):
        self.node_id = node_id
        self.smart_contract = smart_contract
        self.p2p_network = p2p_network
        self.min_required = max(1, min_required)
        self.total_nodes = total_nodes
        self._ingested_keys = set()  # (client_id, round_number)

    @staticmethod
    def _tensorize(param):
        if isinstance(param, torch.Tensor):
            return param.clone()
        return torch.tensor(param, dtype=torch.float32)

    @staticmethod
    def _serialize_model(model) -> Dict[str, Any]:
        state = {}
        for name, param in model.named_parameters():
            state[name] = param.detach().cpu().numpy().tolist()
        return state

    def _payload_to_submission(self, payload: Dict[str, Any]) -> Optional[ModelUpdateSubmission]:
        client_id = payload.get("client_id")
        round_number = payload.get("round")
        sample_count = int(payload.get("sample_count", 0))
        model_state = payload.get("model_state", {})
        if client_id is None or round_number is None or not model_state:
            return None

        tensor_state = {}
        for name, value in model_state.items():
            tensor_state[name] = self._tensorize(value)

        submission = ModelUpdateSubmission(
            client_id=client_id,
            round_number=round_number,
            model_params=tensor_state,
            sample_count=sample_count,
            timestamp=payload.get("timestamp", time.time()),
        )
        return submission

    def _ingest_payload(self, payload: Dict[str, Any]):
        submission = self._payload_to_submission(payload)
        if submission is None:
            return
        key = (submission.client_id, submission.round_number)
        if key in self._ingested_keys:
            return
        if self.smart_contract.submit_update(submission):
            self._ingested_keys.add(key)

    def submit_local_update(self, round_number: int, model, sample_count: int):
        """本地节点训练完成后调用，提交并广播模型更新"""
        if self.smart_contract is None or model is None:
            return

        payload = {
            "client_id": self.node_id,
            "round": round_number,
            "sample_count": sample_count,
            "model_state": self._serialize_model(model),
            "timestamp": time.time(),
        }

        # 自己先提交
        self._ingest_payload(payload)

        # 广播给网络中的其他节点
        if self.p2p_network is not None:
            self.p2p_network.broadcast(
                sender_id=self.node_id,
                payload=payload,
                message_type="model_update",
                round_number=round_number,
                exclude_self=True,
            )

    def _collect_network_updates(self, round_number: int):
        if self.p2p_network is None:
            return
        messages = self.p2p_network.receive(self.node_id)
        for message in messages:
            if message.type != "model_update":
                continue
            if message.round != round_number:
                # 仅缓存当前轮次的数据
                continue
            self._ingest_payload(message.payload)

    def run_round(self, round_number: int):
        """执行当前轮次的聚合"""
        self._collect_network_updates(round_number)
        aggregated = self.smart_contract.aggregate(round_number)
        if aggregated is not None:
            # 清理状态，准备下一轮
            self.smart_contract.clear_round_updates(round_number)
            self._ingested_keys = {k for k in self._ingested_keys if k[1] != round_number}
        return aggregated

    def received_update_count(self, round_number: int) -> int:
        return len([key for key in self._ingested_keys if key[1] == round_number])

