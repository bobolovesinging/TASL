"""
智能合约模块
用于去中心化的联邦学习模型聚合
"""
import torch
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import copy


@dataclass
class ModelUpdateSubmission:
    """模型更新提交"""
    client_id: int
    round_number: int
    model_params: Dict[str, torch.Tensor]
    gradient_params: Optional[Dict[str, torch.Tensor]] = None  # 梯度参数
    sample_count: int = 0  # 客户端样本数量（用于加权聚合）
    train_loss: float = 0.0  # 训练损失
    timestamp: float = 0.0
    signature: Optional[str] = None  # 可选的数字签名


class SmartContract:
    """
    智能合约：定义联邦学习的聚合规则和执行逻辑
    替代集中式服务器的聚合功能
    """
    
    def __init__(self, 
                 aggregation_method: str = "fedavg",
                 min_clients: int = 1,
                 max_clients: Optional[int] = None,
                 validator_func: Optional[Callable] = None,
                 keep_recent_rounds: int = 2):
        """
        初始化智能合约
        :param aggregation_method: 聚合方法 ("fedavg", "weighted_avg", "median", "krum")
        :param min_clients: 最小参与客户端数量
        :param max_clients: 最大参与客户端数量（None表示不限制）
        :param validator_func: 可选的验证函数，用于验证模型更新的有效性
        :param keep_recent_rounds: 内存中保留的最近轮次数（默认2轮，不考虑溯源性时只保留最近数据）
        """
        self.aggregation_method = aggregation_method
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.validator_func = validator_func
        self.keep_recent_rounds = keep_recent_rounds  # 内存中保留的最近轮次数
        self.update_history: List[ModelUpdateSubmission] = []
        self.aggregation_history: List[Dict[str, Any]] = []
    
    def submit_update(self, submission: ModelUpdateSubmission) -> bool:
        """
        提交模型更新到智能合约
        :param submission: 模型更新提交
        :return: 是否成功提交
        """
        # 验证提交
        if not self._validate_submission(submission):
            return False
        
        # 添加到历史记录
        self.update_history.append(submission)
        return True
    
    def _validate_submission(self, submission: ModelUpdateSubmission) -> bool:
        """验证提交的有效性"""
        # 检查是否超过最大客户端数
        if self.max_clients and len(self.update_history) >= self.max_clients:
            return False
        
        # 检查是否重复提交（同一客户端同一轮次）
        for existing in self.update_history:
            if (existing.client_id == submission.client_id and 
                existing.round_number == submission.round_number):
                return False
        
        # 使用自定义验证函数（如果提供）
        if self.validator_func:
            return self.validator_func(submission)
        
        return True
    
    def aggregate(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        执行聚合（智能合约的核心功能）
        :param round_number: 轮次编号
        :return: 聚合后的模型参数，如果参与客户端不足则返回None
        """
        # 筛选当前轮次的更新
        round_updates = [
            u for u in self.update_history 
            if u.round_number == round_number
        ]
        
        if len(round_updates) < self.min_clients:
            print(f"[SmartContract] 参与客户端数量不足: {len(round_updates)} < {self.min_clients}")
            return None
        
        print(f"[SmartContract] 开始聚合 {len(round_updates)} 个客户端的更新 (方法: {self.aggregation_method})")
        
        # 根据聚合方法执行聚合
        if self.aggregation_method == "fedavg":
            aggregated = self._fedavg_aggregate(round_updates)
        elif self.aggregation_method == "weighted_avg":
            aggregated = self._weighted_avg_aggregate(round_updates)
        elif self.aggregation_method == "median":
            aggregated = self._median_aggregate(round_updates)
        elif self.aggregation_method == "krum":
            aggregated = self._krum_aggregate(round_updates)
        else:
            raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")
        
        # 记录聚合历史
        self.aggregation_history.append({
            'round_number': round_number,
            'client_count': len(round_updates),
            'method': self.aggregation_method,
            'timestamp': round_updates[0].timestamp if round_updates else 0
        })
        
        return aggregated
    
    def _fedavg_aggregate(self, updates: List[ModelUpdateSubmission]) -> Dict[str, torch.Tensor]:
        """FedAvg聚合：简单平均（支持summary模式和全量模式）"""
        aggregated = {}
        num_updates = len(updates)
        
        if not updates:
            return aggregated
        
        # 初始化聚合模型
        first_params = updates[0].model_params
        for key in first_params:
            param_value = first_params[key]
            
            # 检查是否是summary模式（字典格式）
            if isinstance(param_value, dict):
                # Summary模式：聚合统计信息
                aggregated[key] = {
                    "mean": sum(u.model_params[key].get("mean", 0.0) for u in updates) / num_updates,
                    "std": sum(u.model_params[key].get("std", 0.0) for u in updates) / num_updates,
                    "l2": sum(u.model_params[key].get("l2", 0.0) for u in updates) / num_updates
                }
            elif isinstance(param_value, torch.Tensor):
                # 全量模式：聚合Tensor
                aggregated[key] = torch.zeros_like(param_value)
                for update in updates:
                    aggregated[key] += update.model_params[key] / num_updates
            else:
                # 其他类型：直接复制第一个
                aggregated[key] = param_value
        
        return aggregated
    
    def _weighted_avg_aggregate(self, updates: List[ModelUpdateSubmission]) -> Dict[str, torch.Tensor]:
        """加权平均聚合：根据样本数量加权（支持summary模式和全量模式）"""
        aggregated = {}
        total_samples = sum(u.sample_count for u in updates)
        
        if not updates or total_samples == 0:
            return aggregated
        
        # 初始化聚合模型
        first_params = updates[0].model_params
        for key in first_params:
            param_value = first_params[key]
            
            # 检查是否是summary模式（字典格式）
            if isinstance(param_value, dict):
                # Summary模式：加权聚合统计信息
                aggregated[key] = {
                    "mean": sum(u.model_params[key].get("mean", 0.0) * u.sample_count for u in updates) / total_samples,
                    "std": sum(u.model_params[key].get("std", 0.0) * u.sample_count for u in updates) / total_samples,
                    "l2": sum(u.model_params[key].get("l2", 0.0) * u.sample_count for u in updates) / total_samples
                }
            elif isinstance(param_value, torch.Tensor):
                # 全量模式：加权聚合Tensor
                aggregated[key] = torch.zeros_like(param_value)
                for update in updates:
                    weight = update.sample_count / total_samples
                    aggregated[key] += update.model_params[key] * weight
            else:
                # 其他类型：直接复制第一个
                aggregated[key] = param_value
        
        return aggregated
    
    def _median_aggregate(self, updates: List[ModelUpdateSubmission]) -> Dict[str, torch.Tensor]:
        """中位数聚合：对每个参数取中位数（鲁棒聚合）"""
        aggregated = {}
        first_params = updates[0].model_params
        
        for key in first_params:
            # 收集所有客户端该参数的值
            param_stack = torch.stack([u.model_params[key] for u in updates], dim=0)
            # 计算中位数
            aggregated[key] = torch.median(param_stack, dim=0)[0]
        
        return aggregated
    
    def _krum_aggregate(self, updates: List[ModelUpdateSubmission], f: int = 1) -> Dict[str, torch.Tensor]:
        """
        Krum聚合：选择最接近其他更新的模型（防御投毒攻击）
        :param f: 假设的恶意客户端数量
        """
        if len(updates) <= 2 * f + 2:
            # 客户端数量不足，回退到FedAvg
            return self._fedavg_aggregate(updates)
        
        # 计算每个更新到其他更新的距离
        distances = []
        for i, update_i in enumerate(updates):
            dists = []
            for j, update_j in enumerate(updates):
                if i != j:
                    # 计算L2距离
                    dist = 0.0
                    for key in update_i.model_params:
                        diff = update_i.model_params[key] - update_j.model_params[key]
                        dist += torch.sum(diff ** 2).item()
                    dists.append(np.sqrt(dist))
            # 选择最近的 n-f-2 个距离
            dists.sort()
            score = sum(dists[:len(updates) - f - 2])
            distances.append((score, i))
        
        # 选择得分最低的更新（最接近其他更新）
        distances.sort()
        best_idx = distances[0][1]
        
        return copy.deepcopy(updates[best_idx].model_params)
    
    def clear_round_updates(self, round_number: int):
        """
        清除指定轮次的更新记录（聚合完成后调用）
        同时清理旧轮次的数据以释放内存
        """
        # 清除指定轮次的更新
        updates_to_remove = []
        for u in self.update_history:
            if u.round_number == round_number:
                # 释放tensor内存
                if hasattr(u, 'model_params') and u.model_params:
                    for key, value in u.model_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.model_params.clear()
                if hasattr(u, 'gradient_params') and u.gradient_params:
                    for key, value in u.gradient_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.gradient_params.clear()
                updates_to_remove.append(u)
        
        # 从历史记录中移除
        self.update_history = [
            u for u in self.update_history 
            if u.round_number != round_number
        ]
        
        # 清理旧轮次的数据（只保留最近keep_recent_rounds轮）
        if self.update_history:
            max_round = max(u.round_number for u in self.update_history)
            min_keep_round = max(0, max_round - self.keep_recent_rounds + 1)
            
            old_updates = []
            for u in self.update_history:
                if u.round_number < min_keep_round:
                    old_updates.append(u)
            
            # 释放旧更新的内存
            for u in old_updates:
                if hasattr(u, 'model_params') and u.model_params:
                    for key, value in u.model_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.model_params.clear()
                if hasattr(u, 'gradient_params') and u.gradient_params:
                    for key, value in u.gradient_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.gradient_params.clear()
            
            # 从历史记录中移除旧轮次
            self.update_history = [
                u for u in self.update_history 
                if u.round_number >= min_keep_round
            ]
            
            if old_updates:
                print(f"[SmartContract] 已清理 {len(old_updates)} 个旧轮次的更新记录，保留最近 {self.keep_recent_rounds} 轮")
        
        # 清理CUDA缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return {
            'total_updates': len(self.update_history),
            'total_aggregations': len(self.aggregation_history),
            'method': self.aggregation_method,
            'recent_aggregations': self.aggregation_history[-10:] if self.aggregation_history else []
        }


class SmartContractNetwork:
    """
    智能合约网络：管理多个智能合约实例（支持不同聚合策略）
    """
    
    def __init__(self):
        self.contracts: Dict[str, SmartContract] = {}
        self.default_contract: Optional[SmartContract] = None
    
    def deploy_contract(self, 
                       contract_id: str,
                       aggregation_method: str = "fedavg",
                       min_clients: int = 1,
                       max_clients: Optional[int] = None,
                       set_as_default: bool = False) -> SmartContract:
        """
        部署智能合约
        :param contract_id: 合约ID
        :param aggregation_method: 聚合方法
        :param min_clients: 最小客户端数
        :param max_clients: 最大客户端数
        :param set_as_default: 是否设为默认合约
        :return: 创建的智能合约实例
        """
        contract = SmartContract(
            aggregation_method=aggregation_method,
            min_clients=min_clients,
            max_clients=max_clients
        )
        self.contracts[contract_id] = contract
        
        if set_as_default or self.default_contract is None:
            self.default_contract = contract
        
        print(f"[SmartContractNetwork] 已部署合约: {contract_id} (方法: {aggregation_method})")
        return contract
    
    def get_contract(self, contract_id: Optional[str] = None) -> Optional[SmartContract]:
        """获取智能合约实例"""
        if contract_id:
            return self.contracts.get(contract_id)
        return self.default_contract
    
    def list_contracts(self) -> List[str]:
        """列出所有合约ID"""
        return list(self.contracts.keys())


class NPCDeputySmartContract(SmartContract):
    """
    NPC Deputy智能合约：只保存M份梯度，由deputy管理
    支持基于余弦相似度和权益的聚合
    只保留最近5轮数据在内存中，旧数据保存到本地文件
    """
    
    def __init__(self,
                 aggregation_method: str = "fedavg",
                 min_clients: int = 1,
                 max_clients: Optional[int] = None,
                 max_gradients: int = 5,  # 最多保存M份梯度
                 validator_func: Optional[Callable] = None,
                 keep_recent_rounds: int = 2,  # 内存中保留的最近轮次数（默认2轮，与区块链保持一致）
                 storage_dir: str = "blockchain_gradient_storage"):  # 本地存储目录
        """
        初始化NPC Deputy智能合约
        :param max_gradients: 最多保存的梯度数量（M）
        :param keep_recent_rounds: 内存中保留的最近轮次数（默认5轮）
        :param storage_dir: 本地存储目录（用于保存旧轮次数据）
        """
        super().__init__(aggregation_method, min_clients, max_clients, validator_func)
        self.max_gradients = max_gradients
        self.gradient_storage: Dict[int, List[ModelUpdateSubmission]] = {}  # {round: [submissions]}
        self.keep_recent_rounds = keep_recent_rounds  # 内存中保留的最近轮次数
        self.storage_dir = storage_dir  # 本地存储目录
        
        # 创建存储目录
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def submit_gradient(self, submission: ModelUpdateSubmission, 
                       deputy_ids: List[int], round_number: int) -> bool:
        """
        提交梯度到区块链账单（只保存M份，由deputy管理）
        :param submission: 模型更新提交（可以是任何客户端的梯度，由deputy筛选后提交）
        :param deputy_ids: 当前deputy节点ID列表（用于验证，但不再限制只有deputy的梯度）
        :param round_number: 轮次
        :return: 是否成功提交
        """
        # 注意：现在允许提交任何客户端的梯度，deputy负责筛选
        # deputy_ids参数保留用于验证deputy身份，但不限制梯度来源
        
        # 验证提交
        if not self._validate_submission(submission):
            return False
        
        # 初始化该轮次的存储
        if round_number not in self.gradient_storage:
            self.gradient_storage[round_number] = []
        
        # 注意：现在所有梯度都保存到区块链（保证数据不变）
        # max_gradients参数保留用于其他用途，但不限制存储数量
        # 所有梯度都上链，保证数据不可篡改
        
        # 添加到梯度存储
        self.gradient_storage[round_number].append(submission)
        
        # 同时添加到历史记录（用于聚合）
        self.update_history.append(submission)
        
        # 管理存储：确保内存中只保留最近keep_recent_rounds轮
        self._manage_storage(round_number)
        
        return True
    
    def _save_round_to_disk(self, round_number: int, gradients: List[ModelUpdateSubmission]):
        """将指定轮次的梯度数据保存到磁盘"""
        if not gradients:
            return
        
        # 将梯度数据转换为可序列化的格式
        serialized_data = []
        for grad in gradients:
            grad_dict = {
                'client_id': grad.client_id,
                'round_number': grad.round_number,
                'sample_count': grad.sample_count,
                'train_loss': grad.train_loss,
                'timestamp': grad.timestamp,
                'signature': grad.signature,
                # 将tensor转换为numpy数组以便序列化
                'model_params': {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                                for k, v in grad.model_params.items()},
                'gradient_params': {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v 
                                   for k, v in (grad.gradient_params or {}).items()}
            }
            serialized_data.append(grad_dict)
        
        # 保存到文件
        file_path = os.path.join(self.storage_dir, f"gradients_round_{round_number}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)
        
        print(f"[NPCDeputySmartContract] 已将轮次 {round_number} 的 {len(gradients)} 个梯度保存到磁盘: {file_path}")
    
    def _load_round_from_disk(self, round_number: int) -> List[ModelUpdateSubmission]:
        """从磁盘加载指定轮次的梯度数据"""
        file_path = os.path.join(self.storage_dir, f"gradients_round_{round_number}.json")
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                serialized_data = json.load(f)
            
            gradients = []
            for grad_dict in serialized_data:
                # 将numpy数组转换回tensor
                model_params = {k: torch.tensor(v) if isinstance(v, list) else v 
                               for k, v in grad_dict['model_params'].items()}
                gradient_params = None
                if grad_dict.get('gradient_params'):
                    gradient_params = {k: torch.tensor(v) if isinstance(v, list) else v 
                                     for k, v in grad_dict['gradient_params'].items()}
                
                grad = ModelUpdateSubmission(
                    client_id=grad_dict['client_id'],
                    round_number=grad_dict['round_number'],
                    model_params=model_params,
                    gradient_params=gradient_params,
                    sample_count=grad_dict['sample_count'],
                    train_loss=grad_dict['train_loss'],
                    timestamp=grad_dict['timestamp'],
                    signature=grad_dict.get('signature')
                )
                gradients.append(grad)
            
            print(f"[NPCDeputySmartContract] 从磁盘加载轮次 {round_number} 的 {len(gradients)} 个梯度")
            return gradients
        except Exception as e:
            print(f"[NPCDeputySmartContract] 加载轮次 {round_number} 失败: {e}")
            return []
    
    def _manage_storage(self, current_round: int):
        """
        管理存储：将超过keep_recent_rounds的旧轮次数据从内存中删除
        同时清理update_history中的旧数据（不考虑溯源性，只保留最近数据）
        """
        if not self.gradient_storage:
            return
        
        # 获取所有轮次
        all_rounds = sorted(self.gradient_storage.keys())
        
        # 计算需要保留在内存中的轮次范围
        min_keep_round = max(0, current_round - self.keep_recent_rounds + 1)
        
        # 将旧轮次的数据从内存中删除（不保存到磁盘，因为区块链已经保存了）
        rounds_to_remove = [r for r in all_rounds if r < min_keep_round]
        
        for round_num in rounds_to_remove:
            gradients = self.gradient_storage[round_num]
            if gradients:
                # 释放梯度数据中的tensor内存
                for grad in gradients:
                    if grad.gradient_params:
                        for key, value in grad.gradient_params.items():
                            if isinstance(value, torch.Tensor):
                                del value
                        grad.gradient_params.clear()
                    if grad.model_params:
                        for key, value in grad.model_params.items():
                            if isinstance(value, torch.Tensor):
                                del value
                        grad.model_params.clear()
                gradients.clear()
                del gradients
            del self.gradient_storage[round_num]
        
        # 同时清理update_history中的旧轮次数据（重要：防止内存泄漏）
        if self.update_history:
            old_updates = []
            for u in self.update_history:
                if u.round_number < min_keep_round:
                    old_updates.append(u)
            
            # 释放旧更新的内存
            for u in old_updates:
                if hasattr(u, 'model_params') and u.model_params:
                    for key, value in u.model_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.model_params.clear()
                if hasattr(u, 'gradient_params') and u.gradient_params:
                    for key, value in u.gradient_params.items():
                        if isinstance(value, torch.Tensor):
                            del value
                    u.gradient_params.clear()
            
            # 从历史记录中移除旧轮次
            self.update_history = [
                u for u in self.update_history 
                if u.round_number >= min_keep_round
            ]
            
            if old_updates:
                print(f"[NPCDeputySmartContract] 已清理 {len(old_updates)} 个旧轮次的update_history记录")
        
        if rounds_to_remove:
            print(f"[NPCDeputySmartContract] 存储管理完成: 已删除 {len(rounds_to_remove)} 个旧轮次的梯度数据，"
                  f"内存中保留轮次 {min_keep_round} 到 {current_round}")
        
        # 清理CUDA缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    def get_stored_gradients(self, round_number: int) -> List[ModelUpdateSubmission]:
        """
        获取某一轮次存储的梯度
        如果不在内存中，尝试从磁盘加载
        """
        # 首先检查内存中是否有
        if round_number in self.gradient_storage:
            return self.gradient_storage[round_number]
        
        # 尝试从磁盘加载
        disk_gradients = self._load_round_from_disk(round_number)
        if disk_gradients:
            # 如果从磁盘加载成功，可以临时放回内存（可选）
            # 这里直接返回，不加载到内存，避免占用过多内存
            return disk_gradients
        
        return []
    
    def clear_round_updates(self, round_number: int):
        """
        清除指定轮次的更新记录（聚合完成后调用）
        如果该轮次不在最近keep_recent_rounds轮中，会保存到磁盘
        """
        super().clear_round_updates(round_number)
        
        # 如果该轮次在内存中，立即删除（不保存到磁盘，因为区块链已经保存了）
        if round_number in self.gradient_storage:
            # 释放梯度数据中的tensor内存
            gradients = self.gradient_storage[round_number]
            for grad in gradients:
                if grad.gradient_params:
                    grad.gradient_params.clear()
                if grad.model_params:
                    grad.model_params.clear()
            gradients.clear()
            del gradients
            del self.gradient_storage[round_number]
        
        # 管理存储：确保内存中只保留最近keep_recent_rounds轮
        # 使用当前最大轮次来管理存储
        if self.gradient_storage:
            max_round = max(self.gradient_storage.keys())
            self._manage_storage(max_round)
        
        # 清理CUDA缓存
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

