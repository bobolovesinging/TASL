"""
Deputy验证模块
实现deputy节点的验证职责（验证梯度格式、签名、时间戳等）
"""
import time
from typing import Dict, Any, Optional, Tuple, List
from flcore.blockchain.smart_contract import ModelUpdateSubmission


class DeputyValidator:
    """Deputy验证器：验证梯度提交的有效性"""
    
    def __init__(self, max_age: float = 3600.0, min_sample_count: int = 1):
        """
        初始化验证器
        :param max_age: 梯度提交的最大年龄（秒），超过此时间的提交视为无效
        :param min_sample_count: 最小样本数量
        """
        self.max_age = max_age
        self.min_sample_count = min_sample_count
    
    def validate_gradient_submission(self, submission: ModelUpdateSubmission, 
                                   round_number: int) -> Tuple[bool, str]:
        """
        验证梯度提交的有效性
        :param submission: 梯度提交
        :param round_number: 当前轮次
        :return: (是否有效, 错误信息)
        """
        # 1. 验证轮次
        if submission.round_number != round_number:
            return False, f"轮次不匹配: 提交的轮次={submission.round_number}, 当前轮次={round_number}"
        
        # 2. 验证时间戳
        current_time = time.time()
        age = current_time - submission.timestamp
        if age < 0:
            return False, f"时间戳异常: 提交时间在未来"
        if age > self.max_age:
            return False, f"梯度提交过期: 年龄={age:.2f}秒, 最大允许={self.max_age}秒"
        
        # 3. 验证模型参数
        if not submission.model_params:
            return False, "模型参数为空"
        
        # 检查模型参数格式
        for name, param in submission.model_params.items():
            if param is None:
                return False, f"模型参数 {name} 为None"
            # 如果param是tensor，检查是否有NaN或Inf
            try:
                import torch
                if isinstance(param, torch.Tensor):
                    if torch.isnan(param).any():
                        return False, f"模型参数 {name} 包含NaN值"
                    if torch.isinf(param).any():
                        return False, f"模型参数 {name} 包含Inf值"
            except:
                pass  # 如果不是tensor，跳过
        
        # 4. 验证梯度参数（如果提供）
        if submission.gradient_params:
            for name, grad in submission.gradient_params.items():
                if grad is None:
                    return False, f"梯度参数 {name} 为None"
                try:
                    import torch
                    if isinstance(grad, torch.Tensor):
                        if torch.isnan(grad).any():
                            return False, f"梯度参数 {name} 包含NaN值"
                        if torch.isinf(grad).any():
                            return False, f"梯度参数 {name} 包含Inf值"
                except:
                    pass
        
        # 5. 验证样本数量
        if submission.sample_count < self.min_sample_count:
            return False, f"样本数量不足: {submission.sample_count} < {self.min_sample_count}"
        
        # 6. 验证训练损失（如果提供）
        if submission.train_loss is not None:
            if submission.train_loss < 0:
                return False, f"训练损失为负: {submission.train_loss}"
            if submission.train_loss > 1000:  # 异常大的损失值
                return False, f"训练损失异常大: {submission.train_loss}"
        
        # 7. 验证签名（如果提供）
        if submission.signature:
            # 这里可以添加数字签名验证逻辑
            # 目前只是检查签名是否存在
            pass
        
        return True, ""
    
    def validate_batch_submissions(self, submissions: List[ModelUpdateSubmission],
                                 round_number: int) -> Tuple[List[ModelUpdateSubmission], List[Tuple[int, str]]]:
        """
        批量验证梯度提交
        :param submissions: 梯度提交列表
        :param round_number: 当前轮次
        :return: (有效提交列表, [(client_id, 错误信息), ...])
        """
        valid_submissions = []
        invalid_info = []
        
        for submission in submissions:
            is_valid, error_msg = self.validate_gradient_submission(submission, round_number)
            if is_valid:
                valid_submissions.append(submission)
            else:
                invalid_info.append((submission.client_id, error_msg))
        
        return valid_submissions, invalid_info
    
    def check_gradient_consistency(self, submission1: ModelUpdateSubmission,
                                  submission2: ModelUpdateSubmission) -> Tuple[bool, str]:
        """
        检查两个梯度提交的一致性（用于检测异常）
        :param submission1: 第一个提交
        :param submission2: 第二个提交
        :return: (是否一致, 错误信息)
        """
        # 检查模型参数结构是否一致
        params1_keys = set(submission1.model_params.keys())
        params2_keys = set(submission2.model_params.keys())
        
        if params1_keys != params2_keys:
            return False, f"模型参数键不一致: {params1_keys} vs {params2_keys}"
        
        # 检查梯度参数结构是否一致（如果都有）
        if submission1.gradient_params and submission2.gradient_params:
            grad1_keys = set(submission1.gradient_params.keys())
            grad2_keys = set(submission2.gradient_params.keys())
            
            if grad1_keys != grad2_keys:
                return False, f"梯度参数键不一致: {grad1_keys} vs {grad2_keys}"
        
        return True, ""
    
    def detect_anomalous_gradients(self, submissions: List[ModelUpdateSubmission],
                                   similarity_threshold: float = 0.3) -> List[int]:
        """
        检测异常梯度（基于相似度）
        :param submissions: 梯度提交列表
        :param similarity_threshold: 相似度阈值（低于此值的视为异常）
        :return: 异常客户端ID列表
        """
        # 这里可以添加更复杂的异常检测逻辑
        # 目前只是返回空列表，实际应该基于相似度计算
        anomalous_clients = []
        
        # TODO: 实现基于相似度的异常检测
        # 1. 计算所有梯度的平均梯度
        # 2. 计算每个梯度与平均梯度的相似度
        # 3. 相似度低于阈值的视为异常
        
        return anomalous_clients

