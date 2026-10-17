"""
相似度计算模块
计算客户端梯度与全局梯度的余弦相似度
"""
import torch
import numpy as np
from typing import Dict, Optional


class SimilarityCalculator:
    """相似度计算器：计算梯度之间的余弦相似度"""
    
    @staticmethod
    def flatten_gradients(gradients: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将梯度字典展平为一维张量
        :param gradients: 梯度字典 {param_name: tensor}
        :return: 展平的一维张量（统一为float32类型）
        """
        flat_list = []
        for name in sorted(gradients.keys()):
            param = gradients[name]
            if isinstance(param, dict):
                # 如果是summary模式，使用L2范数作为代表值
                if "l2" in param:
                    flat_list.append(torch.tensor([param["l2"]], dtype=torch.float32))
                else:
                    # 如果有mean和std，组合使用
                    mean_val = param.get("mean", 0.0)
                    std_val = param.get("std", 0.0)
                    flat_list.append(torch.tensor([mean_val, std_val], dtype=torch.float32))
            else:
                # 确保转换为float32
                param_flat = param.view(-1)
                if param_flat.dtype != torch.float32:
                    param_flat = param_flat.float()
                flat_list.append(param_flat)
        
        if not flat_list:
            return torch.tensor([], dtype=torch.float32)
        
        return torch.cat(flat_list)
    
    @staticmethod
    def cosine_similarity(grad1: Dict[str, torch.Tensor], 
                         grad2: Dict[str, torch.Tensor]) -> float:
        """
        计算两个梯度字典的余弦相似度
        :param grad1: 第一个梯度字典
        :param grad2: 第二个梯度字典
        :return: 余弦相似度（-1到1之间）
        """
        # 展平梯度
        flat1 = SimilarityCalculator.flatten_gradients(grad1)
        flat2 = SimilarityCalculator.flatten_gradients(grad2)
        
        if flat1.numel() == 0 or flat2.numel() == 0:
            return 0.0
        
        # 统一数据类型为 float32
        flat1 = flat1.float()
        flat2 = flat2.float()
        
        # 确保形状一致
        min_len = min(flat1.numel(), flat2.numel())
        flat1 = flat1[:min_len]
        flat2 = flat2[:min_len]
        
        # 计算余弦相似度
        dot_product = torch.dot(flat1, flat2)
        norm1 = torch.norm(flat1)
        norm2 = torch.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = (dot_product / (norm1 * norm2)).item()
        
        # 确保在[-1, 1]范围内
        similarity = max(-1.0, min(1.0, similarity))
        
        return similarity
    
    @staticmethod
    def calculate_similarities_to_global(
        client_gradients: Dict[int, Dict[str, torch.Tensor]],
        global_gradient: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[int, float]:
        """
        计算所有客户端梯度相对于全局梯度的余弦相似度
        :param client_gradients: {client_id: gradient_dict}
        :param global_gradient: 全局梯度字典（如果为None，则计算平均梯度）
        :return: {client_id: similarity}
        """
        similarities = {}
        
        # 如果没有提供全局梯度，计算平均梯度
        if global_gradient is None:
            if not client_gradients:
                return similarities
            
            # 计算平均梯度
            global_gradient = {}
            first_client_id = list(client_gradients.keys())[0]
            first_grad = client_gradients[first_client_id]
            
            for name in first_grad.keys():
                if isinstance(first_grad[name], dict):
                    # Summary模式：计算统计量的平均值
                    global_gradient[name] = {
                        "mean": np.mean([client_gradients[cid][name].get("mean", 0.0) 
                                        for cid in client_gradients if name in client_gradients[cid]]),
                        "std": np.mean([client_gradients[cid][name].get("std", 0.0) 
                                       for cid in client_gradients if name in client_gradients[cid]]),
                        "l2": np.mean([client_gradients[cid][name].get("l2", 0.0) 
                                      for cid in client_gradients if name in client_gradients[cid]])
                    }
                else:
                    # 全量模式：计算平均值
                    stacked = torch.stack([
                        client_gradients[cid][name] 
                        for cid in client_gradients 
                        if name in client_gradients[cid]
                    ])
                    global_gradient[name] = torch.mean(stacked, dim=0)
        
        # 计算每个客户端与全局梯度的相似度
        for client_id, client_grad in client_gradients.items():
            similarity = SimilarityCalculator.cosine_similarity(
                client_grad, global_gradient
            )
            similarities[client_id] = similarity
        
        return similarities

