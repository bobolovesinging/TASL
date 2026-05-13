import torch
import numpy as np

class RobustWeightedAggregator:
    def __init__(self, previous_global_update=None):
        # 存储上一轮的全局更新作为参考方向 (Consensus Momentum)
        self.v_ref = previous_global_update 

    def compute_trust_weights(self, client_updates, current_v_ref):
        """
        计算每个客户端的信任权重 (ReLU Trust Scoring)
        client_updates: dict {client_id: flattened_tensor}
        current_v_ref: flattened_tensor (上一轮的全局位移)
        """
        weights = {}
        if current_v_ref is None:
            # 第一轮没有参考，默认平权
            return {cid: 1.0 / len(client_updates) for cid in client_updates.keys()}

        # 确保参考矢量是单位向量
        v_ref_norm = torch.norm(current_v_ref) + 1e-9
        
        for cid, update in client_updates.items():
            # 1. 计算余弦相似度
            update_norm = torch.norm(update) + 1e-9
            cosine_sim = torch.dot(update, current_v_ref) / (update_norm * v_ref_norm)
            
            # 2. ReLU 评分逻辑: max(0, sim)
            # 负相关(投毒)权重归零, 正相关(好人)按一致程度分配权重
            weights[cid] = max(0.0, float(cosine_sim))
            
        # 3. 归一化权重，使总和为 1
        sum_w = sum(weights.values())
        if sum_w > 1e-9:
            weights = {cid: w / sum_w for cid, w in weights.items()}
        else:
            # 如果全员负相关(极端情况)，本轮放弃更新
            weights = {cid: 0.0 for cid in weights}
            
        return weights

    def aggregate(self, client_updates, trust_weights):
        """
        执行加权聚合
        client_updates: dict {client_id: state_dict}
        """
        aggregated_update = None
        
        for cid, weight in trust_weights.items():
            if weight <= 0: continue
            
            update = client_updates[cid] # 这是一个 state_dict
            if aggregated_update is None:
                aggregated_update = {k: torch.zeros_like(v) for k, v in update.items()}
            
            for k in aggregated_update:
                aggregated_update[k] += weight * update[k]
                
        return aggregated_update