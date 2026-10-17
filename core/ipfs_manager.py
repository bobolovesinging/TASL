"""
Mock IPFS Manager
模拟 IPFS 的本地文件存储实现 (Bit-Exact Binary Version)
功能：
1. add(data) -> cid: 将数据 (state_dict) 使用 torch.save 保存为二进制，生成 SHA256 CID
2. get(cid) -> data: 从本地读取二进制文件并 torch.load 还原
"""
import os
import hashlib
import torch
import shutil
import io
from typing import Any, Union, Dict

class MockIPFSManager:
    """本地模拟 IPFS 管理器 (Bit-Exact)"""
    
    def __init__(self, storage_dir: str = "storage/ipfs_mock"):
        """
        初始化
        :param storage_dir: 本地存储目录
        """
        if not os.path.isabs(storage_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # block_chain/
            self.storage_dir = os.path.join(base_dir, storage_dir)
        else:
            self.storage_dir = storage_dir
            
        os.makedirs(self.storage_dir, exist_ok=True)
        print(f"[MockIPFS] 初始化存储目录 (Binary Mode): {self.storage_dir}")

    def add(self, data: Union[Dict, Any]) -> str:
        """
        上传数据到 Mock IPFS
        :param data: 模型参数字典 (state_dict) 或其他 PyTorch 对象
        :return: CID (SHA256 Hash of binary content)
        """
        # 1. 序列化为二进制流 (InMemory)
        buffer = io.BytesIO()
        torch.save(data, buffer)
        serialized_data = buffer.getvalue()

        # 2. 计算二进制流的哈希 (CID)
        cid = hashlib.sha256(serialized_data).hexdigest()

        # 3. 保存文件 (Bit-Exact Dump)
        file_path = os.path.join(self.storage_dir, cid)
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            # print(f"[MockIPFS] Uploaded {cid} ({len(serialized_data)} bytes)")
        
        return cid

    def get(self, cid: str, map_location=None) -> Any:
        """
        从 Mock IPFS 下载数据
        :param cid: 内容 ID
        :param map_location: torch.load 的设备映射
        :return: 反序列化后的原始数据
        """
        if not cid:
            return None
            
        file_path = os.path.join(self.storage_dir, cid)
        
        if not os.path.exists(file_path):
            print(f"[MockIPFS] Error: CID {cid} not found locally.")
            return None
            
        try:
            # 直接加载二进制文件
            data = torch.load(file_path, map_location=map_location)
            return data
        except Exception as e:
            print(f"[MockIPFS] 反序列化失败 {cid}: {e}")
            return None

    def clear_storage(self):
        """清空存储 (测试用)"""
        if os.path.exists(self.storage_dir):
            shutil.rmtree(self.storage_dir)
            os.makedirs(self.storage_dir, exist_ok=True)
            print("[MockIPFS] Storage cleared.")
