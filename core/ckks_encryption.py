"""
CKKS homomorphic encryption utilities + KGC.
"""

from typing import Dict, Optional, List, Any
import warnings
import numpy as np
import torch

from .crypto_utils import generate_ed25519_keypair

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    ts = None
    TENSEAL_AVAILABLE = False
    warnings.warn("tenseal库未安装，CKKS加密功能将不可用。请使用: pip install tenseal")


class CKKSEncryption:
    """CKKS全同态加密类"""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        global_scale: float = 2**40,
    ):
        self.context = None
        self.public_key = None
        self.secret_key = None
        self.poly_modulus_degree = poly_modulus_degree
        self.coeff_mod_bit_sizes = coeff_mod_bit_sizes
        self.global_scale = float(global_scale)

        if TENSEAL_AVAILABLE:
            self._init_context()
        else:
            print("[WARNING] tenseal未安装，使用模拟加密模式（不加密）")

    def _init_context(self):
        if not TENSEAL_AVAILABLE:
            return
        try:
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.poly_modulus_degree,
                coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
            )
            self.context.generate_galois_keys()
            self.context.global_scale = self.global_scale
            self.secret_key = self.context.secret_key()
            self.public_key = self.context.public_key()
            print(f"[CKKS] 初始化成功 (N={self.poly_modulus_degree}, scale={self.global_scale})")
        except Exception as e:
            print(f"[ERROR] CKKS上下文初始化失败: {e}")
            self.context = None

    def get_public_context(self):
        if not TENSEAL_AVAILABLE or self.context is None:
            return None
        public_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_modulus_degree,
            coeff_mod_bit_sizes=self.coeff_mod_bit_sizes,
        )
        public_context.global_scale = self.context.global_scale
        public_context.generate_galois_keys()
        public_context.make_context_public()
        return public_context

    def encrypt_gradients(self, gradients: Dict[str, torch.Tensor], public_context: Optional[Any] = None) -> Dict[str, Any]:
        if not TENSEAL_AVAILABLE or public_context is None:
            return {
                name: {
                    "encrypted": False,
                    "data": grad.cpu().detach().numpy().tolist() if isinstance(grad, torch.Tensor) else grad,
                    "shape": list(grad.shape) if isinstance(grad, torch.Tensor) else None,
                }
                for name, grad in gradients.items()
            }

        encrypted_grads: Dict[str, Any] = {}
        slot_count = public_context.slot_count()
        for name, grad in gradients.items():
            if not isinstance(grad, torch.Tensor):
                encrypted_grads[name] = {"encrypted": False, "data": grad, "shape": None}
                continue

            grad_np = grad.detach().cpu().numpy().flatten()
            if len(grad_np) <= slot_count:
                padded = np.pad(grad_np, (0, slot_count - len(grad_np)), mode="constant")
                encrypted = ts.ckks_vector(public_context, padded.tolist())
                encrypted_grads[name] = {
                    "encrypted": True,
                    "ciphertext": encrypted.serialize(),
                    "shape": list(grad.shape),
                    "original_length": len(grad_np),
                }
            else:
                chunks = []
                for i in range(0, len(grad_np), slot_count):
                    chunk = grad_np[i : i + slot_count]
                    padded = np.pad(chunk, (0, slot_count - len(chunk)), mode="constant")
                    encrypted_chunk = ts.ckks_vector(public_context, padded.tolist())
                    chunks.append({"ciphertext": encrypted_chunk.serialize(), "start_idx": i, "length": len(chunk)})
                encrypted_grads[name] = {
                    "encrypted": True,
                    "chunks": chunks,
                    "shape": list(grad.shape),
                    "original_length": len(grad_np),
                }
        return encrypted_grads

    def decrypt_gradients(self, encrypted_gradients: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        decrypted_grads: Dict[str, torch.Tensor] = {}
        if not TENSEAL_AVAILABLE or self.context is None:
            for name, enc_grad in encrypted_gradients.items():
                if isinstance(enc_grad, dict):
                    data = enc_grad.get("data", enc_grad)
                    shape = enc_grad.get("shape")
                    t = torch.tensor(data, dtype=torch.float32)
                    decrypted_grads[name] = t.reshape(shape) if shape is not None else t
                else:
                    decrypted_grads[name] = torch.tensor(enc_grad, dtype=torch.float32)
            return decrypted_grads

        for name, enc_grad in encrypted_gradients.items():
            if not isinstance(enc_grad, dict) or not enc_grad.get("encrypted", False):
                data = enc_grad.get("data", enc_grad) if isinstance(enc_grad, dict) else enc_grad
                shape = enc_grad.get("shape") if isinstance(enc_grad, dict) else None
                t = torch.tensor(data, dtype=torch.float32)
                decrypted_grads[name] = t.reshape(shape) if shape is not None else t
                continue

            if "ciphertext" in enc_grad:
                ciphertext = ts.ckks_vector_from(self.context, enc_grad["ciphertext"])
                dec = np.array(ciphertext.decrypt())[: enc_grad.get("original_length", 0)]
            else:
                parts = []
                for chunk in enc_grad.get("chunks", []):
                    ciphertext = ts.ckks_vector_from(self.context, chunk["ciphertext"])
                    parts.append(np.array(ciphertext.decrypt())[: chunk["length"]])
                dec = np.concatenate(parts) if parts else np.array([], dtype=np.float32)

            shape = enc_grad.get("shape")
            t = torch.tensor(dec, dtype=torch.float32)
            decrypted_grads[name] = t.reshape(shape) if shape is not None else t

        return decrypted_grads


class KeyGenerationCenter:
    """密钥生成中心：生成 CKKS 与身份签名密钥"""

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes: List[int] = [60, 40, 40, 60],
        global_scale: float = 2**40,
    ):
        self.ckks = CKKSEncryption(
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            global_scale=global_scale,
        )
        self.public_context = self.ckks.get_public_context()
        self.identity_keys: Dict[int, Dict[str, str]] = {}

    def register_client_identity(self, client_id: int) -> Dict[str, str]:
        if client_id not in self.identity_keys:
            self.identity_keys[client_id] = generate_ed25519_keypair()
        return dict(self.identity_keys[client_id])

    def register_clients(self, client_ids: List[int]) -> Dict[int, Dict[str, str]]:
        for cid in client_ids:
            self.register_client_identity(int(cid))
        return {cid: dict(keys) for cid, keys in self.identity_keys.items()}

    def get_client_private_key(self, client_id: int) -> Optional[str]:
        keys = self.identity_keys.get(client_id)
        return keys.get("private_key") if keys else None

    def get_client_public_key(self, client_id: int) -> Optional[str]:
        keys = self.identity_keys.get(client_id)
        return keys.get("public_key") if keys else None

    def get_all_public_keys(self) -> Dict[int, str]:
        return {cid: k["public_key"] for cid, k in self.identity_keys.items() if "public_key" in k}

    def get_public_key(self):
        return self.public_context

    def get_secret_key(self):
        return self.ckks.context if self.ckks.context else None

    def get_decryption_context(self):
        return self.ckks.context
