"""
Cryptographic utilities for PBFL identity signatures.
Implements Ed25519 signing/verification with deterministic payload serialization.
"""

import base64
import json
from typing import Any, Dict

import torch
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def serialize_for_signing(data: Any) -> bytes:
    normalized = _to_serializable(data)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")


def generate_ed25519_keypair() -> Dict[str, str]:
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()

    sk_bytes = sk.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pk_bytes = pk.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    return {
        "private_key": base64.b64encode(sk_bytes).decode("utf-8"),
        "public_key": base64.b64encode(pk_bytes).decode("utf-8"),
    }


def sign_data(private_key: str, data: Any) -> str:
    sk = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key.encode("utf-8")))
    payload = serialize_for_signing(data)
    sig = sk.sign(payload)
    return base64.b64encode(sig).decode("utf-8")


def verify_signature(public_key: str, data: Any, signature: str) -> bool:
    try:
        pk_bytes = base64.b64decode(public_key.encode("utf-8"), validate=True)
        sig = base64.b64decode(signature.encode("utf-8"), validate=True)
        if len(pk_bytes) != 32 or len(sig) != 64:
            return False
        pk = Ed25519PublicKey.from_public_bytes(pk_bytes)
        payload = serialize_for_signing(data)
        pk.verify(sig, payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False
