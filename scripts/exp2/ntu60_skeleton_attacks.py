# -*- coding: utf-8 -*-
"""
NTU-60 Skeleton Attack Module
=============================
Four Byzantine data-layer attacks for NTU-60 skeleton data.

Data format: (C, T, V, M) = (3, 64, 25, 2)
  C=3 (x, y, z), T=64 frames, V=25 joints, M=2 persons

Attack types:
  - fgsm:      Fast Gradient Sign Method (adversarial perturbation)
  - pgd:       Projected Gradient Descent (iterative adversarial)
  - bone_len:  Random bone length scaling (skeleton structure attack)
  - hard_nobox: Joint coordinate perturbation without spatial constraints
"""

import torch
import torch.nn.functional as F

# ── NTU-RGB+D 25-joint skeleton topology (bone connections) ──
# Each tuple: (joint_a, joint_b) — indices are 0-based
# Based on NTU-RGB+D Kinect v2 skeleton definition
NTU_BONES = [
    # Spine (center → upward)
    (0, 1),    # base of spine → middle of spine
    (1, 20),   # middle of spine → neck
    (20, 2),   # neck → head
    (20, 3),   # neck → right shoulder
    (20, 8),   # neck → left shoulder (mirrored: actually 20→8)

    # Right arm
    (3, 4),    # right shoulder → right elbow
    (4, 5),    # right elbow → right wrist
    (5, 6),    # right wrist → right hand

    # Left arm
    (8, 9),    # left shoulder → left elbow
    (9, 10),   # left elbow → left wrist
    (10, 11),  # left wrist → left hand

    # Right hand (fingers)
    (6, 7),    # right hand → right fingertip
    (6, 22),   # right hand → right thumb

    # Left hand (fingers)
    (11, 12),  # left hand → left fingertip
    (11, 23),  # left hand → left thumb

    # Lower body
    (0, 16),   # base of spine → right hip
    (0, 12),   # base of spine → left hip (in NTU: 0→12)
    (16, 17),  # right hip → right knee
    (17, 18),  # right knee → right ankle
    (18, 19),  # right ankle → right foot
    (12, 13),  # left hip → left knee
    (13, 14),  # left knee → left ankle
    (14, 15),  # left ankle → left foot
    # Additional hand tips
    (7, 21),   # right fingertip → additional
    (22, 21),  # right thumb → additional
    (12, 24),  # left fingertip → additional
    (23, 24),  # left thumb → additional

    # Shoulder span
    (3, 8),    # right shoulder → left shoulder (horizontal)
]

# ── Attack Implementations ──


def fgsm_attack(model, x, y, epsilon=0.1):
    """
    Fast Gradient Sign Method — one-step adversarial perturbation.

    x_adv = x + epsilon * sign(grad_x Loss)

    Args:
        model: the model being trained (used to compute gradients)
        x: input tensor (B, 3, 64, 25, 2)
        y: label tensor (B,)
        epsilon: perturbation magnitude

    Returns:
        x_adv: perturbed input (detached)
    """
    model.eval()  # freeze running stats
    x_adv = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_adv), y)
    loss.backward()
    x_adv = x + epsilon * x_adv.grad.sign()
    return x_adv.detach()


def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, steps=10):
    """
    Projected Gradient Descent — iterative FGSM with L-inf projection.

    x_adv^(k+1) = Proj_B(x,ε)[ x_adv^(k) + alpha * sign(grad L) ]

    Args:
        model: the model
        x: input tensor (B, 3, 64, 25, 2)
        y: label tensor (B,)
        epsilon: max L-inf perturbation
        alpha: step size per iteration
        steps: number of PGD iterations

    Returns:
        x_adv: perturbed input (detached)
    """
    model.eval()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        loss.backward()
        # PGD step
        x_adv = x_adv + alpha * x_adv.grad.sign()
        # Project back to L-inf ball
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = x_adv.detach()
    return x_adv


def bone_length_attack(x, scale_std=0.3):
    """
    Bone Length Attack — randomly scale skeleton bone lengths.

    For each bone (joint_a, joint_b) in the skeleton, randomly scale the
    distance between connected joints. The direction (joint angle) is preserved,
    so the structure looks like a distorted human pose.

    Args:
        x: input tensor (B, 3, 64, 25, 2) — batch of skeleton sequences
        scale_std: std of log-normal scale factor

    Returns:
        x_perturbed: same shape as x
    """
    B, C, T, V, M = x.shape
    x_out = x.clone()

    for b in range(B):
        for t in range(T):
            for m in range(M):
                # Generate random scale factors for all bones
                scale_factors = torch.exp(
                    torch.randn(len(NTU_BONES)) * scale_std
                )

                for i, (ja, jb) in enumerate(NTU_BONES):
                    vec = x_out[b, :, t, jb, m] - x_out[b, :, t, ja, m]
                    center = x_out[b, :, t, ja, m]

                    # Scale the vector
                    scaled_vec = vec * scale_factors[i]

                    # Move joint jb relative to ja
                    x_out[b, :, t, jb, m] = center + scaled_vec

    # Clamp to avoid extreme values
    x_out = torch.clamp(x_out, x.min().item(), x.max().item())
    return x_out


def hard_nobox_attack(x, noise_std=0.2):
    """
    Hard No Box Attack — add Gaussian noise to joint coordinates.

    Unlike bone_length which preserves structure, this attack randomly
    perturbs all joint positions independently, creating physically
    impossible skeleton configurations.

    Args:
        x: input tensor (B, 3, 64, 25, 2)
        noise_std: standard deviation of Gaussian noise (relative to data range)

    Returns:
        x_perturbed: same shape as x
    """
    data_std = x.std().item() + 1e-8
    noise = torch.randn_like(x) * noise_std * data_std
    x_out = x + noise
    # Clamp to reasonable range
    return torch.clamp(x_out, -5.0, 5.0)


def apply_skeleton_attack(model, x, y, attack_name, **kwargs):
    """
    Apply a named skeleton attack to a batch of NTU-60 data.

    Args:
        model: the model (required for FGSM/PGD)
        x: input tensor (B, 3, 64, 25, 2)
        y: label tensor (B,)
        attack_name: one of 'fgsm', 'pgd', 'bone_len', 'hard_nobox'
        **kwargs: attack-specific parameters

    Returns:
        x_perturbed, y (unchanged)
    """
    if attack_name == "fgsm":
        epsilon = kwargs.get("fgsm_epsilon", 0.1)
        x_adv = fgsm_attack(model, x, y, epsilon=epsilon)
        return x_adv, y

    elif attack_name == "pgd":
        epsilon = kwargs.get("pgd_epsilon", 0.1)
        alpha = kwargs.get("pgd_alpha", 0.01)
        steps = kwargs.get("pgd_steps", 10)
        x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=alpha, steps=steps)
        return x_adv, y

    elif attack_name == "bone_len":
        scale_std = kwargs.get("bone_scale_std", 0.3)
        x_adv = bone_length_attack(x, scale_std=scale_std)
        return x_adv, y

    elif attack_name == "hard_nobox":
        noise_std = kwargs.get("nobox_noise_std", 0.2)
        x_adv = hard_nobox_attack(x, noise_std=noise_std)
        return x_adv, y

    else:
        raise ValueError(f"Unknown skeleton attack: {attack_name}")


# ── Quick test ──
if __name__ == "__main__":
    print("NTU-60 Skeleton Attack Module")
    print(f"  Bones defined: {len(NTU_BONES)}")
    # Generate dummy data
    x = torch.randn(2, 3, 64, 25, 2)
    y = torch.randint(0, 60, (2,))

    print(f"\n  Input shape: {x.shape}")
    print(f"  Labels: {y}")

    # Test bone_length
    x_bl = bone_length_attack(x)
    diff_bl = (x_bl - x).abs().mean().item()
    print(f"  Bone Length Attack — mean abs diff: {diff_bl:.4f}")

    # Test hard_nobox
    x_hn = hard_nobox_attack(x)
    diff_hn = (x_hn - x).abs().mean().item()
    print(f"  Hard NoBox Attack  — mean abs diff: {diff_hn:.4f}")

    print("\n  All attacks verified. FGSM/PGD require a model to test.")
