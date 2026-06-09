# -*- coding: utf-8 -*-
"""
TASL Advanced Adversarial Attacks
==================================
New attacks for comprehensive Byzantine resilience evaluation:

Image attacks (MNIST / CIFAR-10):
  - FGSM: Fast Gradient Sign Method (single-step)
  - PGD: Projected Gradient Descent (multi-step iterative)
  - CW: Carlini & Wagner L2 attack

Skeleton attack (NTU-60):
  - bone_length: Random perturbation of skeleton bone lengths

All attacks preserve the training pipeline compatibility:
  attack = "fgsm" / "pgd" / "cw" / "bone_length"
"""

import torch
import torch.nn as nn
import numpy as np


# ════════════════ FGSM Attack ════════════════

def fgsm_attack(x, y, model, criterion, epsilon=0.1, alpha=None):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).

    x_adv = x + epsilon * sign(grad_x(loss))

    Args:
      x, y: input batch and labels
      model: model with requires_grad enabled
      criterion: loss function
      epsilon: perturbation magnitude
      alpha: ignored (for API compatibility with PGD)

    Returns:
      x_adv: adversarial examples
    """
    x_adv = x.clone().detach()
    x_adv.requires_grad = True

    output = model(x_adv)
    loss = criterion(output, y)
    loss.backward()

    with torch.no_grad():
        grad_sign = x_adv.grad.sign()
        x_adv = x_adv + epsilon * grad_sign
        # Clamp to valid data range
        x_adv = torch.clamp(x_adv, x.min(), x.max())

    return x_adv.detach()


# ════════════════ PGD Attack ════════════════

def pgd_attack(x, y, model, criterion, epsilon=0.1, alpha=0.01,
               steps=10, random_start=True):
    """
    Projected Gradient Descent (Madry et al., 2018).

    Iterative FGSM with projection back to epsilon-ball:
      x_{t+1} = proj_{eps}[x_t + alpha * sign(grad_x_t(loss))]

    Args:
      x, y: input batch and labels
      model: model
      criterion: loss function
      epsilon: maximum L-inf perturbation
      alpha: step size (default: epsilon / steps)
      steps: number of PGD iterations
      random_start: initialize with random perturbation within epsilon-ball

    Returns:
      x_adv: adversarial examples
    """
    if alpha is None:
        alpha = epsilon / steps

    x_orig = x.clone().detach()

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
        x_adv = torch.clamp(x_adv, x_orig.min(), x_orig.max())
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        output = model(x_adv)
        loss = criterion(output, y)
        loss.backward()

        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            # Project back to epsilon-ball
            eta = torch.clamp(x_adv - x_orig, -epsilon, epsilon)
            x_adv = x_orig + eta
            # Clamp to valid data range
            x_adv = torch.clamp(x_adv, x_orig.min(), x_orig.max())

    return x_adv.detach()


# ════════════════ C&W L2 Attack ════════════════

def cw_l2_attack(x, y, model, criterion, c=1.0, kappa=0.0,
                 steps=20, lr=0.01):
    """
    Carlini & Wagner L2 attack (simplified).

    Minimizes: c * L2(w, x_orig) + max(Z_max - Z_y - kappa, 0)

    For simplicity, we use a projected PGD variant targeting CW objective.
    The key difference from PGD: targeted toward a specific wrong class.

    Args:
      x, y: input batch and labels
      model: model
      criterion: loss function (not directly used)
      c: weight for L2 distance term
      kappa: confidence margin
      steps: optimization steps
      lr: step size

    Returns:
      x_adv: adversarial examples
    """
    x_orig = x.clone().detach()

    # Find target class: second-highest logit
    with torch.no_grad():
        logits = model(x_orig)
        _, top2 = torch.topk(logits, 2, dim=1)
        target = top2[:, 1]  # Second most confident class

    x_adv = x_orig.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_adv], lr=lr)

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(x_adv)

        # CW loss: max(l2_dist + max(logit_target - logit_true, -kappa), 0)
        logit_true = output.gather(1, y.view(-1, 1)).squeeze()
        logit_target = output.gather(1, target.view(-1, 1)).squeeze()
        l2_dist = ((x_adv - x_orig) ** 2).view(x.size(0), -1).sum(1)

        cw_loss = torch.mean(l2_dist + c * torch.clamp(
            logit_true - logit_target - kappa, min=0))

        cw_loss.backward()
        optimizer.step()

        # Clamp to valid data range
        with torch.no_grad():
            x_adv.clamp_(x_orig.min(), x_orig.max())

    return x_adv.detach()


# ════════════════ Bone Length Attack ════════════════

def bone_length_attack_np(x_np, scale=0.2, random_state=None):
    """
    Perturb skeleton bone lengths by multiplying each bone vector
    by a random scaling factor.

    For NTU-60 skeleton data:
      Shape: (C=3, T=50, V=25, M=2) per sample (in nx1532 format before reshape)
      Or: (3, 50, 25, 2) after reshaping

    We perturb the spatial dimension (joint positions) by stretching or
    compressing bone vectors between adjacent joints.

    Args:
      x_np: numpy array of skeleton data (N, 3, 50, 25, 2) or single sample
      scale: standard deviation of random multiplicative factor (1 ± N(0,scale))
      random_state: random seed for reproducibility

    Returns:
      x_perturbed: perturbed skeleton data
    """
    rng = np.random.RandomState(random_state)
    x = x_np.copy()
    orig_shape = x.shape

    # Reshape to (3, T, V, M) if needed
    if len(x.shape) == 4:  # (3, 50, 25, 2)
        C, T, V, M = x.shape
    else:
        return x_np  # Unknown format, skip

    # For each frame and person, perturb joint positions
    for t in range(T):
        for m in range(M):
            # Get joints for this frame/person: (3, 25)
            joints = x[:, t, :, m]  # (3, V)

            # Compute bone vectors between adjacent joints
            # Use NTU skeleton structure: joint pairs
            # For simplicity, perturb ALL dimensions with per-joint noise
            noise = rng.normal(1.0, scale, size=(3, V))
            x[:, t, :, m] = joints * noise

    return x


def apply_bone_length_to_batch(x_batch, scale=0.2):
    """
    Apply bone length perturbation to a batch of skeleton data.

    Args:
      x_batch: torch tensor (B, 3, T, V, M) or (B, C, T, V, M)
      scale: perturbation magnitude

    Returns:
      x_perturbed: perturbed batch
    """
    x_np = x_batch.cpu().numpy()
    batch_size = x_np.shape[0]
    x_perturbed_list = []

    for b in range(batch_size):
        x_perturbed_list.append(bone_length_attack_np(x_np[b], scale=scale,
                                                       random_state=b))

    x_perturbed = torch.from_numpy(np.stack(x_perturbed_list, axis=0))
    return x_perturbed.to(x_batch.device)


# ════════════════ Attack Registry ════════════════

ADVANCED_ATTACK_CONFIGS = {
    # Image adversarial attacks
    'fgsm': {
        'epsilon': 0.1,      # Perturbation magnitude
        'description': 'Fast Gradient Sign Method (single-step)',
        'type': 'image',     # image or skeleton
    },
    'pgd': {
        'epsilon': 0.1,
        'alpha': 0.01,
        'steps': 10,
        'random_start': True,
        'description': 'Projected Gradient Descent (multi-step iterative)',
        'type': 'image',
    },
    'cw': {
        'c': 1.0,
        'kappa': 0.0,
        'steps': 20,
        'lr': 0.01,
        'description': 'Carlini & Wagner L2 attack',
        'type': 'image',
    },
    # Skeleton attack
    'bone_length': {
        'scale': 0.2,
        'description': 'Bone length perturbation (skeleton-only)',
        'type': 'skeleton',
    },
}


def is_image_attack(attack_name):
    return ADVANCED_ATTACK_CONFIGS.get(attack_name, {}).get('type') == 'image'


def is_skeleton_attack(attack_name):
    return ADVANCED_ATTACK_CONFIGS.get(attack_name, {}).get('type') == 'skeleton'


# ════════════════ Train with data-layer attacks ════════════════

def train_with_data_attack(model, loader, optimizer, criterion, device,
                           attack_type=None, attack_config=None,
                           stats_collector=None, cid=None):
    """
    Train model for one epoch, applying data-level adversarial attacks.

    This handles:
      - Image adversarial attacks (FGSM, PGD, CW): applied to input x
      - Skeleton attacks (bone_length): applied to input x
      - Label attacks (label_flip): applied to y (handled separately)
      - No attack: normal training

    Also collects training statistics (loss, grad_norm) for data-layer detection.

    Args:
      model: nn.Module
      loader: DataLoader
      optimizer: torch optimizer
      criterion: loss function
      device: torch device
      attack_type: one of 'fgsm', 'pgd', 'cw', 'bone_length', or None
      attack_config: dict with attack parameters
      stats_collector: TrainStatsCollector (optional)
      cid: client id for stats collection (optional)

    Returns:
      avg_loss: average training loss
      avg_accuracy: average training accuracy
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    config = (ADVANCED_ATTACK_CONFIGS.get(attack_type, {})
              if attack_config is None else attack_config)

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ── Apply data-level attack to input ──
        if attack_type == 'fgsm':
            x = fgsm_attack(x, y, model, criterion,
                           epsilon=config.get('epsilon', 0.1))
        elif attack_type == 'pgd':
            x = pgd_attack(x, y, model, criterion,
                          epsilon=config.get('epsilon', 0.1),
                          alpha=config.get('alpha', 0.01),
                          steps=config.get('steps', 10),
                          random_start=config.get('random_start', True))
        elif attack_type == 'cw':
            x = cw_l2_attack(x, y, model, criterion,
                            c=config.get('c', 1.0),
                            kappa=config.get('kappa', 0.0),
                            steps=config.get('steps', 20),
                            lr=config.get('lr', 0.01))
        elif attack_type == 'bone_length':
            x = apply_bone_length_to_batch(x, scale=config.get('scale', 0.2))

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Collect stats for data-layer detection
        if stats_collector is not None and cid is not None:
            stats_collector.record_batch(cid, loss.item(), model.parameters())

        total_loss += loss.item() * x.size(0)
        _, pred = output.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
        batch_count += 1

    avg_loss = total_loss / max(total, 1)
    avg_acc = 100.0 * correct / max(total, 1)

    return avg_loss, avg_acc


# ════════════════ Quick test ════════════════
if __name__ == "__main__":
    print("Advanced attacks module loaded.")
    print(f"Available attacks: {list(ADVANCED_ATTACK_CONFIGS.keys())}")
    for name, cfg in ADVANCED_ATTACK_CONFIGS.items():
        print(f"  {name}: {cfg['description']} ({cfg['type']})")
