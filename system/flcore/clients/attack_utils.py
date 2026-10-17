import random
from typing import List, Sequence, Tuple, Any, Optional

import torch


DatasetType = Sequence[Tuple[torch.Tensor, torch.Tensor]]


class AttackManager:
    """Utility class to orchestrate malicious client behaviours."""

    def __init__(self, config: Optional[dict] = None, num_classes: Optional[int] = None):
        self.config = config or {}
        self.num_classes = num_classes
        self.enabled = bool(self.config.get("enabled", False))
        self.start_round = int(self.config.get("start_round", 0))
        self.end_round = self.config.get("end_round")
        self.verbose = bool(self.config.get("verbose", False))

        malicious_ids = self.config.get("malicious_ids")
        if not malicious_ids:
            num_malicious = int(self.config.get("num_malicious", 0))
            if num_malicious > 0:
                malicious_ids = list(range(num_malicious))
        self.malicious_ids = set(malicious_ids or [])

        self.data_cfg = self.config.get("data_poisoning", {})
        self.model_cfg = self.config.get("model_poisoning", {})

    def is_malicious(self, client_id: int) -> bool:
        if not self.enabled:
            return False
        if not self.malicious_ids:
            return True
        return client_id in self.malicious_ids

    def _within_round_window(self, round_number: Optional[int]) -> bool:
        if round_number is None:
            return True
        if round_number < self.start_round:
            return False
        if self.end_round is not None and round_number > self.end_round:
            return False
        return True

    def _can_attack(self, client_id: int, round_number: Optional[int]) -> bool:
        return self.is_malicious(client_id) and self._within_round_window(round_number)

    def poison_dataset(
        self,
        dataset: DatasetType,
        round_number: Optional[int],
        client_id: int,
    ) -> DatasetType:
        if not dataset or not self._can_attack(client_id, round_number):
            return dataset
        if not self.data_cfg.get("enabled", False):
            return dataset

        ratio = float(self.data_cfg.get("poison_ratio", 0.0))
        if ratio <= 0:
            return dataset

        total = len(dataset)
        poison_count = max(1, int(total * ratio))
        poison_count = min(poison_count, total)
        poison_indices = set(random.sample(range(total), poison_count))

        strategy = self.data_cfg.get("strategy", "label_flip")
        target_label = self.data_cfg.get("target_label")
        flip_to_label = self.data_cfg.get("flip_to_label")
        noise_std = float(self.data_cfg.get("noise_std", 0.0))
        trigger_offset = float(self.data_cfg.get("trigger_offset", 0.0))

        poisoned_samples: List[Tuple[Any, Any]] = []
        for idx, (sample, label) in enumerate(dataset):
            new_sample = sample.clone() if torch.is_tensor(sample) else sample
            new_label = label.clone() if torch.is_tensor(label) else label

            if idx in poison_indices:
                if torch.is_tensor(new_label):
                    new_label = self._poison_label(
                        new_label,
                        strategy=strategy,
                        target_label=target_label,
                        flip_to_label=flip_to_label,
                    )
                if torch.is_tensor(new_sample):
                    if noise_std > 0:
                        new_sample = new_sample + torch.randn_like(new_sample) * noise_std
                    if trigger_offset != 0:
                        new_sample = new_sample + trigger_offset

            poisoned_samples.append((new_sample, new_label))

        if self.verbose and poison_indices:
            print(
                f"[Attack] Client {client_id} poisoned {len(poison_indices)} samples in round {round_number}."
            )
        return poisoned_samples

    def _poison_label(
        self,
        label: torch.Tensor,
        strategy: str,
        target_label: Optional[int],
        flip_to_label: Optional[int],
    ) -> torch.Tensor:
        poisoned = label.clone()
        current_value = int(poisoned.item()) if poisoned.numel() == 1 else None

        if strategy == "targeted":
            if target_label is None:
                return poisoned
            if current_value is not None and current_value != target_label:
                return poisoned
            new_label = flip_to_label if flip_to_label is not None else target_label
            return torch.full_like(poisoned, new_label)

        if strategy == "label_flip":
            if flip_to_label is not None:
                return torch.full_like(poisoned, flip_to_label)
            if current_value is not None and self.num_classes:
                new_label = (current_value + 1) % self.num_classes
                return torch.full_like(poisoned, new_label)

        return poisoned

    def poison_model(self, module, round_number: Optional[int], client_id: int):
        if module is None or not self._can_attack(client_id, round_number):
            return
        if not self.model_cfg.get("enabled", False):
            return

        strategy = self.model_cfg.get("strategy", "scaling")
        if strategy == "scaling":
            factor = float(self.model_cfg.get("scaling_factor", 1.0))
            if factor == 1.0:
                return
            for param in module.parameters():
                param.data.mul_(factor)
        elif strategy == "additive":
            noise_std = float(self.model_cfg.get("noise_std", 0.0))
            if noise_std <= 0:
                return
            for param in module.parameters():
                param.data.add_(torch.randn_like(param) * noise_std)

        if self.verbose:
            print(
                f"[Attack] Client {client_id} applied {strategy} model poisoning in round {round_number}."
            )

