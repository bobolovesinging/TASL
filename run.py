"""Train TASL with the Executor and both audit roles."""

import argparse
import os
from pathlib import Path

import torch

from src.data import load_client_data
from src.governance import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mnist", "fashionmnist"], default="fashionmnist")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for training output")
    args = parser.parse_args()

    if args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir already exists")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cuda_available = torch.cuda.is_available()
    if args.device == "cuda" and not cuda_available:
        parser.error("CUDA is unavailable")
    device = torch.device(("cuda" if cuda_available else "cpu") if args.device == "auto" else args.device)
    torch.use_deterministic_algorithms(True)

    clients, test = load_client_data(
        args.dataset, Path(__file__).resolve().parent / "data",
        clients=10, batch_size=128, alpha=0.3, seed=args.seed,
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=False)
    accuracy = train(clients, test, args.rounds, device, args.seed, args.output_dir)
    print(f"Final accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
