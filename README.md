# TASL

TASL training with trust-weighted aggregation, two auditors, and recovery.

Use Python 3.10. For CUDA, use the PyTorch wheel index for your CUDA build.

    python -m pip install -r requirements.txt
    python run.py --dataset fashionmnist --output-dir outputs/fashionmnist

The trainer downloads MNIST or Fashion-MNIST through torchvision. It selects CUDA when available and accepts --device cpu or --device cuda. The default is 50 rounds with 10 clients, seed 42, and a Dirichlet concentration of 0.3. Training writes model.pt and trust_v1v2.jsonl to the new output directory.
