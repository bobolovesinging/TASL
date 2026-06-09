#!/bin/bash
# Night run: fill missing MNIST + CIFAR-10 attack experiments
# Waits for GPU0 to be free, then runs sequentially

set -e

PY=/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/miniconda3/envs/tasl/bin/python3
TASL=/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/TASL
LOG=/logs
CFG=/configs

echo === Night Run: MNIST + CIFAR-10 Missing Attacks ===
echo Start: Tue Jun 2 01:21:27 2026

# Wait for GPU0 to be free (centralized training PID check)
echo Waiting for GPU0 to be free...
while nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | grep '^0,' | grep -q '[1-9][0-9]'; do
    sleep 30
done
sleep 10  # grace period
echo GPU0 is free! Tue Jun 2 01:21:27 2026

# ===== MNIST: min_max + scaling =====
echo 
echo === Running MNIST v4: min_max + scaling ===
cd 
CUDA_VISIBLE_DEVICES=0  -u scripts/exp2/exp2_run.py     --config /exp2_mnist_v4.yaml     2>&1 | tee /exp2_mnist_v4.log
echo MNIST v4 done! Tue Jun 2 01:21:27 2026

# ===== CIFAR-10: gaussian_noise + min_max + scaling =====
echo 
echo === Running CIFAR-10 v4: gaussian + min_max + scaling ===
CUDA_VISIBLE_DEVICES=0  -u scripts/exp2/exp2_run.py     --config /exp2_cifar10_v4.yaml     2>&1 | tee /exp2_cifar10_v4.log
echo CIFAR-10 v4 done! Tue Jun 2 01:21:27 2026

echo 
echo === Night run complete! Tue Jun 2 01:21:27 2026 ===
