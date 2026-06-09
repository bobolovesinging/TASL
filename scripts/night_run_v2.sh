#!/bin/bash
set -e

PY=/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/miniconda3/envs/tasl/bin/python3
TASL=/media/njit5/39e16d05-05ba-46b4-bc9b-47b3c98f1d4f/tlf/TASL

echo Night Run v2: MNIST plus CIFAR-10 Missing Attacks

cd 

echo Waiting for GPU0
while true; do
    gpu_util=0
    if [ "" -lt 30 ] 2>/dev/null; then
        echo GPU0 available util=%
        break
    fi
    sleep 20
done

CUDA_VISIBLE_DEVICES=0 nohup  -u scripts/exp2/exp2_run.py     --config /configs/exp2_mnist_v4.yaml     > /logs/exp2_mnist_v4.log 2>&1 &
MNIST_PID=
wait  2>/dev/null
echo MNIST done

CUDA_VISIBLE_DEVICES=0 nohup  -u scripts/exp2/exp2_run.py     --config /configs/exp2_cifar10_v4.yaml     > /logs/exp2_cifar10_v4.log 2>&1 &
CIFAR_PID=
wait  2>/dev/null
echo CIFAR-10 done

echo Night Run v2 complete
