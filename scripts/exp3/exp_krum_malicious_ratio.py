import os
import sys
import argparse
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
blockchain_dir = os.path.dirname(script_dir)
sys.path.insert(0, script_dir)
sys.path.insert(0, blockchain_dir)

import exp_accuracy_vs_epoch as exp


def plot_krum_ratio_results(results, epochs, attack, seed):
    plt.figure(figsize=(10, 6))
    for ratio, accs in results.items():
        plt.plot(range(1, epochs + 1), accs, label=f'Krum (ratio={ratio:.1f})')

    plt.xlabel('Epoch')
    plt.xticks(range(1, epochs + 1, max(1, epochs // 10)))
    plt.ylabel('Accuracy (%)')
    plt.title(f'Krum under {attack} (seed={seed})')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(
        blockchain_dir,
        'results',
        f'exp_krum_ratio_compare_{attack}_epochs{epochs}_seed{seed}.png'
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--attack', type=str, default='label_flip', choices=['label_flip', 'backdoor'])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    args = parser.parse_args()

    exp.set_seed(args.seed)

    dataset_name = args.dataset
    attack_type = args.attack
    epochs = args.epoch
    client_num = 10
    ratios = [0.1, 0.4, 0.5]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client_loaders = exp.get_client_loaders_iid(client_num=client_num, batch_size=128)
    test_loader = exp.get_test_loader(batch_size=128)

    results = {}

    for ratio in ratios:
        num_malicious = int(client_num * ratio)
        rng = np.random.RandomState(args.seed)
        malicious_clients = sorted(rng.choice(list(range(client_num)), size=num_malicious, replace=False).tolist())

        print(f"[Krum Experiment] ratio={ratio}, malicious_clients={malicious_clients}")

        attack_config = None
        if exp.HAS_ATTACK:
            attack_config = exp.AttackConfig()
            attack_config.enable_attack = True
            attack_config.malicious_clients = malicious_clients
            attack_config.attack_type = attack_type
            if attack_type == 'label_flip':
                attack_config.label_flip_map = {i: (9 - i) for i in range(10)}
            elif attack_type == 'backdoor':
                pattern = torch.zeros((1, 28, 28))
                pattern[0, 24:, 24:] = 1.0
                attack_config.backdoor_pattern = pattern
                attack_config.backdoor_target_label = 0

        accs = exp.train_fl_epoch(
            client_loaders,
            test_loader,
            device,
            epochs,
            client_num,
            malicious_clients,
            'krum',
            attack_config,
            local_epochs=args.local_epochs,
        )
        results[ratio] = accs

    plot_krum_ratio_results(results, epochs, attack_type, args.seed)

    csv_path = os.path.join(blockchain_dir, 'results', 'results_krum_ratio_compare.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'attack', 'ratio', 'seed', 'epoch', 'accuracy'])
        for ratio, accs in results.items():
            for ep, acc in enumerate(accs, start=1):
                writer.writerow([dataset_name, attack_type, ratio, args.seed, ep, float(acc)])

    print(f"Saved CSV to {csv_path}")


if __name__ == '__main__':
    main()
