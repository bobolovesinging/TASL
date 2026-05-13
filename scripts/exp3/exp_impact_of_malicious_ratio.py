
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import defaultdict

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
blockchain_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(blockchain_dir)
sys.path.insert(0, project_root)
core_dir = os.path.join(blockchain_dir, 'core')
sys.path.insert(0, core_dir)

from blockchain_node import BlockchainNetwork
from gradient_aggregator import GradientAggregator
from model import FedAvgCNN
from utils_data import get_data_loaders

# Import attack module
try:
    from attack_module import (
        AttackConfig,
        apply_label_flip_attack,
        apply_backdoor_attack,
        apply_random_attack,
        apply_scale_attack,
        apply_model_poison_attack,
    )
    HAS_ATTACK = True
except ImportError:
    HAS_ATTACK = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_experiment(dataset_name='mnist', 
                   malicious_ratios=[0.1, 0.2, 0.3, 0.4, 0.5], 
                   methods=['avg', 'psdl', 'bulyan', 'krum', 'trimmed_mean'],
                   epochs=30,
                   client_num=10,
                   attack_type='label_flip'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Experiment on {dataset_name} with {attack_type} attack")
    
    # 1. Prepare Data
    client_loaders, test_loader = get_data_loaders(dataset_name, client_num)
    
    results = {m: [] for m in methods}
    
    for method in methods:
        print(f"\nTesting Method: {method}")
        for ratio in malicious_ratios:
            print(f"  Malicious Ratio: {ratio}")
            
            # Setup Attack Config
            num_malicious = int(client_num * ratio)
            malicious_clients = list(range(num_malicious))
            
            attack_config = None
            if HAS_ATTACK:
                attack_config = AttackConfig()
                attack_config.enable_attack = True
                attack_config.malicious_clients = malicious_clients
                attack_config.attack_type = attack_type
                if attack_type == 'label_flip':
                    # Flip 0->9, 1->8 ...
                    attack_config.label_flip_map = {i: 9-i for i in range(5)} 
                elif attack_type == 'backdoor':
                    # Square pattern at bottom right
                    pattern = torch.zeros((1, 28, 28))
                    pattern[0, 24:, 24:] = 1.0
                    attack_config.backdoor_pattern = pattern
                    attack_config.backdoor_target_label = 0
            
            # Run Training
            final_acc = train_fl(client_loaders, test_loader, device, 
                                 epochs, client_num, malicious_clients, 
                                 method, attack_config)
            
            results[method].append(final_acc)
            print(f"  -> Accuracy: {final_acc:.2f}%")
            
    # Plotting
    plot_results(results, malicious_ratios, dataset_name, attack_type)

def train_fl(client_loaders, test_loader, device, epochs, client_num, malicious_clients, method, attack_config,
             sign_flip_scale: float = -5.0, random_noise_std: float = 5.0):
    # Initialize Global Model
    global_model = FedAvgCNN().to(device)
    global_params = global_model.state_dict()
    criterion = nn.CrossEntropyLoss()
    
    aggregator = GradientAggregator(similarity_threshold=0.5)
    
    for epoch in range(epochs):
        client_gradients = {}
        sample_counts = {}
        
        # All clients participate (C=1.0)
        for client_id in range(client_num):
            model = FedAvgCNN().to(device)
            model.load_state_dict(global_params)
            model.train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            loader = client_loaders[client_id]
            is_malicious = (client_id in malicious_clients)
            
            total_loss = 0
            total_samples = 0
            
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                
                # Apply Attacks during training (Data Poisoning)
                if is_malicious and attack_config:
                     if attack_config.attack_type == 'label_flip':
                         target = apply_label_flip_attack(target, attack_config.label_flip_map)
                     elif attack_config.attack_type == 'backdoor':
                         data, target = apply_backdoor_attack(data, target, 
                                                            attack_config.backdoor_pattern, 
                                                            attack_config.backdoor_target_label)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_samples += len(target)
            
            # Compute local update; for malicious nodes, poison via attack_module APIs
            local_params = model.state_dict()
            if is_malicious and attack_config:
                if attack_config.attack_type == 'sign_flip':
                    local_params = apply_scale_attack(
                        model_params=local_params,
                        global_params=global_params,
                        device=device,
                        scale_factor=float(getattr(attack_config, 'scale_factor', sign_flip_scale)),
                    )
                elif attack_config.attack_type == 'random':
                    local_params = apply_random_attack(
                        model_params=local_params,
                        device=device,
                        attack_strength=float(getattr(attack_config, 'attack_strength', 1.0)),
                    )
                elif attack_config.attack_type == 'model_poison':
                    local_params = apply_model_poison_attack(
                        model_params=local_params,
                        global_params=global_params,
                        device=device,
                        attack_strength=float(getattr(attack_config, 'attack_strength', 1.0)),
                    )

            grad = {}
            for k in global_params:
                if isinstance(global_params[k], torch.Tensor) and global_params[k].dtype in [torch.int64, torch.int32, torch.long]:
                    continue
                grad[k] = (local_params[k] - global_params[k]).detach()

            client_gradients[client_id] = grad
            sample_counts[client_id] = total_samples
            
        # Aggregation
        # Start of Epoch Global Gradient for PSDL? 
        # For simplicity, pass None, aggregator calculates avg as baseline if needed
        
        # Prepare params for aggregators
        f = len(malicious_clients)
        
        # Aggregate
        aggregated_grad, _ = aggregator.aggregate(
            client_gradients, 
            sample_counts, 
            global_gradient=None,
            normalize=True, # PSDL and others benefit from normalization
            rule=method,
            f=f, # for Krum/Bulyan
            beta=0.1 # for Trimmed Mean
        )
        
        # Update Global Model
        new_params = {}
        for k in global_params:
            if k in aggregated_grad:
                new_params[k] = global_params[k] + aggregated_grad[k]
            else:
                 new_params[k] = global_params[k]
        global_params = new_params
        
    # Final Test
    global_model.load_state_dict(global_params)
    return test_model(global_model, test_loader, device)

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

def plot_results(results, ratios, dataset, attack):
    plt.figure(figsize=(10, 6))
    for method, accs in results.items():
        plt.plot(np.array(ratios)*100, accs, label=method, marker='o')
        
    plt.xlabel('% of malicious MCs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{dataset} under {attack} attack (Fixed Epochs)')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(blockchain_dir, 'results', f'exp_ratio_{dataset}_{attack}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--attack', type=str, default='sign_flip', choices=['label_flip', 'backdoor', 'sign_flip', 'random'])
    parser.add_argument('--epoch', type=int, default=30)
    args = parser.parse_args()
    
    set_seed(42)
    
    # Map friendly names to aggregator rules
    # 'Trimmed Mean' -> 'trimmed_mean'
    # 'Baseline' -> 'avg'
    # 'PSDL' -> 'psdl' ('Strict Governance' in earlier summary, but user requested 'PSDL')
    
    methods_map = {
        'Baseline': 'avg',
        'PSDL': 'psdl', 
        'Bulyan': 'bulyan',
        'Krum': 'krum',
        'Trimmed Mean': 'trimmed_mean'
    }
    
    # Invert for running but keep display names for results
    
    # We run using rule names
    rules = list(methods_map.values())
    
    results = {name: [] for name in methods_map.keys()}
    
    # Re-implement run logic here to map names correctly
    dataset_name = args.dataset
    attack_type = args.attack
    malicious_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    client_num = 10
    epochs = args.epoch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders, test_loader = get_data_loaders(dataset_name, client_num)
    
    for method_name, rule in methods_map.items():
        print(f"\nRunning Method: {method_name} (rule={rule})")
        accs = []
        for ratio in malicious_ratios:
             num_malicious = int(client_num * ratio)
             malicious_clients = list(range(num_malicious))
             
             attack_config = None
             if HAS_ATTACK:
                attack_config = AttackConfig()
                attack_config.enable_attack = True
                attack_config.malicious_clients = malicious_clients
                attack_config.attack_type = attack_type
                if attack_type == 'label_flip':
                    attack_config.label_flip_map = {i: (i + 1) % 10 for i in range(10)}
                elif attack_type == 'backdoor':
                    # Backdoor logic: target label 0
                    pattern = torch.zeros((1, 28, 28))
                    pattern[0, 24:, 24:] = 1.0  # 4x4 patch
                    attack_config.backdoor_pattern = pattern
                    attack_config.backdoor_target_label = 0
                elif attack_type == 'sign_flip':
                    attack_config.scale_factor = -5.0
                elif attack_type == 'random':
                    attack_config.attack_strength = 5.0
             
             final_acc = train_fl(client_loaders, test_loader, device, epochs, 
                                  client_num, malicious_clients, rule, attack_config)
             accs.append(final_acc)
             print(f"  Ratio {ratio}: {final_acc:.2f}%")
        
        results[method_name] = accs
        
    plot_results(results, malicious_ratios, dataset_name, attack_type)
