import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple

# Get project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_root = os.path.join(project_root, 'data')

class FederatedDataset(Dataset):
    """Simple Dataset Wrapper"""
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def iid_split(labels: np.ndarray, client_num: int) -> Dict[int, List[int]]:
    """IID Partitioning"""
    client_indices = {i: [] for i in range(client_num)}
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, client_num)
        for i in range(client_num):
            client_indices[i].extend(split_indices[i])
    return client_indices

def get_data_loaders(dataset_name='mnist', client_num=10, batch_size=32, split='iid'):
    """
    Get data loaders for clients and test set.
    Automatically prepares data if not exists.
    """
    dataset_name = dataset_name.lower()
    base_dir = os.path.join(data_root, dataset_name)
    
    # Check if prepared data exists
    clients_exist = True
    for i in range(client_num):
        if not os.path.exists(os.path.join(base_dir, f'client_{i+1}', 'data.npy')):
            clients_exist = False
            break
            
    if not clients_exist:
        print(f"[DataUtils] Preparing {dataset_name} data...")
        prepare_data(dataset_name, client_num, split)
        
    # Load loaders
    client_loaders = []
    for i in range(client_num):
        client_dir = os.path.join(base_dir, f'client_{i+1}')
        data = np.load(os.path.join(client_dir, 'data.npy'))
        labels = np.load(os.path.join(client_dir, 'label.npy'))
        dataset = FederatedDataset(data, labels)
        client_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        
    # Load test set
    test_dir = os.path.join(base_dir, 'test_dataset')
    test_data = np.load(os.path.join(test_dir, 'data.npy'))
    test_labels = np.load(os.path.join(test_dir, 'label.npy'))
    test_loader = DataLoader(FederatedDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader

def prepare_data(dataset_name, client_num, split_method='iid'):
    """Download and split data"""
    temp_dir = os.path.join(data_root, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    base_dir = os.path.join(data_root, dataset_name)
    
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root=temp_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.MNIST(root=temp_dir, train=False, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'fashionmnist':
        train_dataset = datasets.FashionMNIST(root=temp_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.FashionMNIST(root=temp_dir, train=False, download=True, transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    train_data = train_dataset.data.numpy()
    train_labels = train_dataset.targets.numpy()
    
    # Preprocess
    if dataset_name in ['mnist', 'fashionmnist']:
         # Normalize and add channel dim
         train_data = train_data.astype(np.float32) / 255.0
         if len(train_data.shape) == 3:
             train_data = np.expand_dims(train_data, axis=1)
             
         test_data = test_dataset.data.numpy().astype(np.float32) / 255.0
         if len(test_data.shape) == 3:
             test_data = np.expand_dims(test_data, axis=1)
         test_labels = test_dataset.targets.numpy()
         
    # Split
    if split_method == 'iid':
        client_indices = iid_split(train_labels, client_num)
    else:
        # Fallback to IID for now
        print(f"[DataUtils] Warning: Method {split_method} not implemented in utils, using IID.")
        client_indices = iid_split(train_labels, client_num)
        
    # Save Clients
    for i in range(client_num):
        indices = client_indices[i]
        c_data = train_data[indices]
        c_labels = train_labels[indices]
        
        path = os.path.join(base_dir, f'client_{i+1}')
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, 'data.npy'), c_data)
        np.save(os.path.join(path, 'label.npy'), c_labels)
        
    # Save Test
    test_path = os.path.join(base_dir, 'test_dataset')
    os.makedirs(test_path, exist_ok=True)
    np.save(os.path.join(test_path, 'data.npy'), test_data)
    np.save(os.path.join(test_path, 'label.npy'), test_labels)
    print(f"[DataUtils] Saved {dataset_name} data to {base_dir}")
