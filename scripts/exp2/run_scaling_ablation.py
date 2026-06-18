"""One-off: run scaling attack on Fashion-MNIST ablation modes."""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from exp2_fashionmnist_ablation import run_ablation_single, ABLATION_MODES, SAVE_DIR
from exp2_fashionmnist import prepare_data_loaders

seed = 42
n_clients, n_byz, rounds = 10, 4, 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
client_loaders, test_loader = prepare_data_loaders(n_clients, 128, 0.3, 'non_iid', seed)

results = []
for mode in ABLATION_MODES:
    print(f'=== Scaling / {mode} ===')
    r = run_ablation_single('scaling', mode, client_loaders, test_loader,
                            n_clients, n_byz, rounds, device, 2, 0.02, seed)
    results.append(r)
    print(f'  Best={r["best_acc"]:.2f} Avg={r["avg_acc"]:.2f} ASR={r["asr"]:.2f}')

csv_path = os.path.join(SAVE_DIR, 'exp2_fashionmnist_ablation.csv')
with open(csv_path, 'a', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['attack','ablation','best_acc','avg_acc','last_acc','asr'])
    for r in results:
        w.writerow({'attack':'scaling','ablation':r['ablation'],
                    'best_acc':f'{r["best_acc"]:.2f}','avg_acc':f'{r["avg_acc"]:.2f}',
                    'last_acc':f'{r.get("last_acc",0):.2f}','asr':f'{r["asr"]:.2f}'})
print(f'Saved to {csv_path}')
