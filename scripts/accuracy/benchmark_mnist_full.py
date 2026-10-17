import os
import csv
import numpy as np
import torch

from exp_accuracy_vs_epoch import (
    set_seed,
    get_client_loaders_iid,
    get_test_loader,
    train_fl_epoch,
)


def build_attack_config(malicious_clients):
    try:
        from attack_module import AttackConfig
    except ImportError:
        return None

    cfg = AttackConfig()
    cfg.enable_attack = True
    cfg.malicious_clients = list(malicious_clients)
    cfg.attack_type = "label_flip"
    # Required mapping: 3->8, 1->7
    cfg.label_flip_map = {3: 8, 1: 7}
    return cfg


def pad_to_len(values, n, fill=0.0):
    vals = list(values)
    if len(vals) < n:
        vals.extend([fill] * (n - len(vals)))
    return vals[:n]


def run_group(name, rule, client_loaders, test_loader, device, epochs, client_num, malicious_clients, enable_storage_pruning):
    print(f"\n===== Running {name} =====")
    attack_cfg = build_attack_config(malicious_clients) if malicious_clients else None

    details = train_fl_epoch(
        client_loaders=client_loaders,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        client_num=client_num,
        malicious_clients=malicious_clients,
        method=rule,
        attack_config=attack_cfg,
        tau=0.0,
        M=5,
        alpha=1.0,
        beta=0.5,
        return_details=True,
        enable_storage_pruning=enable_storage_pruning,
    )

    return details


def main():
    set_seed(100)

    epochs = 50
    client_num = 10
    malicious_clients = [0, 1, 2]  # 30%

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    client_loaders = get_client_loaders_iid(client_num=client_num, batch_size=128)
    test_loader = get_test_loader(batch_size=128)

    # A/B/C keep linear growth reference (disable pruning)
    A = run_group("Group A (Clean Baseline)", "avg", client_loaders, test_loader, device, epochs, client_num, [], False)
    B = run_group("Group B (No Defense - Attacked)", "avg", client_loaders, test_loader, device, epochs, client_num, malicious_clients, False)
    C = run_group("Group C (Krum Defense)", "krum", client_loaders, test_loader, device, epochs, client_num, malicious_clients, False)
    D = run_group("Group D (Our PBFL)", "our", client_loaders, test_loader, device, epochs, client_num, malicious_clients, True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "accuracy")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results_mnist_v1.csv")

    acc_a = pad_to_len(A.get("accuracies", []), epochs, 0.0)
    acc_b = pad_to_len(B.get("accuracies", []), epochs, 0.0)
    acc_c = pad_to_len(C.get("accuracies", []), epochs, 0.0)
    acc_d = pad_to_len(D.get("accuracies", []), epochs, 0.0)
    ct_d = pad_to_len(D.get("ct_sizes", []), epochs, 0)
    cd_d = pad_to_len(D.get("cd_sizes", []), epochs, 0)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Round", "Acc_A", "Acc_B", "Acc_C", "Acc_D", "CT_Size", "CD_Size"])
        for i in range(epochs):
            pruned = "Yes" if i > 0 and ct_d[i] <= ct_d[i - 1] else "No"
            print(f"Round {i+1}: PBFL Acc: {acc_d[i]:.2f}%, CT pruned: [{pruned}]")
            writer.writerow([i + 1, acc_a[i], acc_b[i], acc_c[i], acc_d[i], int(ct_d[i]), int(cd_d[i])])

    final_a = acc_a[-1]
    final_d = acc_d[-1]
    ratio = (final_d / final_a) if final_a > 0 else 0.0

    v1_removed = sum(1 for rec in D.get("v2_invalid_counts", []) if rec > 0)

    # compare with linear reference from A (same round count)
    ct_linear_final = A.get("ct_sizes", [0])[-1] if A.get("ct_sizes") else 0
    ct_pbfl_final = ct_d[-1]
    savings = 0.0
    if ct_linear_final > 0:
        savings = (1.0 - (ct_pbfl_final / float(ct_linear_final))) * 100.0

    print("\n===== Benchmark Summary =====")
    print(f"CSV: {out_csv}")
    print(f"Final Acc A: {final_a:.2f}%")
    print(f"Final Acc D: {final_d:.2f}%")
    print(f"D/A ratio: {ratio*100:.2f}%")
    print(f"Rounds with invalid signatures detected by V2: {v1_removed}")
    print(f"Storage metadata savings vs linear chain: {savings:.2f}%")


if __name__ == "__main__":
    main()
