# -*- coding: utf-8 -*-
"""Exp5: Economic Scalability + Sensitivity Analysis (Revised)"""
import csv, math, os
from dataclasses import dataclass
from typing import Dict, List
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class GasAssumptions:
    base_tx_gas=21000; ecrecover_gas=3000; multisig_overhead_gas=15000
    sstore_new_slot_gas=20000; calldata_nonzero_gas=16; event_log_gas=2000
    cid_store_gas=20000; audit_snapshot_gas=20000; final_aggregation_snapshot_gas=30000

MODEL_SIZES = {"MNIST": int(1.6*1024*1024), "ST-GCN": int(12.5*1024*1024)}
SCALING = [10,20,30,40,50]; ROUNDS=300

def words(b): return int(math.ceil(b/32))
def gas2eth(g,gp): return g*gp*1e-9
def gas2usd(g,gp,ep): return gas2eth(g,gp)*ep

def est_a(n,ms,a):
    w=words(ms)
    return int(n*w*a.sstore_new_slot_gas + n*(a.base_tx_gas+a.calldata_nonzero_gas*ms) + w*max(1,n-1)*120)

def est_b(n,a):
    return int(n*(a.base_tx_gas+a.cid_store_gas+a.ecrecover_gas+a.event_log_gas))

def est_c(mn,n,a):
    ac=1.0 if mn=="MNIST" else 1.15
    tri=3*(a.base_tx_gas+int(a.audit_snapshot_gas*ac)+a.multisig_overhead_gas)
    fc=a.base_tx_gas+int(a.final_aggregation_snapshot_gas*ac)+a.multisig_overhead_gas
    tpc=800 if mn=="MNIST" else 950
    return int(tri+fc+n*tpc)

def build_rows(ns,rounds,gp,ep,a):
    rows=[]
    for mn,ms in MODEL_SIZES.items():
        for n in ns:
            sa,sb,sc=est_a(n,ms,a),est_b(n,a),est_c(mn,n,a)
            rv_a=100*(1-sc/sa) if sa>0 else 0
            rv_b=100*(1-sc/sb) if sb>0 else 0
            rows.append({"model":mn,"n_clients":float(n),"model_size_bytes":float(ms),
                "sa_per_round":float(sa),"sb_per_round":float(sb),"sc_per_round":float(sc),
                "sa_300r":float(rounds*sa),"sb_300r":float(rounds*sb),"sc_300r":float(rounds*sc),
                "sa_300r_usd":gas2usd(rounds*sa,gp,ep),"sb_300r_usd":gas2usd(rounds*sb,gp,ep),
                "sc_300r_usd":gas2usd(rounds*sc,gp,ep),
                "red_vs_a":rv_a,"red_vs_b":rv_b})
    return rows

def plot_gas(out,rows):
    clr={"A":"#1A4595","B":"#F2921D","C":"#008F7A"}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for idx,mn in enumerate(["MNIST","ST-GCN"]):
        sub=[r for r in rows if r["model"]==mn]
        ns=[int(r["n_clients"]) for r in sub]
        ax=axes[idx]
        for s,k,m in [("A","sa_per_round","s"),("B","sb_per_round","o"),("C","sc_per_round","^")]:
            ax.plot(ns,[r[k] for r in sub],marker=m,lw=2.2,color=clr[s],markersize=6,
                    label=["Monolithic","IPFS","TASL (Ours)"][ord(s)-65])
        ax.set_yscale("log");ax.set_xlabel("Clients");ax.set_ylabel("Gas/round (log)")
        ax.set_title(mn);ax.grid(ls="--",alpha=0.2);ax.legend(fontsize=8)
    fig.suptitle("Exp5: Gas Cost Scaling",fontweight='bold')
    plt.tight_layout();os.makedirs(os.path.dirname(out),exist_ok=True)
    plt.savefig(out,dpi=200,bbox_inches='tight');plt.close()

def plot_sensitivity(out,rows,a,results_dir):
    gas_scenes={"10 Gwei":10,"20 Gwei":20,"60 Gwei":60,"100 Gwei":100}
    eth_scenes={"$2000":2000,"$3000":3000,"$4000":4000}
    print("\n  --- Gas Price Sensitivity (ST-GCN, n=50) ---")
    for l,gp in gas_scenes.items():
        r=build_rows([50],300,gp,3000,a)
        r=[x for x in r if x["model"]=="ST-GCN"][0]
        print(f"  {l:<12}: TASL=${r['sc_300r_usd']:>8,.0f}  Mono=${r['sa_300r_usd']:>12,.0f}")
    print("\n  --- ETH Price Sensitivity (ST-GCN, n=50) ---")
    for l,ep in eth_scenes.items():
        r=build_rows([50],300,20,ep,a)
        r=[x for x in r if x["model"]=="ST-GCN"][0]
        print(f"  {l:<12}: TASL=${r['sc_300r_usd']:>8,.0f}  Mono=${r['sa_300r_usd']:>12,.0f}")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4.5))
    x=np.arange(4);w=0.35
    mg,tg=[],[]
    for gp in gas_scenes.values():
        r=build_rows([50],300,gp,3000,a);r=[x for x in r if x["model"]=="ST-GCN"][0]
        mg.append(r["sa_300r_usd"]);tg.append(r["sc_300r_usd"])
    ax1.bar(x-w/2,mg,w,color="#1A4595",alpha=0.3,label="Monolithic")
    ax1.bar(x+w/2,tg,w,color="#008F7A",label="TASL")
    ax1.set_yscale("log");ax1.set_xticks(x);ax1.set_xticklabels(list(gas_scenes.keys()),fontsize=8)
    ax1.set_ylabel("300r Cost (USD,log)");ax1.set_title("Gas Price Sensitivity");ax1.legend()
    me,te=[],[]
    for ep in eth_scenes.values():
        r=build_rows([50],300,20,ep,a);r=[x for x in r if x["model"]=="ST-GCN"][0]
        me.append(r["sa_300r_usd"]);te.append(r["sc_300r_usd"])
    ax2.bar(x[:3]-w/2,me,w,color="#1A4595",alpha=0.3,label="Monolithic")
    ax2.bar(x[:3]+w/2,te,w,color="#008F7A",label="TASL")
    ax2.set_yscale("log");ax2.set_xticks(x[:3]);ax2.set_xticklabels(list(eth_scenes.keys()),fontsize=8)
    ax2.set_ylabel("300r Cost (USD,log)");ax2.set_title("ETH Price Sensitivity");ax2.legend()
    fig.suptitle("Exp5: Sensitivity Analysis (ST-GCN, n=50)",fontweight='bold')
    plt.tight_layout();os.makedirs(os.path.dirname(out),exist_ok=True)
    plt.savefig(out,dpi=200,bbox_inches='tight');plt.close()
    print(f"  Saved: {out}")

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--rounds",type=int,default=ROUNDS)
    p.add_argument("--gas",type=float,default=20)
    p.add_argument("--eth",type=float,default=3000)
    args=p.parse_args()
    res=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),"results")
    os.makedirs(res,exist_ok=True)
    a=GasAssumptions()
    rows=build_rows(SCALING,args.rounds,args.gas,args.eth,a)
    print("[Exp5] Economic Scalability + Sensitivity (Revised)")
    for r in rows:
        print(f"  {r['model']:<6} n={int(r['n_clients']):>2} | "
              f"A={r['sa_per_round']:>14,.0f} B={r['sb_per_round']:>8,} C={r['sc_per_round']:>8,} | "
              f"vsA={r['red_vs_a']:>7.2f}% vsB={r['red_vs_b']:>6.1f}%")
    print("\n  --- 300-Round USD ---")
    for r in rows:
        print(f"  {r['model']:<6} n={int(r['n_clients']):>2} | "
              f"A=${r['sa_300r_usd']:>12,.0f} B=${r['sb_300r_usd']:>8,.0f} C=${r['sc_300r_usd']:>8,.0f}")
    plot_gas(os.path.join(res,"exp5_gas_scaling.png"),rows)
    with open(os.path.join(res,"exp5_gas_data.csv"),"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader()
        for r in rows:w.writerow(r)
    plot_sensitivity(os.path.join(res,"exp5_sensitivity.png"),rows,a,res)
    st=[x for x in rows if x["model"]=="ST-GCN" and int(x["n_clients"])==50][0]
    print(f"\n  Key: ST-GCN n=50: Mono=${st['sa_300r_usd']:,.0f} IPFS=${st['sb_300r_usd']:,.0f} TASL=${st['sc_300r_usd']:,.0f}")

if __name__=="__main__":
    main()
