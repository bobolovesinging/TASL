# -*- coding: utf-8 -*-
"""CIFAR-10 min_max + scaling 补充实验"""
import csv, os, sys, numpy as np
import torch, torch.nn as nn

SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,SCRIPT_DIR)
from exp2_cifar10 import (CifarCNN,set_seed,evaluate_full,
    state_sub,state_add,flatten_update,
    fedavg_aggregate,multi_krum_aggregate,trimmed_mean_aggregate,fltrust_aggregate,
    compute_tasl_trust_weights,
    train_honest,_get_lr_for_round,prepare_cifar10_loaders)

def train_min_max(global_state,loader,device,local_epochs=5,lr=0.1,z_max=2.0,
                   honest_flat_updates=None):
    if honest_flat_updates is None or len(honest_flat_updates)==0:
        return train_honest(global_state,loader,device,local_epochs,lr)
    honest_arr=np.stack(list(honest_flat_updates.values()),axis=0)
    mu=honest_arr.mean(axis=0);sigma=honest_arr.std(axis=0)+1e-12
    mal_flat=mu-z_max*sigma
    model=CifarCNN().to(device);model.load_state_dict(global_state)
    model.train();opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    cr=nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x,y in loader:x,y=x.to(device),y.to(device);opt.zero_grad();cr(model(x),y).backward();opt.step()
    local_state=model.state_dict()  # on device
    ms={};offset=0
    for k,v in local_state.items():
        if isinstance(v,torch.Tensor) and v.is_floating_point():
            n=v.numel();chunk=torch.from_numpy(mal_flat[offset:offset+n]).float().to(device).reshape(v.shape)
            ms[k]=global_state[k].detach()+chunk;offset+=n
        else:ms[k]=v
    return ms

def train_scaling(global_state,loader,device,local_epochs=5,lr=0.1,scaling_factor=100.0):
    model=CifarCNN().to(device);model.load_state_dict(global_state)
    model.train();opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=5e-4)
    cr=nn.CrossEntropyLoss()
    for _ in range(local_epochs):
        for x,y in loader:x,y=x.to(device),y.to(device);opt.zero_grad();cr(model(x),y).backward();opt.step()
    ls=model.state_dict()  # on device
    ss={}
    for k,v in ls.items():
        if isinstance(v,torch.Tensor) and v.is_floating_point():
            delta=v-global_state[k].detach();ss[k]=global_state[k].detach()+scaling_factor*delta
        else:ss[k]=v
    return ss

ALGOS=['fedavg','multi_krum','trimmed_mean','fltrust']
ATTACKS=['min_max','scaling']
N_CLIENTS=10;N_BYZ=4;ROUNDS=100;LOCAL_EP=5;LR=0.1;BATCH=64

def run_one(algo,attack,loaders,test_loader,device,seed=42):
    set_seed(seed)
    byz_set=set(range(N_BYZ));gm=CifarCNN().to(device);gs=gm.state_dict()
    acc_hist=[];asr_hist=[];ema_weights=None;cos_history=[];prev_anchor=None;temporal_anchor=None
    for r in range(1,ROUNDS+1):
        current_lr=_get_lr_for_round(LR,r,ROUNDS)
        local_updates={};flat_updates={}

        if attack=='min_max':
            for cid in range(N_CLIENTS):
                if cid not in byz_set:
                    ls=train_honest(gs,loaders[cid],device,LOCAL_EP,current_lr)
                    upd=state_sub(ls,gs);local_updates[cid]=upd;flat_updates[cid]=flatten_update(upd)
            honest_flat={cid:flat_updates[cid] for cid in range(N_CLIENTS) if cid not in byz_set}
            for cid in byz_set:
                ls=train_min_max(gs,loaders[cid],device,LOCAL_EP,current_lr,z_max=2.0,honest_flat_updates=honest_flat)
                upd=state_sub(ls,gs);local_updates[cid]=upd;flat_updates[cid]=flatten_update(upd)
        else:
            for cid in range(N_CLIENTS):
                if cid in byz_set and attack!='none':
                    ls=train_scaling(gs,loaders[cid],device,LOCAL_EP,current_lr,scaling_factor=100.0)
                else:
                    ls=train_honest(gs,loaders[cid],device,LOCAL_EP,current_lr)
                upd=state_sub(ls,gs);local_updates[cid]=upd;flat_updates[cid]=flatten_update(upd)

        if algo=='fedavg':agg_update=fedavg_aggregate(local_updates)
        elif algo=='multi_krum':agg_update=multi_krum_aggregate(flat_updates,local_updates,f=N_BYZ)
        elif algo=='trimmed_mean':agg_update=trimmed_mean_aggregate(local_updates,beta=0.2)
        elif algo=='fltrust':agg_update=fltrust_aggregate(flat_updates,local_updates,device)
        elif algo=='tasl':
            trust_weights,cos_sims,new_anchor=compute_tasl_trust_weights(
                flat_updates,trust_power=5.0,max_weight_ratio=1.5,min_cos_threshold=0.2,
                norm_penalty_strength=0.8,refine_anchor=True,
                ema_weights=ema_weights,ema_alpha=0.7,
                cos_history=cos_history,prev_anchor=prev_anchor,temporal_anchor=temporal_anchor)
            ema_weights=dict(trust_weights);cos_history.append(dict(cos_sims))
            prev_anchor=new_anchor;agg_update=fedavg_aggregate(local_updates,trust_weights)
            agg_flat=flatten_update(agg_update)
            if np.linalg.norm(agg_flat)>1e-12:temporal_anchor=agg_flat

        gs=state_add(gs,agg_update);gm.load_state_dict(gs)
        acc,asr=evaluate_full(gm,test_loader,device,attack=attack)
        acc_hist.append(acc);asr_hist.append(asr)
        if r%20==0 or r==1: print(f'  [{algo}/{attack}] R{r:03d} acc={acc:.2f}% asr={asr:.2f}%')

    best=float(max(acc_hist));avg=float(np.mean(acc_hist))
    best_asr=float(max(asr_hist));avg_asr=float(np.mean(asr_hist))
    print(f'  => {algo}/{attack}: best={best:.2f}% avg={avg:.2f}% ASR_best={best_asr:.2f}% ASR_avg={avg_asr:.2f}%')
    return {'best':best,'avg':avg,'best_asr':best_asr,'avg_asr':avg_asr,'history':acc_hist}

def main():
    device=torch.device('cuda')
    print(f'[CIFAR-10 Scale/MinMax] {N_CLIENTS}c/{N_BYZ}b/{ROUNDS}r')
    print(f'  Algos: {ALGOS}');print(f'  Attacks: {ATTACKS}')
    loaders,tl=prepare_cifar10_loaders(N_CLIENTS,BATCH)
    results={};total=len(ALGOS)*len(ATTACKS);done=0
    for atk in ATTACKS:
        for algo in ALGOS:
            done+=1
            print(f'\n[{done}/{total}] {algo} x {atk}')
            r=run_one(algo,atk,loaders,tl,device,seed=42+hash(atk+algo)%1000)
            results[f'{algo}_{atk}']=r
    print('\n'+'='*60);print('SUMMARY')
    print(f'{"Algo":<14} {"Attack":<12} {"Best":>8} {"Avg":>8} {"ASR":>8}')
    print('-'*50)
    for atk in ATTACKS:
        for algo in ALGOS:
            r=results[f'{algo}_{atk}']
            print(f'{algo:<14} {atk:<12} {r["best"]:>7.2f}% {r["avg"]:>7.2f}% {r["best_asr"]:>7.2f}%')
    od=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),'results')
    os.makedirs(od,exist_ok=True)
    cp=os.path.join(od,'exp2_cifar10_v5_scale.csv')
    with open(cp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['algorithm','attack','best_accuracy','avg_accuracy','best_asr','avg_asr'])
        w.writeheader()
        for atk in ATTACKS:
            for algo in ALGOS:
                r=results[f'{algo}_{atk}']
                w.writerow({'algorithm':algo,'attack':atk,'best_accuracy':r['best'],
                           'avg_accuracy':r['avg'],'best_asr':r['best_asr'],'avg_asr':r['avg_asr']})
    print(f'\nSaved: {cp}')

if __name__=='__main__':main()
