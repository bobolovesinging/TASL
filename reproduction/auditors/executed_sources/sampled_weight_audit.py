"""V_P sampled trust-weight admissibility audit.

Bounds use only this auditor's bound updates and trusted parent history.
V_S's supplied coefficients are the object being checked, never a reference.
"""
import time
import numpy as np
import torch

def flatten_update(update):
    arrays=[value.detach().cpu().numpy().reshape(-1).astype(np.float64)
            for value in update.values() if isinstance(value,torch.Tensor) and value.is_floating_point()]
    return np.concatenate(arrays) if arrays else np.array([],dtype=np.float64)


def _normalized_intervals(lower,upper,factor):
    low_n=lower*factor;high_n=upper*factor
    low=np.zeros(len(lower));high=np.zeros(len(lower))
    for i in range(len(lower)):
        if factor[i]<=0:continue
        other_high=sum(float(high_n[j]) for j in range(len(lower)) if j!=i)
        other_low=sum(float(low_n[j]) for j in range(len(lower)) if j!=i)
        denominator=float(low_n[i])+other_high
        low[i]=float(low_n[i])/denominator if denominator>0 else 0.
        denominator=float(high_n[i])+other_low
        high[i]=float(high_n[i])/denominator if denominator>0 else (1. if high_n[i]>0 else 0.)
    return low,high


def build_admissibility_bounds(local_updates,prior_trust_state,*,ema_alpha=.7,screen_tau=.6):
    start=time.perf_counter_ns();timings={}
    flat={c:flatten_update(u) for c,u in local_updates.items()}
    timings['input_flattening']=(time.perf_counter_ns()-start)/1e6
    ids=list(flat);n=len(ids)
    if not n:return dict(valid=False,intervals={},timings_ms=timings)
    step=time.perf_counter_ns()
    if any(not np.isfinite(v).all() for v in flat.values()):
        return dict(valid=False,intervals={},timings_ms=timings)
    norms=np.array([np.linalg.norm(flat[c]) for c in ids],dtype=np.float64)
    if not np.isfinite(norms).all():return dict(valid=False,intervals={},timings_ms=timings)
    eligible=norms<=2.5*float(np.median(norms))
    timings['norms_and_mask']=(time.perf_counter_ns()-step)/1e6
    step=time.perf_counter_ns()
    ema,_,previous,temporal,_,_=prior_trust_state
    reference=temporal if temporal is not None else previous
    direction=np.ones(n)
    if reference is not None and np.linalg.norm(reference)>=1e-12:
        reference_norm=float(np.linalg.norm(reference))+1e-12
        for i,c in enumerate(ids):
            cosine=float(np.dot(flat[c],reference)/((norms[i]+1e-12)*reference_norm))
            direction[i]=0. if cosine<=0 else (1. if cosine>=screen_tau else cosine/screen_tau)
    timings['historical_direction']=(time.perf_counter_ns()-step)/1e6
    step=time.perf_counter_ns()
    pad=64.*(n+1)*np.finfo(np.float64).eps
    lower=np.zeros(n);upper=np.ones(n)
    if ema is not None:
        history=np.array([ema.get(c,1./n) for c in ids],dtype=np.float64)
        if not np.isfinite(history).all() or (history<0).any():
            return dict(valid=False,intervals={},timings_ms=timings)
        denominator=ema_alpha+(1.-ema_alpha)*float(np.sum(history))
        if denominator>1e-12:
            lower=np.maximum(0.,(1.-ema_alpha)*history/denominator-pad)
            upper=np.minimum(1.,(ema_alpha+(1.-ema_alpha)*history)/denominator+pad)
    mask=eligible.astype(np.float64)
    fused=mask*direction
    lower_total=float(np.sum(lower*fused));upper_total=float(np.sum(upper*fused))
    fused_possible=upper_total>1e-12-pad
    fallback_possible=lower_total<=1e-12+pad
    candidates=[]
    if fused_possible:candidates.append(_normalized_intervals(lower,upper,fused))
    if fallback_possible:
        if float(np.sum(upper*mask))<=1e-12:
            candidates.append((np.zeros(n),np.zeros(n)))
        else:
            bounds=_normalized_intervals(lower,upper,mask)
            if float(np.sum(lower*mask))<=1e-12+pad:bounds=(np.zeros(n),bounds[1])
            candidates.append(bounds)
    if not candidates:candidates.append((np.zeros(n),np.zeros(n)))
    lows=np.min(np.stack([v[0] for v in candidates]),axis=0)
    highs=np.max(np.stack([v[1] for v in candidates]),axis=0)
    intervals={c:(max(0.,float(lows[i])-pad),min(1.,float(highs[i])+pad))
               if eligible[i] else (0.,0.) for i,c in enumerate(ids)}
    timings['bound_arithmetic']=(time.perf_counter_ns()-step)/1e6
    timings['total']=(time.perf_counter_ns()-start)/1e6
    return dict(valid=True,intervals=intervals,guaranteed_fused=not fallback_possible,
                eligible={c:bool(eligible[i]) for i,c in enumerate(ids)},
                direction={c:float(direction[i]) for i,c in enumerate(ids)},
                timings_ms=timings,reference_mode='admissibility_bounds')


def verify_sampled_bounds(published_weights,bounds,num_clients,seed,*,sample_ratio=.3,tolerance=1e-6):
    count=max(1,int(num_clients*sample_ratio))
    sampled=set(np.random.RandomState(seed).choice(num_clients,size=count,replace=False).tolist())
    values=list(published_weights.values())
    if not bounds['valid'] or any(not np.isfinite(v) or v<0 for v in values):
        return True,sampled
    if abs(sum(values)-1.)>max(1,num_clients)*tolerance:return True,sampled
    for cid in sampled:
        low,high=bounds['intervals'].get(cid,(0.,0.))
        value=published_weights.get(cid,0.)
        if value<low-tolerance or value>high+tolerance:return True,sampled
    return False,sampled


def lightweight_weight_crosscheck(published_weights,local_updates,prior_trust_state,
                                 num_clients,seed,*,sample_ratio=.3,tolerance=1e-6,
                                 ema_alpha=.7,screen_tau=.6):
    bounds=build_admissibility_bounds(local_updates,prior_trust_state,ema_alpha=ema_alpha,screen_tau=screen_tau)
    bad,sampled=verify_sampled_bounds(published_weights,bounds,num_clients,seed,
                                     sample_ratio=sample_ratio,tolerance=tolerance)
    return bad,sampled,bounds
