"""Minimal ablation runner: runs TASL with specified ablation mode."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
ROUNDS = 50

# Patch the ablation mode into compute_tasl_trust_weights
import exp2_mnist_v3 as mn
orig_fn = mn.compute_tasl_trust_weights

def ablated_fn(*args, **kwargs):
    if MODE == 'nodata':
        kwargs['data_trust'] = None
    elif MODE == 'nograd':
        # Return equal weights based only on data trust
        kwargs['force_equal_grad'] = True
    elif MODE == 'no_temporal':
        kwargs['temporal_anchor'] = None
    elif MODE == 'no_normgate':
        kwargs['norm_penalty_strength'] = 0.0
    elif MODE == 'cos3':
        kwargs['trust_power'] = 3.0
    return orig_fn(*args, **kwargs)

# Handle nograd specially in compute_tasl_trust_weights
import numpy as np
orig_compute = mn.compute_tasl_trust_weights.__code__

# Use a cleaner approach: wrap the training call
mn.compute_tasl_trust_weights = ablated_fn

# Run
mn.main()
