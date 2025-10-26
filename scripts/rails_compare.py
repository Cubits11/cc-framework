# scripts/rails_compare.py
# Compute J@FPR-window for two rails + independence baselines + destructive flags + diagnostics.

import argparse, numpy as np, pandas as pd
from sklearn.metrics import roc_curve, auc
import statsmodels.formula.api as smf
from pathlib import Path

def binarize(scores, thr): return (scores >= thr).astype(int)

def confusion(y, yhat):
    tp = ((y==1)&(yhat==1)).sum(); fn = ((y==1)&(yhat==0)).sum()
    tn = ((y==0)&(yhat==0)).sum(); fp = ((y==0)&(yhat==1)).sum()
    return dict(tp=int(tp), fn=int(fn), tn=int(tn), fp=int(fp))

def rates(m):
    tp,fn,tn,fp = m["tp"],m["fn"],m["tn"],m["fp"]
    tpr = tp/(tp+fn) if (tp+fn) else 0.0
    fpr = fp/(fp+tn) if (fp+tn) else 0.0
    tnr = 1-fpr
    return tpr,fpr,tnr

def J(tpr,tnr): return tpr+tnr-1.0

def sweep_thr_for_fpr(y, scores, fpr_min, fpr_max, grid=None):
    if grid is None: grid = np.linspace(0,1,2001)  # fine grid
    best = dict(thr=None, j=-1, tpr=0, fpr=1, tnr=0)
    for thr in grid:
        yb = binarize(scores, thr); tpr,fpr,tnr = rates(confusion(y,yb))
        if fpr_min <= fpr <= fpr_max:
            jj = J(tpr,tnr)
            if jj > best["j"]: best = dict(thr=float(thr), j=float(jj), tpr=float(tpr), fpr=float(fpr), tnr=float(tnr))
    return best

def compose_any(a,b): return ((a==1)|(b==1)).astype(int)
def compose_both(a,b): return ((a==1)&(b==1)).astype(int)

def independence_j(tpr_a,fpr_a,tpr_b,fpr_b, mode):
    if mode=="any":
        tpr = 1-(1-tpr_a)*(1-tpr_b)
        fpr = 1-(1-fpr_a)*(1-fpr_b)
    elif mode=="both":
        tpr = tpr_a*tpr_b
        fpr = fpr_a*fpr_b
    else:
        raise ValueError("mode must be any|both")
    return J(tpr, 1-fpr)

def mutual_information(a_block, b_block):
    # simple 2x2 MI (no external deps)
    import math
    p11 = np.mean((a_block==1)&(b_block==1))
    p10 = np.mean((a_block==1)&(b_block==0))
    p01 = np.mean((a_block==0)&(b_block==1))
    p00 = np.mean((a_block==0)&(b_block==0))
    P = [p11,p10,p01,p00]; px1 = p11+p10; px0 = p01+p00; py1 = p11+p01; py0 = p10+p00
    def s(p,q): 
        return 0 if p==0 or q==0 else p*math.log(p/q,2)
    return sum([
        s(p11, px1*py1), s(p10, px1*py0),
        s(p01, px0*py1), s(p00, px0*py0)
    ])

def overlap_ratio(a_block,b_block):
    inter = int(((a_block==1)&(b_block==1)).sum())
    union = int(((a_block==1)|(b_block==1)).sum())
    return inter/union if uni
