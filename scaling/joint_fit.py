import csv, numpy as np
from collections import defaultdict
# FAST joint IsoFLOP fit via PROFILE over alpha. For a FIXED alpha, define shifted coordinate
# u = logN - alpha*(logC - logCbar). The bowl err = c0 + curv*(u - u0)^2 is LINEAR in [1,u,u^2] ->
# closed-form OLS. Sweep alpha on a grid, pick min-SSE alpha; bootstrap over cells for a CI. ms not min.
rows=list(csv.DictReader(open('scaling/metrics_c256_scores.csv')))
mets=['metric_b_ridge','cka_linear','cka_rbf','mutual_knn','procrustes']

def prep(metric):
    pts=[]
    for r in rows:
        e=r[metric]
        if e in ('',None): continue
        pts.append((np.log10(float(r['n_params'])), np.log10(float(r['budget_flops'])), float(e)))
    P=np.array(pts); return P

def sse_at_alpha(P, alpha):
    logN,logC,err=P[:,0],P[:,1],P[:,2]
    u=logN-alpha*(logC-logC.mean())
    A=np.vstack([np.ones_like(u),u,u*u]).T
    coef,res,*_=np.linalg.lstsq(A,err,rcond=None)
    pred=A@coef
    curv=coef[2]
    return np.sum((err-pred)**2), curv

def best_alpha(P, grid):
    sses=[(sse_at_alpha(P,a)[0],a,sse_at_alpha(P,a)[1]) for a in grid]
    sses=[s for s in sses if s[2]>1e-6]  # require convex bowl (curv>0)
    if not sses: return np.nan
    return min(sses)[1]

grid=np.linspace(-0.3,1.0,131)
print(f"{'metric':12s} {'alpha_hat':>9} {'boot 5-95 CI':>20} {'excl0':>6} {'convex?':>7}")
for m in mets:
    P=prep(m)
    ah=best_alpha(P,grid)
    # bootstrap over cells
    rng=np.random.default_rng(0); n=len(P); a=[]
    for _ in range(2000):
        idx=rng.integers(0,n,n)
        v=best_alpha(P[idx],grid)
        if np.isfinite(v): a.append(v)
    a=np.array(a)
    if len(a)>100:
        lo,hi=np.percentile(a,[5,95]); ci=f"[{lo:+.2f},{hi:+.2f}]"; excl="YES" if (lo>0 or hi<0) else "no"
    else: ci="unstable"; excl="?"
    conv = "yes" if np.isfinite(ah) else "NO"
    print(f"{m:12s} {ah:+9.3f} {ci:>20} {excl:>6} {conv:>7}")
print()
print("interpretation: alpha_hat = joint compute-scaling exponent of N_opt. CI excluding 0 = resolvable.")
print("LLM reference alpha ~ 0.5 (Chinchilla). alpha<0.25 => N_opt grows slowly with compute for V-JEPA.")
