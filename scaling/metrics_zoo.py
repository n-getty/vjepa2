"""Candidate scaling-law y-axis metrics, computed on CACHED ruler features (pure numpy, no GPU).

Motivation (see scaling/FINDINGS.md): Metric B (1-R^2 of a ridge map run-encoder -> T*) has intrinsically
shallow per-budget bowls (values 0.78-0.97) so its curvature `a` is tiny and vertex localization is
noise-limited. The vertex-ordering test fails because per-budget vertex noise ~= the compute-driven
vertex drift. A more size-SENSITIVE, dimension-INVARIANT metric may deepen the bowls (bigger `a`) and
resolve the ordering WITHOUT any new GPU training.

Each metric takes cached X (run encoder, n x d_run) and Y (T*, n x d_tstar) on the SAME fixed ruler and
returns a scalar "error" (lower = closer to T* = better representation), so all are drop-in y-axes for
the same IsoFLOP parabola fit. We report ERROR (1 - similarity) for the bounded [0,1] similarities so
"down = better" matches Metric B and the parabola code (which fits a convex-up bowl in error).

Metrics
-------
- metric_b_ridge : the existing 1-R^2 ridge alignment (standardized X + intercept + CV lambda). Baseline.
- cka_linear     : 1 - linear CKA(X, Y). Dimension-invariant by construction (no fit, no lambda). Bounded.
- cka_rbf        : 1 - kernel CKA(X, Y) with median-heuristic RBF. Captures nonlinear structure R^2 misses.
- mutual_knn     : 1 - mutual-kNN alignment (Huh et al. "Platonic Representation"). Rank-based, affine-
                   invariant, high dynamic range; measures shared local neighborhood structure.
- procrustes     : orthogonal-Procrustes normalized residual (rotation-only alignment; tighter than free
                   ridge, less prone to "a big encoder can linearly reconstruct anything" saturation).

All operate per the SAME train/test split convention as Metric B where a split is meaningful; CKA and
mutual-kNN are fit-free so they use the full ruler (no split needed / no overfitting risk).
"""

import numpy as np

from scaling.eval_metric_b import align_error_cv, layernorm_np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _center(K):
    """Center a Gram matrix: HKH with H = I - 11^T/n."""
    n = K.shape[0]
    u = K.mean(axis=0, keepdims=True)
    m = K.mean()
    return K - u - u.T + m


def _subsample_rows(X, Y, max_n, seed):
    """CKA/kNN are O(n^2); cap rows for tractability + a fixed subsample for determinism."""
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    if max_n and n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)[:max_n]
        X, Y = X[idx], Y[idx]
    return X, Y


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def cka_linear(X, Y, standardize=True):
    """Linear CKA. Dimension-invariant, in [0,1]; 1 => identical up to orthogonal transform + scale.

    CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), on column-centered X,Y (== HSIC_linear
    normalized). Equivalent to centered-Gram-matrix agreement but computed in feature space (cheaper
    when d << n)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    if standardize:
        Xc = Xc / (X.std(axis=0, keepdims=True) + 1e-8)
        Yc = Yc / (Y.std(axis=0, keepdims=True) + 1e-8)
    # ||Y^T X||_F^2 = sum of squared cross-covariance entries
    cross = Xc.T @ Yc                     # (d_x, d_y)
    hsic_xy = (cross ** 2).sum()
    xx = Xc.T @ Xc
    yy = Yc.T @ Yc
    hsic_xx = (xx ** 2).sum()
    hsic_yy = (yy ** 2).sum()
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def cka_rbf(X, Y, max_n=1500, seed=0):
    """Kernel (RBF) CKA with the median-heuristic bandwidth. O(n^2) -> row-capped."""
    X, Y = _subsample_rows(X, Y, max_n, seed)

    def _rbf(Z):
        sq = np.sum(Z * Z, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
        d2 = np.maximum(d2, 0)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        return np.exp(-d2 / (med + 1e-12))

    Kx = _center(_rbf(X))
    Ky = _center(_rbf(Y))
    hsic_xy = (Kx * Ky).sum()
    hsic_xx = (Kx * Kx).sum()
    hsic_yy = (Ky * Ky).sum()
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def mutual_knn(X, Y, k=10, max_n=2000, seed=0):
    """Mutual-kNN alignment (Huh et al. 2024). For each point, take its k nearest neighbors in X-space
    and in Y-space; score = mean Jaccard-style overlap of the two neighbor sets. Rank/affine-invariant,
    high dynamic range. Returns similarity in [0,1]."""
    X, Y = _subsample_rows(X, Y, max_n, seed)
    n = len(X)
    if n <= k + 1:
        return 0.0

    def _knn(Z):
        sq = np.sum(Z * Z, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * Z @ Z.T
        np.fill_diagonal(d2, np.inf)
        # indices of k smallest per row
        return np.argpartition(d2, k, axis=1)[:, :k]

    nx = _knn(X)
    ny = _knn(Y)
    overlaps = np.empty(n)
    for i in range(n):
        a = set(nx[i].tolist())
        b = set(ny[i].tolist())
        overlaps[i] = len(a & b) / k
    return float(overlaps.mean())


def procrustes_error(X, Y, train_frac=0.7, seed=0):
    """Orthogonal-Procrustes alignment residual (1 - R^2-like), on standardized X and LayerNorm'd Y.

    Fit rotation+scale R (min ||X R - Y||) on train via SVD of X^T Y, report normalized residual on test.
    Restricting the map to orthogonal (vs free ridge) prevents high-dim encoders from trivially
    reconstructing Y, which is what saturates Metric B's right arm."""
    # standardize X to a common d via PCA-free path: pad/truncate is wrong; instead reduce both to min-d
    # by keeping top singular directions so the orthogonal map is square. Simpler + robust: whiten X, then
    # solve for a (d_x -> d_y) orthonormal-columns map via SVD (Procrustes handles rectangular).
    Yl = layernorm_np(Y)
    Xs = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
    n = min(len(Xs), len(Yl))
    Xs, Yl = Xs[:n], Yl[:n]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = perm[:ntr], perm[ntr:]
    # Procrustes: M = X^T Y ; U S Vt = svd(M) ; R = U Vt  (rectangular orthonormal map d_x->d_y)
    M = Xs[tr].T @ Yl[tr]
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt                              # (d_x, d_y), columns orthonormal
    # optimal scale (global) to match magnitudes
    pred_tr = Xs[tr] @ R
    scale = (pred_tr * Yl[tr]).sum() / ((pred_tr ** 2).sum() + 1e-12)
    pred = scale * (Xs[te] @ R)
    resid = ((Yl[te] - pred) ** 2).sum(axis=0)
    total = ((Yl[te] - Yl[te].mean(0)) ** 2).sum(axis=0)
    total = np.where(total <= 0, np.nan, total)
    return float(np.nanmean(resid / total))


def rankme(X, eps=1e-7):
    """RankMe (Garrido et al. 2023): smooth effective rank of the feature matrix — a REFERENCE-FREE,
    LABEL-FREE unsupervised proxy for representation quality. RankMe = exp(-sum p_k log p_k) where
    p_k = sigma_k / sum(sigma) are the L1-normalized singular values of X.

    CRITICAL for this study: unlike every T*-referenced metric (ridge/CKA/kNN/procrustes), RankMe has
    NO ceiling at the reference size — it measures the encoder's OWN feature-space dimensionality, so a
    1.9B encoder is not penalized for exceeding T*'s 1B. At fixed IsoFLOP budget it still trades off
    capacity (rank up with N) against undertraining (rank down when data-starved), so it can define a
    genuine vertex free of the saturation confound. Higher = better, so we return ERROR = -RankMe/d
    (normalized by feature dim, negated so down=better matches the parabola fitter)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # singular values of the (n x d) feature matrix
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / (s.sum() + eps)
    p = p[p > 0]
    entropy = -(p * np.log(p)).sum()
    rm = float(np.exp(entropy))
    d = X.shape[1]
    # normalize by dim so cross-encoder comparison isn't dominated by raw dim; error = 1 - RankMe/d
    return 1.0 - rm / d


def score_all(X, Y, seed=239):
    """Compute every candidate metric as an ERROR (down=better) for one cell's cached features."""
    out = {}
    out["rankme"] = rankme(X)
    # ridge baseline (train/test split inside)
    n = min(len(X), len(Y))
    Xn, Yn = X[:n], Y[:n]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(0.7 * n)
    tr, te = perm[:ntr], perm[ntr:]
    try:
        err, lam = align_error_cv(Xn[tr], Yn[tr], Xn[te], Yn[te], seed=seed)
        out["metric_b_ridge"] = err
    except Exception as e:
        out["metric_b_ridge"] = float("nan")
    out["cka_linear"] = 1.0 - cka_linear(Xn, Yn)
    out["cka_rbf"] = 1.0 - cka_rbf(Xn, Yn, seed=seed)
    out["mutual_knn"] = 1.0 - mutual_knn(Xn, Yn, seed=seed)
    try:
        out["procrustes"] = procrustes_error(Xn, Yn, seed=seed)
    except Exception:
        out["procrustes"] = float("nan")
    return out


if __name__ == "__main__":
    # quick self-test on synthetic data: a rotated+noised copy should score LOW error on all metrics,
    # random Y should score HIGH.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((800, 64))
    Q, _ = np.linalg.qr(rng.standard_normal((96, 64)))  # (96,64) orthonormal cols -> map 64->96 via Q
    Y_good = X @ Q.T + 0.05 * rng.standard_normal((800, 96))  # (800,64)@(64,96)
    Y_bad = rng.standard_normal((800, 96))
    print("aligned :", {k: round(v, 4) for k, v in score_all(X, Y_good).items()})
    print("random  :", {k: round(v, 4) for k, v in score_all(X, Y_bad).items()})
