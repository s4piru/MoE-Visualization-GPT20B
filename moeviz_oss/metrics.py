import numpy as np
import pandas as pd

def _build_sparse_vec(indices, probs, size):
    v = np.zeros(size, dtype=np.float64)
    v[indices] = probs
    s = v.sum()
    if s > 0:
        v /= s
    return v

def _jsd(p, q):
    m = 0.5 * (p + q)
    pz = np.where(p > 0, p, 1e-12)
    qz = np.where(q > 0, q, 1e-12)
    mz = np.where(m > 0, m, 1e-12)
    kl_pm = np.sum(pz * (np.log(pz) - np.log(mz)))
    kl_qm = np.sum(qz * (np.log(qz) - np.log(mz)))
    return 0.5 * (kl_pm + kl_qm)

def layerwise_jsd_from_topk(moeA, moeB, gen_steps, startA, startB):
    names = sorted(set(moeA.keys()).intersection(set(moeB.keys())))
    mats = {}
    for name in names:
        Pa = moeA[name]["topk_probs"]
        Ia = moeA[name]["topk_indices"]
        Pb = moeB[name]["topk_probs"]
        Ib = moeB[name]["topk_indices"]
        Sa = Pa.shape[1]
        Sb = Pb.shape[1]
        ea = Pa.shape[2]
        eb = Pb.shape[2]
        T = gen_steps
        M = np.zeros((T,))
        for t in range(T):
            ia = startA + t
            ib = startB + t
            if ia >= Sa or ib >= Sb:
                continue
            idxa = Ia[0, ia]
            proba = Pa[0, ia]
            idxb = Ib[0, ib]
            probb = Pb[0, ib]
            sup = np.union1d(idxa, idxb)
            pa = _build_sparse_vec(idxa, proba, int(np.max(sup) + 1))
            pb = _build_sparse_vec(idxb, probb, int(np.max(sup) + 1))
            d = _jsd(pa, pb)
            M[t] = d
        mats[name] = M
    return mats

def summarize_expert_shifts(moeA, moeB, gen_steps, startA, startB, topn=10):
    rows = []
    names = sorted(set(moeA.keys()).intersection(set(moeB.keys())))
    for name in names:
        Pa = moeA[name]["topk_probs"]
        Ia = moeA[name]["topk_indices"]
        Pb = moeB[name]["topk_probs"]
        Ib = moeB[name]["topk_indices"]
        Sa = Pa.shape[1]
        Sb = Pb.shape[1]
        base_mass = {}
        var_mass = {}
        for t in range(gen_steps):
            ia = startA + t
            ib = startB + t
            if ia >= Sa or ib >= Sb:
                continue
            idxa = Ia[0, ia].tolist()
            proba = Pa[0, ia].tolist()
            idxb = Ib[0, ib].tolist()
            probb = Pb[0, ib].tolist()
            dA = {int(e): float(p) for e, p in zip(idxa, proba)}
            dB = {int(e): float(p) for e, p in zip(idxb, probb)}
            keys = set(dA.keys()) | set(dB.keys())
            for e in keys:
                base_mass[e] = base_mass.get(e, 0.0) + dA.get(e, 0.0)
                var_mass[e] = var_mass.get(e, 0.0) + dB.get(e, 0.0)
        all_keys = set(base_mass.keys()) | set(var_mass.keys())
        for e in all_keys:
            bm = float(base_mass.get(e, 0.0))
            vm = float(var_mass.get(e, 0.0))
            rows.append((name, int(e), bm, vm, vm - bm, abs(vm - bm)))
    df = pd.DataFrame(rows, columns=["layer", "expert", "base_mass", "var_mass", "delta", "abs_delta"])
    if df.empty:
        return df
    df = df.sort_values(["layer", "abs_delta"], ascending=[True, False])
    df = df.groupby("layer", as_index=False).head(topn)
    df = df.sort_values(["layer", "abs_delta"], ascending=[True, False]).reset_index(drop=True)
    return df