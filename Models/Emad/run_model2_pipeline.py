import os
import sys
import time
import json
import numpy as np
import pandas as pd
import cv2

print("="*75)
print("RUNNING MODEL 2: HYBRID MIEW-ID + SIFT OPEN-SET RECOGNITION PIPELINE")
print("="*75)

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(BASE_DIR, "DATA", "reid_student_data_main")
PROCESSED_DIR = os.path.join(BASE_DIR, "Processed_Data")
OUTPUT_SUBMISSION = os.path.join(PROCESSED_DIR, "submission_model2_hybrid_openset.csv")

# 1. Load Data
master_df = pd.read_csv(os.path.join(PROCESSED_DIR, "master_dataset.csv"))
features_raw = np.load(os.path.join(PROCESSED_DIR, "miewid_features.npy"), mmap_mode="r")
norm_features = features_raw / np.maximum(np.linalg.norm(features_raw, axis=1, keepdims=True), 1e-12)
sift_data = np.load(os.path.join(PROCESSED_DIR, "sift_features.npz"))

train_df = master_df[master_df["split"] == "train"].reset_index(drop=True)
test_df = master_df[master_df["split"] == "test"].reset_index(drop=True)

print(f"Master Dataset: {len(master_df)} rows")
print(f"Train set: {len(train_df)} rows, Test set: {len(test_df)} rows")

# 2. SIFT Matcher
bf = cv2.BFMatcher(cv2.NORM_L2)

def verify_sift_pair(q_img_id, g_img_id, ratio_thresh=0.75, ransac_thresh=12.0):
    q_desc = sift_data[f"{q_img_id}_desc"]
    g_desc = sift_data[f"{g_img_id}_desc"]
    if len(q_desc) < 4 or len(g_desc) < 4:
        return 0, 0, 0.0
    matches = bf.knnMatch(q_desc, g_desc, k=2)
    good = []
    q_pts, g_pts = [], []
    q_xy = sift_data[f"{q_img_id}_xy"]
    g_xy = sift_data[f"{g_img_id}_xy"]
    for m in matches:
        if len(m) == 2 and m[0].distance < ratio_thresh * m[1].distance:
            good.append(m[0])
            q_pts.append(q_xy[m[0].queryIdx])
            g_pts.append(g_xy[m[0].trainIdx])
    if len(good) < 4:
        return len(good), 0, 0.0
    q_pts = np.float32(q_pts)
    g_pts = np.float32(g_pts)
    M, inliers = cv2.estimateAffinePartial2D(q_pts, g_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    n_inliers = int(np.sum(inliers)) if inliers is not None else 0
    inlier_ratio = (n_inliers / len(good)) if len(good) > 0 else 0.0
    return len(good), n_inliers, inlier_ratio

# 3. Deterministic Open-set Split Construction
def make_openset_split(df, unknown_prop=0.25, seed=42):
    rng = np.random.default_rng(seed)
    gallery_idx, known_val_idx, unknown_val_idx = [], [], []
    for sp in sorted(df["species"].unique()):
        sp_df = df[df["species"] == sp]
        # Sort identities alphabetically to guarantee identical starting permutation
        unique_ids = np.sort(sp_df["identity"].unique())
        rng.shuffle(unique_ids)
        
        n_unknown = int(round(len(unique_ids) * unknown_prop))
        unknown_ids = set(unique_ids[:n_unknown])
        known_ids = set(unique_ids[n_unknown:])
        
        id_groups = sp_df.groupby("identity").groups
        for u_id in sorted(unknown_ids):
            unknown_val_idx.extend(id_groups[u_id].tolist())
            
        for k_id in sorted(known_ids):
            indices = np.sort(id_groups[k_id].to_numpy())
            rng.shuffle(indices)
            if len(indices) >= 2:
                n_val = max(1, int(round(len(indices) * 0.2)))
                n_val = min(n_val, len(indices) - 1)
                known_val_idx.extend(indices[:n_val])
                gallery_idx.extend(indices[n_val:])
            else:
                gallery_idx.extend(indices)
                
    return (
        df.loc[gallery_idx].reset_index(drop=True),
        df.loc[known_val_idx].reset_index(drop=True),
        df.loc[unknown_val_idx].reset_index(drop=True)
    )

# Candidate Retrieval: K_pool = 15 candidates retrieved for every query
K_POOL = 15
K_SIFT = 5

def retrieve_miewid_candidates(query_df, gallery_df, top_k_ids=K_POOL):
    q_vecs = norm_features[query_df["miewid_index"].values]
    g_vecs = norm_features[gallery_df["miewid_index"].values]
    gal_species = gallery_df["species"].to_numpy()
    gal_identities = gallery_df["identity"].to_numpy()
    gal_ids = gallery_df["id"].to_numpy()
    
    sim_matrix = q_vecs @ g_vecs.T
    candidates_per_query = []
    for i in range(len(query_df)):
        sp = query_df.iloc[i]["species"]
        mask = gal_species == sp
        sims = sim_matrix[i, mask]
        c_idents = gal_identities[mask]
        c_img_ids = gal_ids[mask]
        order = np.argsort(-sims)
        seen = set()
        cands = []
        for idx in order:
            ident = c_idents[idx]
            if ident not in seen:
                seen.add(ident)
                cands.append((ident, float(sims[idx]), c_img_ids[idx]))
                if len(seen) >= top_k_ids:
                    break
        candidates_per_query.append(cands)
    return candidates_per_query

# Unified Ranking Function ensuring exact mathematical equivalence:
def rank_candidates_unified(candidates_pool, inliers_top_k, beta, threshold):
    """
    Ranks the full candidate pool of size K_POOL along with 'new_individual'.
    For candidates within rank < K_SIFT, SIFT score is added: score = miew_sim + beta * norm_sift.
    For candidates with rank >= K_SIFT, SIFT is not evaluated: score = miew_sim.
    When beta == 0.0, every candidate's score is strictly miew_sim, identical to Config A.
    """
    scored = []
    for rank_idx, (ident, m_sim, _) in enumerate(candidates_pool):
        if rank_idx < len(inliers_top_k):
            inl = inliers_top_k[rank_idx]
            norm_sift = min(inl / 20.0, 1.0)
            score = m_sim + beta * norm_sift
        else:
            score = m_sim
        scored.append((ident, score))
    scored.append(("new_individual", threshold))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scored[:5]]

# Metric Evaluation
def evaluate_predictions_array(preds, gts):
    scores = []
    for p, gt in zip(preds, gts):
        sc = 0.0
        for r, ident in enumerate(p[:5], 1):
            if ident == gt:
                sc = 1.0 / r
                break
        scores.append(sc)
    return np.array(scores)

# Build Validation Splits across seeds 42, 100, 2026
seeds = [42, 100, 2026]
splits_data = []

print(f"\nBuilding deterministic validation splits and retrieving candidates (K_POOL={K_POOL}, K_SIFT={K_SIFT})...")
total_query_count = 0
for seed in seeds:
    gal, kq, uq = make_openset_split(train_df, unknown_prop=0.25, seed=seed)
    val = pd.concat([kq, uq], ignore_index=True)
    gts = kq['identity'].tolist() + ['new_individual'] * len(uq)
    sp_list = val['species'].tolist()
    
    # Assert leak-free
    assert len(set(gal['identity']).intersection(set(uq['identity']))) == 0, f"Identity leakage in seed {seed}!"
    assert set(kq['identity']).issubset(set(gal['identity'])), f"Known queries not in gallery in seed {seed}!"
    
    kq_c = retrieve_miewid_candidates(kq, gal, K_POOL)
    uq_c = retrieve_miewid_candidates(uq, gal, K_POOL)
    val_cands = kq_c + uq_c
    
    # SIFT verification for top K_SIFT candidates
    val_inls = []
    for i in range(len(val)):
        qid = val.iloc[i]['id']
        val_inls.append([verify_sift_pair(qid, gid)[1] for _, _, gid in val_cands[i][:K_SIFT]])
        
    splits_data.append({
        'seed': seed, 'gal': gal, 'kq': kq, 'uq': uq, 'val': val, 'gts': gts,
        'sp_list': sp_list, 'val_cands': val_cands, 'val_inls': val_inls
    })
    total_query_count += len(val)
    print(f"  Seed {seed}: Gallery={len(gal)} ({gal['identity'].nunique()} IDs), KnownVal={len(kq)}, UnknownVal={len(uq)}, TotalVal={len(val)}")

print(f"Total validation queries evaluated across all 3 seeds: {total_query_count}")

# 4. Step 1: Validation-Based Threshold Selection Procedure (Optimizing Config C)
print("\n" + "="*75)
print("STEP 1: VALIDATION-BASED THRESHOLD SELECTION PROCEDURE (OPTIMIZING CONFIG C)")
print("="*75)
print("Explicit Note: This search tunes species thresholds specifically for Configuration C (beta=0.05).")
print("Selection Constraint: Known-Query MAP@5 >= 0.800 and False Positive Rate <= 25.0%")

threshold_candidates = [(0.51, 0.53), (0.51, 0.55), (0.51, 0.58),
                        (0.53, 0.53), (0.53, 0.55), (0.53, 0.57), (0.53, 0.58),
                        (0.55, 0.53), (0.55, 0.55), (0.55, 0.58)]

grid_evals = []
for th_h, th_l in threshold_candidates:
    ov_list, kn_list, un_list, fpr_list = [], [], [], []
    for d in splits_data:
        n_k = len(d['kq'])
        preds = []
        for i in range(len(d['val'])):
            th = th_h if d['sp_list'][i] == 'hyena' else th_l
            p = rank_candidates_unified(d['val_cands'][i], d['val_inls'][i], beta=0.05, threshold=th)
            preds.append(p)
        sc = evaluate_predictions_array(preds, d['gts'])
        ov_list.append(sc.mean())
        kn_list.append(sc[:n_k].mean())
        un_list.append(sc[n_k:].mean())
        fpr = np.mean([p[0] == 'new_individual' for p in preds[:n_k]]) * 100
        fpr_list.append(fpr)
    grid_evals.append({
        'th_hyena': th_h, 'th_leopard': th_l,
        'overall_map': np.mean(ov_list),
        'known_map': np.mean(kn_list),
        'unknown_map': np.mean(un_list),
        'known_fpr_pct': np.mean(fpr_list)
    })

df_grid = pd.DataFrame(grid_evals)
print(df_grid.to_string(index=False))

# Apply selection criterion: highest overall_map subject to known_map >= 0.800 and known_fpr_pct <= 25.0%
valid_df = df_grid[(df_grid['known_map'] >= 0.800) & (df_grid['known_fpr_pct'] <= 25.0)].sort_values('overall_map', ascending=False)
SELECTED_TH_HYENA = valid_df.iloc[0]['th_hyena']
SELECTED_TH_LEOPARD = valid_df.iloc[0]['th_leopard']

print(f"\nOptimal selected thresholds under constraint: tau_hyena = {SELECTED_TH_HYENA:.2f}, tau_leopard = {SELECTED_TH_LEOPARD:.2f}")
print(f"Config C Constrained Validation Metrics: Known MAP={valid_df.iloc[0]['known_map']:.5f}, Unknown MAP={valid_df.iloc[0]['unknown_map']:.5f}, Overall MAP={valid_df.iloc[0]['overall_map']:.5f}, FPR={valid_df.iloc[0]['known_fpr_pct']:.2f}%")

# 5. Step 2: Final Multi-Split Evaluation
print("\n" + "="*75)
print(f"STEP 2: MULTI-SPLIT EVALUATION (tau_h={SELECTED_TH_HYENA:.2f}, tau_l={SELECTED_TH_LEOPARD:.2f}, K_POOL={K_POOL})")
print("="*75)

results_M1, results_A, results_C, results_dis, detection_stats = [], [], [], [], []

for d in splits_data:
    seed = d['seed']
    val = d['val']
    gts = d['gts']
    sp_list = d['sp_list']
    val_cands = d['val_cands']
    val_inls = d['val_inls']
    n_k, n_u = len(d['kq']), len(d['uq'])
    
    k_mask = np.array([True]*n_k + [False]*n_u)
    u_mask = ~k_mask
    h_mask = np.array([sp == 'hyena' for sp in sp_list])
    l_mask = np.array([sp == 'leopard' for sp in sp_list])
    
    # 1. Model 1 (Closed-Set Baseline): Top 5 known identities from the pool
    preds_M1 = [[c[0] for c in val_cands[i][:5]] for i in range(len(val))]
    sc_M1 = evaluate_predictions_array(preds_M1, gts)
    results_M1.append({
        'seed': seed, 'n_known': n_k, 'n_unknown': n_u,
        'overall': sc_M1.mean(), 'known': sc_M1[k_mask].mean(), 'unknown': sc_M1[u_mask].mean()
    })
    
    # 2. Config A (MIEW-ID + Open-Set): Full pool ranked with threshold (no SIFT)
    preds_A = []
    for i in range(len(val)):
        th = SELECTED_TH_HYENA if sp_list[i] == 'hyena' else SELECTED_TH_LEOPARD
        c = [(ident, sim) for ident, sim, _ in val_cands[i]] + [('new_individual', th)]
        c.sort(key=lambda x: x[1], reverse=True)
        preds_A.append([x[0] for x in c[:5]])
    sc_A = evaluate_predictions_array(preds_A, gts)
    results_A.append({
        'seed': seed, 'n_known': n_k, 'n_unknown': n_u,
        'overall': sc_A.mean(), 'known': sc_A[k_mask].mean(), 'unknown': sc_A[u_mask].mean(),
        'hyena': sc_A[h_mask].mean(), 'leopard': sc_A[l_mask].mean()
    })
    
    # 3. Config C (Hybrid MIEW-ID + SIFT + Open-Set): Full pool ranked with SIFT (beta=0.05) on top K_SIFT
    preds_C = []
    for i in range(len(val)):
        th = SELECTED_TH_HYENA if sp_list[i] == 'hyena' else SELECTED_TH_LEOPARD
        p = rank_candidates_unified(val_cands[i], val_inls[i], beta=0.05, threshold=th)
        preds_C.append(p)
    sc_C = evaluate_predictions_array(preds_C, gts)
    results_C.append({
        'seed': seed, 'n_known': n_k, 'n_unknown': n_u,
        'overall': sc_C.mean(), 'known': sc_C[k_mask].mean(), 'unknown': sc_C[u_mask].mean(),
        'hyena': sc_C[h_mask].mean(), 'leopard': sc_C[l_mask].mean()
    })
    
    # 4. SIFT-Disabled Ablation (Config C with beta=0.0 across full pool)
    preds_dis = []
    for i in range(len(val)):
        th = SELECTED_TH_HYENA if sp_list[i] == 'hyena' else SELECTED_TH_LEOPARD
        p = rank_candidates_unified(val_cands[i], val_inls[i], beta=0.0, threshold=th)
        preds_dis.append(p)
    sc_dis = evaluate_predictions_array(preds_dis, gts)
    results_dis.append({
        'seed': seed, 'n_known': n_k, 'n_unknown': n_u,
        'overall': sc_dis.mean(), 'known': sc_dis[k_mask].mean(), 'unknown': sc_dis[u_mask].mean()
    })
    
    # Assert exact equivalence between Config A and beta=0.0 across all predictions
    mismatches = sum([p1 != p2 for p1, p2 in zip(preds_A, preds_dis)])
    assert mismatches == 0, f"Seed {seed}: Found {mismatches} mismatches between Config A and beta=0.0!"
    
    # 5. Open-Set Detection Rates
    rank1_new_ind = np.array([p[0] == 'new_individual' for p in preds_C])
    detection_stats.append({
        'seed': seed,
        'known_rank1_pct': rank1_new_ind[k_mask].mean() * 100,
        'unknown_rank1_pct': rank1_new_ind[u_mask].mean() * 100,
        'hyena_known_rank1_pct': rank1_new_ind[k_mask & h_mask].mean() * 100,
        'hyena_unknown_rank1_pct': rank1_new_ind[u_mask & h_mask].mean() * 100,
        'leo_known_rank1_pct': rank1_new_ind[k_mask & l_mask].mean() * 100,
        'leo_unknown_rank1_pct': rank1_new_ind[u_mask & l_mask].mean() * 100
    })

print(f"\nAll {total_query_count} queries verified: exactly 0 prediction differences between Config A and Config C (beta=0.0)!")

print("\n--- Model 1 Results ---")
df_M1 = pd.DataFrame(results_M1)
print(df_M1.to_string(index=False))

print("\n--- Config A Results ---")
df_A = pd.DataFrame(results_A)
print(df_A.to_string(index=False))

print("\n--- Config C Results ---")
df_C = pd.DataFrame(results_C)
print(df_C.to_string(index=False))

print("\n--- Config C (SIFT Disabled) Results ---")
df_dis = pd.DataFrame(results_dis)
print(df_dis.to_string(index=False))

print("\n--- Detection Stats (% new_individual at Rank 1) ---")
df_det = pd.DataFrame(detection_stats)
print(df_det.to_string(index=False))

# 6. Step 3: Test Inference with Exactly the Same Unified Candidate Pool Policy
print("\n" + "="*75)
print(f"STEP 3: TEST INFERENCE (K_POOL={K_POOL}, K_SIFT={K_SIFT}, tau_h={SELECTED_TH_HYENA:.2f}, tau_l={SELECTED_TH_LEOPARD:.2f}, beta=0.05)")
print("="*75)

t0 = time.time()
test_cands = retrieve_miewid_candidates(test_df, train_df, top_k_ids=K_POOL)
print(f"Retrieved top-{K_POOL} MIEW candidates for {len(test_df)} test queries in {time.time()-t0:.2f}s")

t_sift = time.time()
test_sift_inls = []
for i in range(len(test_df)):
    qid = test_df.iloc[i]["id"]
    inls = [verify_sift_pair(qid, gid)[1] for _, _, gid in test_cands[i][:K_SIFT]]
    test_sift_inls.append(inls)
print(f"Test SIFT verification completed in {time.time()-t_sift:.2f}s")

final_rows = []
for i in range(len(test_df)):
    qid = test_df.iloc[i]["id"]
    sp = test_df.iloc[i]["species"]
    th = SELECTED_TH_HYENA if sp == "hyena" else SELECTED_TH_LEOPARD
    
    top_5 = rank_candidates_unified(test_cands[i], test_sift_inls[i], beta=0.05, threshold=th)
    final_rows.append({
        "id": qid,
        "predictions": " ".join(top_5)
    })

sub_df = pd.DataFrame(final_rows)
sub_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Saved submission to {OUTPUT_SUBMISSION}")

# Validate CSV
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
assert (sub_df["id"] == sample_sub["id"]).all(), "ID sequence mismatch!"
assert sub_df["predictions"].notna().all(), "Missing predictions found!"
token_counts = sub_df["predictions"].apply(lambda x: len(x.split()))
assert (token_counts == 5).all(), "Every prediction row must contain exactly 5 tokens!"

# Cross-species checks
known_hyenas = set(train_df[train_df["species"] == "hyena"]["identity"].unique())
known_leos = set(train_df[train_df["species"] == "leopard"]["identity"].unique())

for _, row in sub_df.iterrows():
    q_sp = test_df.loc[test_df["id"] == row["id"], "species"].values[0]
    tokens = row["predictions"].split()
    assert len(tokens) == len(set(tokens)), f"Duplicate tokens in row {row['id']}"
    assert tokens.count("new_individual") <= 1, f"Multiple new_individual tokens in row {row['id']}"
    for token in tokens:
        if token == "new_individual":
            continue
        if q_sp == "hyena":
            assert token in known_hyenas, f"Illegal leopard token {token} for hyena test image {row['id']}"
        else:
            assert token in known_leos, f"Illegal hyena token {token} for leopard test image {row['id']}"

print("ALL CSV VALIDATION CHECKS PASSED PERFECTLY!")
print(f"Total test queries: {len(sub_df)}")
print(f"Queries with 'new_individual' at Rank 1: {(sub_df['predictions'].apply(lambda x: x.split()[0] == 'new_individual')).sum()}")
print(f"Queries with 'new_individual' in Top 5: {(sub_df['predictions'].apply(lambda x: 'new_individual' in x.split())).sum()}")

# Cache summary for reporting
with open(os.path.join(PROCESSED_DIR, "validation_metrics_summary.json"), "w") as f:
    json.dump({
        'total_query_count': total_query_count,
        'threshold_selection': grid_evals,
        'selected_thresholds': {'th_hyena': SELECTED_TH_HYENA, 'th_leopard': SELECTED_TH_LEOPARD},
        'results_M1': results_M1,
        'results_A': results_A,
        'results_C': results_C,
        'results_dis': results_dis,
        'detection_stats': detection_stats
    }, f, indent=2)

print("Pipeline execution complete.")
