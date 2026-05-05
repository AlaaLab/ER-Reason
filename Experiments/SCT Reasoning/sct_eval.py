import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
from sklearn.metrics import cohen_kappa_score

# ── Data loading ──────────────────────────────────────────────────────────────
# sct_results.csv  — output of sct.py, contains all three conditions
#                    (baseline, single_oracle, full_oracle) with a 'condition' column.
#                    Must also have a 'model' column identifying which model was run.
#
# gt_clean.csv     — ground-truth physician annotations with columns:
#                    encounterkey, differential, dxupdate, dxtrajectory

# llm = pd.read_csv("sct_results.csv")
# gt  = pd.read_csv("gt_clean.csv")


# ── Typo fixes ────────────────────────────────────────────────────────────────
# Known annotation typos in the dataset — applied before any string matching.

TYPO_FIXES = {
    'aspergillosus':                 'aspergillosis',
    'arrythmia':                     'arrhythmia',
    'Arrythmia':                     'Arrhythmia',
    'electrolyte-induced arrythmia': 'electrolyte-induced arrhythmia',
    'intraparynchemal hemorrhage':   'intraparenchymal hemorrhage',
    'intracranial hemorrhage:':      'intracranial hemorrhage',
}

def apply_typo_fixes(df):
    df = df.copy()
    df['differential'] = df['differential'].apply(
        lambda x: TYPO_FIXES.get(str(x).strip(), str(x).strip())
    )
    return df


# ── Shared helpers ────────────────────────────────────────────────────────────
def parse_json_col(val):
    if pd.isna(val):
        return None
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return None


def ranked_list_from_gt(traj_str):
    traj = parse_json_col(traj_str)
    if not traj:
        return None
    return [dx for dx, _ in sorted(traj.items(), key=lambda x: x[1])]


def ranked_list_from_llm(ranked_str):
    ranked = parse_json_col(ranked_str)
    return ranked if isinstance(ranked, list) else None


def lists_to_rank_vectors(list1, list2):
    map1 = {dx.lower().strip(): i + 1 for i, dx in enumerate(list1)}
    map2 = {dx.lower().strip(): i + 1 for i, dx in enumerate(list2)}
    common = sorted(set(map1.keys()) & set(map2.keys()))
    if len(common) < 2:
        return None, None
    return [map1[dx] for dx in common], [map2[dx] for dx in common]


def extract_llm_update(dx_updates_json, differential):
    updates = parse_json_col(dx_updates_json)
    if not updates:
        return None
    target = differential.lower().strip()
    for item in updates:
        if item.get('diagnosis', '').lower().strip() == target:
            return item.get('update')
    return None


def get_rank(ranked_differential_json, differential):
    ranked = parse_json_col(ranked_differential_json)
    if not isinstance(ranked, list):
        return None
    target = differential.lower().strip()
    for i, dx in enumerate(ranked):
        if dx.lower().strip() == target:
            return i + 1
    return None


# ── Merge gt and llm ──────────────────────────────────────────────────────────
def build_eval_df(gt, llm):
    gt_work  = apply_typo_fixes(gt.copy())
    llm_work = apply_typo_fixes(llm.copy())

    gt_work['dxtrajectory'] = gt_work['dxtrajectory'].apply(lambda traj_str: (
        json.dumps({TYPO_FIXES.get(k, k): v for k, v in json.loads(traj_str).items()})
        if pd.notna(traj_str) else traj_str
    ))

    gt_work['_key_enc']   = gt_work['encounterkey'].str.strip()
    gt_work['_key_diff']  = gt_work['differential'].str.lower().str.strip()
    llm_work['_key_enc']  = llm_work['encounterkey'].str.strip()
    llm_work['_key_diff'] = llm_work['differential'].str.lower().str.strip()

    merged = gt_work.merge(
        llm_work[['_key_enc', '_key_diff', 'condition', 'timestep',
                  'dx_updates_json', 'ranked_differential',
                  'final_diagnosis', 'is_final_step']],
        on=['_key_enc', '_key_diff'],
        how='inner'
    )

    merged['llm_update'] = merged.apply(
        lambda r: extract_llm_update(r['dx_updates_json'], r['differential']), axis=1
    )
    merged['gt_ranked']  = merged['dxtrajectory'].apply(ranked_list_from_gt)
    merged['llm_ranked'] = merged['ranked_differential'].apply(ranked_list_from_llm)

    return merged


# ── Metric 1: DxUpdate ────────────────────────────────────────────────────────
def compute_dxupdate(merged, label=""):
    valid = merged.dropna(subset=['dxupdate', 'llm_update']).copy()
    valid['dxupdate']   = valid['dxupdate'].astype(int)
    valid['llm_update'] = valid['llm_update'].astype(int)
    valid = valid[
        valid['dxupdate'].between(-2, 2) &
        valid['llm_update'].between(-2, 2)
    ]

    print(f"\n[DxUpdate{' — ' + label if label else ''}] valid rows: {len(valid)} / {len(merged)}")

    per_timestep = []
    for t, grp in valid.groupby('timestep'):
        if len(grp) < 2:
            continue
        try:
            k = cohen_kappa_score(
                grp['dxupdate'], grp['llm_update'],
                weights='linear', labels=[-2, -1, 0, 1, 2]
            )
        except Exception as e:
            k = np.nan
        per_timestep.append({'timestep': t, 'kappa': k, 'n': len(grp)})
        print(f"  Timestep {t}: kappa={k:.4f}  (n={len(grp)})")

    ts_df  = pd.DataFrame(per_timestep)
    mean_k = ts_df['kappa'].mean()
    std_k  = ts_df['kappa'].std()

    valid['exact_match']       = valid['dxupdate'] == valid['llm_update']
    valid['correct_direction'] = np.sign(valid['dxupdate']) == np.sign(valid['llm_update'])

    print(f"  Mean per-timestep Kappa: {mean_k:.4f} +/- {std_k:.4f}")
    print(f"  Exact match:             {valid['exact_match'].mean():.3f}")
    print(f"  Correct direction:       {valid['correct_direction'].mean():.3f}")

    return mean_k


# ── Metric 2: DxTrajectory ────────────────────────────────────────────────────
def compute_dxtrajectory(merged, label=""):
    results = []
    for _, row in merged.iterrows():
        gt_ranked  = row['gt_ranked']
        llm_ranked = row['llm_ranked']
        if gt_ranked is None or llm_ranked is None:
            continue
        r1, r2 = lists_to_rank_vectors(gt_ranked, llm_ranked)
        if r1 is None or len(set(r1)) < 2:
            continue
        rho, _ = spearmanr(r1, r2)
        results.append({'timestep': row['timestep'], 'rho': rho})

    rho_df = pd.DataFrame(results)
    if rho_df.empty:
        print(f"\n[DxTrajectory{' — ' + label if label else ''}] no valid rows.")
        return None

    print(f"\n[DxTrajectory{' — ' + label if label else ''}] valid rows: {len(rho_df)} / {len(merged)}")
    print(f"  Mean Spearman rho: {rho_df['rho'].mean():.4f} +/- {rho_df['rho'].std():.4f}")

    return rho_df['rho'].mean()


# ── Metric 3: FinalDx ─────────────────────────────────────────────────────────
def compute_finaldx(merged, label=""):
    final_rows = merged[merged['is_final_step'] == True].copy()
    print(f"\n[FinalDx{' — ' + label if label else ''}] final step rows: {len(final_rows)}")

    rho_results = []
    for _, row in final_rows.iterrows():
        r1, r2 = lists_to_rank_vectors(
            row['gt_ranked'] or [], row['llm_ranked'] or []
        )
        if r1 and len(set(r1)) >= 2:
            rho, _ = spearmanr(r1, r2)
            rho_results.append(rho)

    def gt_top1(traj_str):
        ranked = ranked_list_from_gt(traj_str)
        return ranked[0].lower().strip() if ranked else None

    final_rows['gt_top1'] = final_rows['dxtrajectory'].apply(gt_top1)
    final_rows['llm_top1_fd'] = final_rows['final_diagnosis'].apply(
        lambda x: x.lower().strip() if pd.notna(x) else None
    )
    final_rows['llm_top1_rnk'] = final_rows['llm_ranked'].apply(
        lambda x: x[0].lower().strip() if isinstance(x, list) and x else None
    )
    final_rows['top3_hit'] = final_rows.apply(
        lambda r: float(r['gt_top1'] in [dx.lower().strip() for dx in (r['llm_ranked'] or [])[:3]])
        if r['gt_top1'] and isinstance(r['llm_ranked'], list) else np.nan, axis=1
    )

    top1_acc = (final_rows['gt_top1'] == final_rows['llm_top1_fd']).mean()
    top3_acc = final_rows['top3_hit'].mean()
    final_rho = np.mean(rho_results) if rho_results else None

    print(f"  Final ranking rho: {final_rho:.4f}" if final_rho else "  Final ranking rho: N/A")
    print(f"  Top-1 accuracy:    {top1_acc:.3f}")
    print(f"  Top-3 accuracy:    {top3_acc:.3f}")

    return {'final_rho': final_rho, 'top1_acc': top1_acc, 'top3_acc': top3_acc}


# ── Metric 4: Intra-case coherence (Table 4) ──────────────────────────────────
def compute_internal_coherence(llm_df, label=""):
    df = apply_typo_fixes(llm_df.copy())
    df = df.sort_values(['encounterkey', 'timestep']).reset_index(drop=True)

    rows = []
    for ek, enc_df in df.groupby('encounterkey'):
        enc_df  = enc_df.sort_values('timestep').reset_index(drop=True)
        all_dxs = None
        for _, row in enc_df.iterrows():
            parsed = parse_json_col(row.get('all_dxs'))
            if parsed:
                all_dxs = parsed
                break
        if all_dxs is None:
            continue

        for idx in range(1, len(enc_df)):
            curr_row = enc_df.iloc[idx]
            prev_row = enc_df.iloc[idx - 1]
            if curr_row['timestep'] - prev_row['timestep'] != 1:
                continue
            if pd.isna(prev_row.get('ranked_differential')) or pd.isna(curr_row.get('ranked_differential')):
                continue

            for dx in all_dxs:
                update     = extract_llm_update(curr_row.get('dx_updates_json'), dx.strip())
                if update is None or update == 0:
                    continue
                old_rank   = get_rank(prev_row.get('ranked_differential'), dx.strip())
                new_rank   = get_rank(curr_row.get('ranked_differential'), dx.strip())
                if old_rank is None or new_rank is None:
                    continue
                rank_change = old_rank - new_rank
                if rank_change == 0:
                    continue
                rows.append({
                    'encounterkey': ek,
                    'timestep':     curr_row['timestep'],
                    'differential': dx.strip(),
                    'update':       update,
                    'rank_change':  rank_change,
                    'coherent':     int(np.sign(update) == np.sign(rank_change)),
                })

    cdf = pd.DataFrame(rows)
    if cdf.empty:
        print(f"\n[Coherence{' — ' + label if label else ''}] no valid pairs.")
        return None, cdf

    rate = cdf['coherent'].mean()
    print(f"\n[Coherence{' — ' + label if label else ''}] "
          f"rate={rate:.1%}  N={len(cdf)}")
    return rate, cdf


# ── Figure 3: Top-1 accuracy by timestep ─────────────────────────────────────
def plot_top1_by_timestep(merged, output_prefix="top1_by_timestep"):
    merged = merged.copy()
    merged['gt_top1'] = merged['dxtrajectory'].apply(
        lambda x: (ranked_list_from_gt(x) or [None])[0]
    )
    merged['gt_top1'] = merged['gt_top1'].apply(
        lambda x: x.lower().strip() if x else None
    )
    merged['llm_top1'] = merged['llm_ranked'].apply(
        lambda x: x[0].lower().strip() if isinstance(x, list) and x else None
    )
    merged['top1_hit'] = (merged['gt_top1'] == merged['llm_top1']).astype(float)

    SOLID = 'solid'
    DASH  = (0, (6, 2))
    palette = {
        'Phi-4':               {'color': '#C0392B', 'ls': SOLID, 'marker': 'o'},
        'Claude Sonnet 4.5':   {'color': '#D4A843', 'ls': SOLID, 'marker': 'D'},
        'GPT-5.2':             {'color': '#7B5EA7', 'ls': SOLID, 'marker': 'h'},
        'DeepSeek-R1':         {'color': '#E67E22', 'ls': DASH,  'marker': 's'},
        'o4-mini':             {'color': '#4A9BAD', 'ls': DASH,  'marker': 'P'},
        'Claude 4.5 Thinking': {'color': '#7D9E6A', 'ls': DASH,  'marker': '^'},
        'Gemini 2.5 Flash':    {'color': '#5B7EC9', 'ls': DASH,  'marker': 'v'},
        'GPT-5.2 Thinking':    {'color': '#2C3E50', 'ls': DASH,  'marker': '*'},
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor='white')
    ax.set_facecolor('white')

    for model_name, p in palette.items():
        sub = merged[merged['model'] == model_name].groupby('timestep')['top1_hit'].mean().reset_index()
        if sub.empty:
            continue
        ax.plot(
            sub['timestep'], sub['top1_hit'] * 100,
            color=p['color'], linestyle=p['ls'], linewidth=1.8,
            marker=p['marker'],
            markersize=6.5 if p['marker'] != '*' else 10,
            markerfacecolor='white', markeredgecolor=p['color'],
            markeredgewidth=1.6, label=model_name
        )

    timesteps = sorted(merged['timestep'].unique())
    ns = [merged[merged['timestep'] == t]['encounterkey'].nunique() for t in timesteps]

    ax.set_xticks(timesteps)
    ax.set_xticklabels([f"T{t}\n(n={n})" for t, n in zip(timesteps, ns)],
                       fontsize=9, color='black')
    ax.set_xlabel('Timestep', fontsize=11, fontweight='bold', color='black')
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=11, fontweight='bold', color='black')
    ax.set_title('Top-1 Diagnostic Accuracy by Timestep', fontsize=12,
                 fontweight='bold', color='black')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.grid(axis='y', color='#EEEEEE', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=8.5, frameon=True,
              framealpha=1.0, edgecolor='#CCCCCC', facecolor='white')

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {output_prefix}.pdf + {output_prefix}.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # llm = pd.read_csv("sct_results.csv")   # output of sct.py
    # gt  = pd.read_csv("gt_clean.csv")      # physician ground-truth annotations

    summary = []

    for condition, llm_cond in llm.groupby('condition'):
        print(f"\n{'='*60}")
        print(f"CONDITION: {condition.upper()}")
        print(f"{'='*60}")

        merged = build_eval_df(gt, llm_cond)

        mean_k   = compute_dxupdate(merged, label=condition)
        mean_rho = compute_dxtrajectory(merged, label=condition)
        finaldx  = compute_finaldx(merged, label=condition)
        coh_rate, _ = compute_internal_coherence(llm_cond, label=condition)

        summary.append({
            'condition':   condition,
            'mean_kappa':  mean_k,
            'mean_rho':    mean_rho,
            'final_rho':   finaldx['final_rho'],
            'top1_acc':    finaldx['top1_acc'],
            'top3_acc':    finaldx['top3_acc'],
            'coherence':   coh_rate,
        })

    print(f"\n{'='*60}")
    print("SUMMARY ACROSS CONDITIONS")
    print(f"{'='*60}")
    print(pd.DataFrame(summary).to_string(index=False))

    # Figure 3 — baseline condition only (zero-shot)
    baseline_llm = llm[llm['condition'] == 'baseline']
    merged_baseline = build_eval_df(gt, baseline_llm)
    plot_top1_by_timestep(merged_baseline)