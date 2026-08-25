"""Step 4A: plug-in s(0.2) by denial label / lock status / direction, with
propagated fractions and the deniers-vs-admitters contrast, from the saved
(orig, neutral) fits. Usage: python step4_labels.py [--priors neutral,...]"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import arviz as az
from model import DEFAULT_GRID, load_counts, prepare_data
from three_arm import PAIRS, merge_records, shared_baseline_s_draws

A = Path("../artifacts")

def eti(x, lo=2.5, hi=97.5):
    return [float(np.mean(x)), float(np.percentile(x, lo)), float(np.percentile(x, hi))]

def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--priors", default="skeptical,neutral,informed")
    p.add_argument("--eps", type=float, default=0.2); p.add_argument("--t", type=float, default=0.2)
    args = p.parse_args(argv)
    step3 = load_counts(A / "step3_counts.jsonl"); neutral = load_counts(A / "step4_neutral_counts.jsonl")
    baseline = load_counts(A / "step4_baseline_counts.jsonl")
    merged, _ = merge_records(step3, neutral, baseline)
    cbc = {r["cell_id"]: (int(r["k"]), int(r["n"])) for r in baseline}
    arm_a, arm_b = PAIRS["r"]
    data = prepare_data(merged, grid=DEFAULT_GRID, arm_a=arm_a, arm_b=arm_b)
    g = int(np.argmin(np.abs(np.asarray(DEFAULT_GRID) - args.t)))
    st3 = {u["unit_id"]: u for u in json.load(open(A / "step3_summaries.json"))["neutral"]["summaries"] if u["is_rollout"]}
    lab, dr = {}, {}
    for l in open(A / "prefix_direction.jsonl"):
        r = json.loads(l); lab[r["source_id"]] = r["denial_label"]; dr[r["source_id"]] = r["direction"]
    out = {}
    for prior in args.priors.split(","):
        idata = az.from_netcdf(A / f"step4_fits/{prior}/step4_idata_r_{prior}.nc")
        s = shared_baseline_s_draws(idata, data, cbc, pair="r")[:, :, g]  # (draws, units)
        units = list(data.unit_ids)
        roll = [i for i, u in enumerate(units) if u in st3]
        groups = defaultdict(list)
        for i in roll:
            u = units[i]; late = st3[u]["dependent_after_t0"]
            groups[("all", "all")].append(i); groups[("lock", "late" if late else "early")].append(i)
            groups[("label", lab.get(u))].append(i); groups[("label_x_lock", f"{lab.get(u)}|{'late' if late else 'early'}")].append(i)
            groups[("dir", dr.get(u))].append(i); groups[("dir_x_lock", f"{dr.get(u)}|{'late' if late else 'early'}")].append(i)
        res = {}
        for k, idx in groups.items():
            x = s[:, idx]
            res["/".join(k)] = {"n": len(idx), "s_mean": eti(x.mean(axis=1)),
                                "frac_gt_eps": eti((x > args.eps).mean(axis=1)),
                                "frac_abs_lt_eps": eti((np.abs(x) < args.eps).mean(axis=1))}
        # bet-shaped = late-locked OR s(0.2) > eps, per draw
        late_mask = np.array([st3[units[i]]["dependent_after_t0"] for i in roll])
        sh = (s[:, roll] > args.eps) | late_mask[None, :]
        res["bet_shaped/all"] = {"n": len(roll), "frac": eti(sh.mean(axis=1))}
        for L in ("Admits", "Denies"):
            m = np.array([lab.get(units[i]) == L for i in roll])
            res[f"bet_shaped/{L}"] = {"n": int(m.sum()), "frac": eti(sh[:, m].mean(axis=1))}
        dA = [i for i in roll if lab.get(units[i]) == "Admits"]; dD = [i for i in roll if lab.get(units[i]) == "Denies"]
        res["contrast/Denies-Admits s_mean"] = eti(s[:, dD].mean(axis=1) - s[:, dA].mean(axis=1))
        res["contrast/Denies-Admits frac_gt_eps"] = eti((s[:, dD] > args.eps).mean(axis=1) - (s[:, dA] > args.eps).mean(axis=1))
        mA = np.array([lab.get(units[i]) == "Admits" for i in roll]); mD = np.array([lab.get(units[i]) == "Denies" for i in roll])
        res["contrast/Denies-Admits bet_shaped"] = eti(sh[:, mD].mean(axis=1) - sh[:, mA].mean(axis=1))
        out[prior] = res
        print(f"\n== prior {prior}  (s = p_neutral - p_base at t={args.t}, eps={args.eps}; mean [2.5%, 97.5%])")
        for k, v in res.items():
            if "contrast" in k: print(f"  {k:40s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")
            elif "bet_shaped" in k: print(f"  {k:40s} n={v['n']:3d}  frac {v['frac'][0]:.3f} [{v['frac'][1]:.3f}, {v['frac'][2]:.3f}]")
            else: print(f"  {k:40s} n={v['n']:3d}  s {v['s_mean'][0]:+.3f}  P>eps {v['frac_gt_eps'][0]:.3f} [{v['frac_gt_eps'][1]:.3f},{v['frac_gt_eps'][2]:.3f}]  |s|<eps {v['frac_abs_lt_eps'][0]:.3f} [{v['frac_abs_lt_eps'][1]:.3f},{v['frac_abs_lt_eps'][2]:.3f}]")
    json.dump(out, open(A / "step4_labels.json", "w"), indent=1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
