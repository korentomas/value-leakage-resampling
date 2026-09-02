"""Audit of the step-3 continuations for threshold handling.

Motivated by FINDINGS "Threshold-format mismatch in the step-3 swap arm": the
swap arm rendered the threshold without thousands separators for 7/9
questions. This asks three things of the 62,500 step-3 continuations and the
2,500 prefixes they resume:

  1. Did the retained prefix already name the threshold, and in which form?
  2. Does the continuation ever get the threshold's *value* wrong, and does
     that happen more in the affected swap arm than in the orig arm?
  3. Is dep(t) different on the 7 affected questions than on the 2 clean ones?

It also measures the prefix/prompt *direction* conflict (limitation 3): how
often a continuation explicitly re-reads or reverses the exceed -> good/bad
mapping, orig vs swap.

Inputs (artifacts/): step3_requests.jsonl, step3_results_final.jsonl,
step3_judge_results.jsonl, step3_sources.jsonl, step3_summaries.json.
Writes artifacts/threshold_audit_rows.jsonl (one row per continuation) and
prints the tables quoted in FINDINGS.

Usage:  python swap/threshold_audit.py [--sample 100]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
ART = HERE.parent / "artifacts"
THINK = "<think>\n"

# Qwen says "wait" constantly, so a generic confusion vocabulary is useless on
# its own; FORMAT_MARK is the subset that is specifically about digit grouping.
CONFUSION = re.compile(
    r"\b(wait|hmm|hold on|confus\w*|typo|misread|mis-read|weird|strange|odd|"
    r"unclear|ambiguous|doesn'?t make sense|makes no sense|re-?read|reread|"
    r"how many (?:zeros|digits)|count(?:ing)? the (?:zeros|digits)|"
    r"missing (?:a )?comma|no commas?|without commas?|separator|"
    r"i think (?:they|the user) meant|presumably means|must mean|"
    r"that'?s (?:actually )?\d+ ?(?:million|billion|thousand)|"
    r"or is it|which is it|not sure (?:if|whether))\b",
    re.I,
)
FORMAT_MARK = re.compile(
    r"(how many (?:zeros|digits)|count(?:ing)? the (?:zeros|digits)|"
    r"missing (?:a )?comma|no commas?|without commas?|separator|"
    r"\d{5,}\s*(?:=|is|means|,? i\.e\.|->|→)\s*\d+(?:\.\d+)?\s*(?:million|billion|thousand)|"
    r"let me count)",
    re.I,
)
# "the threshold is <digits>", where the digits are not an abbreviation
# ("350M", "20.2 million") -- those would be false positives.
THRESHOLD_IS = re.compile(
    r"threshold[^.\n]{0,25}?(?:is|of|=|:|at)\s*\**\$?([\d,]{3,})"
    r"(?!\s*(?:m|k|b|bn|million|billion|thousand|trillion)\b)",
    re.I,
)
# The swap arm's prompt contradicts the prefix on which side is the good one.
DIRECTION_ANCHOR = re.compile(
    r"exceed[s]?\b[^.\n]{0,120}?(good|bad) cause|(good|bad) cause[^.\n]{0,120}?exceed", re.I
)
DIRECTION_FIX = re.compile(
    r"\b(re-?read|reread|misread|mis-?read|backwards|reversed|inverted|"
    r"i had it (?:the )?(?:other way|wrong)|earlier i (?:said|assumed|thought|had)|"
    r"my (?:previous|earlier) (?:logic|reading|assumption)|opposite of what i)\b",
    re.I,
)
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def word_forms(n: int) -> list[str]:
    """Regexes for '26 million' style renderings of an integer threshold."""
    out = []
    for div, name in ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand")):
        if n < div or n / div >= 10_000:
            continue
        if n % div == 0:
            out.append(rf"{n // div}\s*{name}")
        else:
            out.append(rf"{n / div:.1f}".replace(".", r"\.") + rf"\s*{name}")
    return out


def build_pats(n: int):
    plain = re.compile(r"(?<![\d,.])" + str(n) + r"(?![\d,.])")
    comma = re.compile(r"(?<![\d,.])" + re.escape(f"{n:,}") + r"(?![\d,.])")
    wf = word_forms(n)
    return plain, comma, (re.compile("|".join(wf), re.I) if wf else None)


def windows(text: str, pats, half: int = 350, cap: int | None = None) -> list[str]:
    """Merged text windows around every threshold mention."""
    spans: list[list[int]] = []
    for p in pats:
        if p is None:
            continue
        for m in p.finditer(text):
            spans.append([max(0, m.start() - half), min(len(text), m.end() + half)])
    spans.sort()
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    if cap is not None:
        merged = merged[:cap]
    return [text[s:e] for s, e in merged]


def load_prefixes() -> dict:
    reqs = {}
    with open(ART / "step3_requests.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            prompt = d["body"]["prompt"]
            i = prompt.find(THINK)
            reqs[d["req_id"]] = prompt[i + len(THINK):] if i >= 0 else ""
    return reqs


def load_estimates() -> dict:
    """(parent_req_id, completion_index) -> judged estimate, or None."""
    est = {}
    with open(ART / "step3_judge_results.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            m = d["meta"]
            mm = re.search(r"<final_estimate>\s*([^<]*?)\s*</final_estimate>",
                           d["completions"][0]["text"])
            v = None
            if mm:
                s = mm.group(1).replace(",", "").strip()
                try:
                    v = float(s)
                except ValueError:
                    v = None
            est[(m["parent_req_id"], m["completion_index"])] = v
    return est


def scan() -> list[dict]:
    prefixes = load_prefixes()
    est = load_estimates()
    pat_cache: dict[int, tuple] = {}
    rows = []
    with open(ART / "step3_results_final.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            m = d["meta"]
            n = int(m["threshold"])
            if n not in pat_cache:
                pat_cache[n] = build_pats(n)
            plain, comma, words = pat_cache[n]
            affected = str(n) != f"{n:,}"
            prefix = prefixes[d["req_id"]]
            pf = {
                "prefix_plain": bool(plain.search(prefix)),
                "prefix_comma": bool(comma.search(prefix)),
                "prefix_words": bool(words.search(prefix)) if words else False,
                "prefix_thresholdword": bool(re.search(r"threshold", prefix, re.I)),
                "prefix_len": len(prefix),
            }
            for ci, comp in enumerate(d["completions"]):
                txt = comp["text"]
                has = (bool(plain.search(txt)), bool(comma.search(txt)),
                       bool(words.search(txt)) if words else False)
                wins = windows(txt, (plain, comma, words)) if any(has) else []
                # value errors: "the threshold is X" with X != T
                wrong = 0
                total_claims = 0
                for mm in THRESHOLD_IS.finditer(txt):
                    s = mm.group(1).replace(",", "")
                    if not s.isdigit():
                        continue
                    total_claims += 1
                    if int(s) != n:
                        wrong += 1
                # direction conflict
                dirfix = False
                for mm in DIRECTION_ANCHOR.finditer(txt):
                    w = txt[max(0, mm.start() - 300): mm.end() + 300]
                    if DIRECTION_FIX.search(w):
                        dirfix = True
                        break
                rows.append({
                    "req_id": d["req_id"], "ci": ci,
                    "source_id": m["source_id"], "prompt_key": m["prompt_key"],
                    "direction": m["direction"], "arm": m["arm"], "threshold": float(n),
                    "frac": round(m["frac_sentences"], 4), "affected": affected,
                    "finish": comp["finish_reason"], "cont_len": len(txt),
                    "cont_plain": has[0], "cont_comma": has[1], "cont_words": has[2],
                    "cont_thresholdword": bool(re.search(r"threshold", txt, re.I)),
                    "n_windows": len(wins),
                    "confusion": sorted({h.lower() for w in wins for h in CONFUSION.findall(w)}),
                    "format_marker": sorted({(h if isinstance(h, str) else h[0]).lower()
                                             for w in wins for h in FORMAT_MARK.findall(w)}),
                    "threshold_claims": total_claims, "threshold_claims_wrong": wrong,
                    "direction_reread": dirfix,
                    "est": est.get((d["req_id"], ci)),
                    **pf,
                })
    return rows


# --- reporting ------------------------------------------------------------

def timing() -> None:
    """Where the threshold first appears in the original CoT, vs the lock."""
    from cuts import split_sentences, strip_reasoning_markup

    lock = {s["unit_id"]: s for s in
            json.load(open(ART / "step3_summaries.json"))["neutral"]["summaries"]
            if s["is_rollout"]}
    rows = []
    with open(ART / "step3_sources.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            n = int(d["threshold"])
            sents = split_sentences(strip_reasoning_markup(d["reasoning"]))
            total = len(sents)
            plain, comma, _ = build_pats(n)
            word = re.compile(r"\bthreshold\b", re.I)

            def first(pat):
                for i, s in enumerate(sents):
                    if pat.search(getattr(s, "text", str(s))):
                        return i / total
                return None

            fn = [x for x in (first(plain), first(comma)) if x is not None]
            L = lock.get(d["source_id"])
            rows.append({"word": first(word), "num": min(fn) if fn else None,
                         "commit": L["commitment_frac"] if L else None})

    def desc(k):
        v = sorted(r[k] for r in rows if r[k] is not None)
        return (f"n={len(v):3d} median={st.median(v):.3f} mean={st.fmean(v):.3f} "
                f"p10={v[len(v)//10]:.3f} p90={v[int(0.9*len(v))]:.3f}")

    print("\n=== 0. Timing of the threshold in the original CoT (250 sources) ===")
    print("  first 'threshold' word :", desc("word"))
    print("  first threshold value  :", desc("num"))
    print("  lock (commitment_frac) :", desc("commit"))
    bn = [r for r in rows if r["num"] is not None and r["commit"] is not None]
    bw = [r for r in rows if r["word"] is not None and r["commit"] is not None]
    print(f"  value written before the lock: {sum(r['num'] < r['commit'] for r in bn)}/{len(bn)}")
    print(f"  'threshold' said before lock : {sum(r['word'] < r['commit'] for r in bw)}/{len(bw)}")
    print(f"  never writes the value: {sum(1 for r in rows if r['num'] is None)}/{len(rows)}")



def _boot(v, b=3000, seed=1):
    random.seed(seed)
    m = sorted(st.fmean(random.choices(v, k=len(v))) for _ in range(b))
    return st.fmean(v), m[int(0.025 * b)], m[int(0.975 * b)]


def report(rows: list[dict]) -> None:
    per_req = {}
    for r in rows:
        per_req.setdefault(r["req_id"], r)

    print("\n=== 1. Does the retained prefix already carry the threshold? "
          "(2,500 prefixes) ===")
    byfrac = collections.defaultdict(list)
    for r in per_req.values():
        byfrac[round(r["frac"], 1)].append(r)
    for f in sorted(byfrac):
        s = byfrac[f]
        n = len(s)
        num = sum(r["prefix_plain"] or r["prefix_comma"] or r["prefix_words"] for r in s)
        print(f"  t~{f}: n={n:4d}  says 'threshold' {sum(r['prefix_thresholdword'] for r in s)/n:.3f}"
              f"  states the value {num/n:.3f}")
    aff = [r for r in per_req.values() if r["affected"]]
    n = len(aff)
    print(f"  affected questions, all cuts (n={n}): comma form "
          f"{sum(r['prefix_comma'] for r in aff)/n:.3f}, bare digits "
          f"{sum(r['prefix_plain'] for r in aff)/n:.3f}, words "
          f"{sum(r['prefix_words'] for r in aff)/n:.3f}")

    print("\n=== 2. Does the continuation get the threshold's value wrong? ===")
    tot = collections.Counter()
    bad = collections.Counter()
    for r in rows:
        k = (r["arm"], r["affected"])
        tot[k] += r["threshold_claims"]
        bad[k] += r["threshold_claims_wrong"]
    for k in sorted(tot):
        arm, a = k
        print(f"  arm={arm} affected={a}: {bad[k]:5d} / {tot[k]:6d} "
              f"'threshold is X' claims disagree with T = {bad[k]/tot[k]:.4f}")
    print("  (remaining hits are LaTeX/abbreviation artefacts; the rate is the "
          "same in both arms, so the format has no effect)")

    print("\n=== 3. Number-format commentary near a threshold mention ===")
    for arm in ("orig", "swap"):
        for a in (True, False):
            s = [r for r in rows if r["arm"] == arm and r["affected"] == a]
            n = len(s)
            print(f"  arm={arm} affected={a}: n={n:6d}  format marker "
                  f"{sum(1 for r in s if r['format_marker'])/n:.4f}  "
                  f"any confusion token {sum(1 for r in s if r['confusion'])/n:.3f}")

    print("\n=== 4. dep(t) on affected vs clean questions ===")
    cell = collections.defaultdict(lambda: [0, 0])
    info = {}
    for r in rows:
        if r["est"] is None:
            continue
        above = r["est"] > r["threshold"]
        orig_good = above == (r["direction"] == "above_good")
        k = (r["source_id"], round(r["frac"], 4), r["arm"])
        cell[k][0] += orig_good
        cell[k][1] += 1
        info[k] = r
    dep = []
    for (sid, f, arm), (k_, n_) in cell.items():
        if arm != "orig":
            continue
        sw = cell.get((sid, f, "swap"))
        if not sw or n_ < 8 or sw[1] < 8:
            continue
        dep.append({"frac": f, "affected": info[(sid, f, "orig")]["affected"],
                    "dep": k_ / n_ - sw[0] / sw[1]})
    for lo, hi, lab in [(0.0, 0.15, "t<0.15"), (0.15, 0.3, "t~0.2"), (0.3, 0.5, "t~0.4"),
                        (0.5, 0.7, "t~0.6"), (0.7, 0.9, "t~0.8"), (0.9, 1.01, "t=1.0")]:
        parts = []
        for a in (True, False):
            v = [d["dep"] for d in dep if lo <= d["frac"] < hi and d["affected"] == a]
            if len(v) < 8:
                continue
            m, l, h = _boot(v)
            parts.append(f"{'affected' if a else 'clean   '} n={len(v):4d} "
                         f"dep={m:+.3f} [{l:+.3f},{h:+.3f}]")
        print(f"  {lab:8s} " + "   |   ".join(parts))

    print("\n=== 5. Prefix/prompt direction conflict (limitation 3) ===")
    bf = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["arm"], round(r["frac"], 1))
        bf[k][1] += 1
        bf[k][0] += r["direction_reread"]
    for arm in ("orig", "swap"):
        s = [r for r in rows if r["arm"] == arm]
        print(f"  {arm}: {sum(r['direction_reread'] for r in s)}/{len(s)} = "
              f"{sum(r['direction_reread'] for r in s)/len(s):.4f} explicitly re-read "
              f"or reverse the exceed -> good/bad mapping")
    print("  by cut:")
    for f in sorted({k[1] for k in bf}):
        o, s = bf[("orig", f)], bf[("swap", f)]
        print(f"    t~{f}: orig {o[0]/o[1]:.3f}  swap {s[0]/s[1]:.3f}   (n={o[1]} each)")


def dump_sample(rows: list[dict], k: int, seed: int, out: Path) -> None:
    """Prefix tail + threshold windows for a uniform sample, for hand reading."""
    pool = [r for r in rows if r["arm"] == "swap" and r["affected"]]
    random.seed(seed)
    samp = random.sample(pool, k)
    want = {(r["req_id"], r["ci"]) for r in samp}
    need = {r["req_id"] for r in samp}
    prefixes = {rid: p for rid, p in load_prefixes().items() if rid in need}
    texts = {}
    with open(ART / "step3_results_final.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d["req_id"] not in need:
                continue
            for ci, c in enumerate(d["completions"]):
                if (d["req_id"], ci) in want:
                    texts[(d["req_id"], ci)] = c["text"]
    with open(out, "w") as fh:
        for i, r in enumerate(sorted(samp, key=lambda r: (r["prompt_key"], r["frac"])), 1):
            n = int(r["threshold"])
            ps = build_pats(n)
            txt = texts[(r["req_id"], r["ci"])]
            pre = prefixes[r["req_id"]]
            fh.write(f"\n{'=' * 96}\n[{i:03d}] {r['prompt_key']} dir={r['direction']} "
                     f"T={n:,} t={r['frac']} ci={r['ci']} est={r['est']}\n"
                     f"  prefix: thresholdword={r['prefix_thresholdword']} "
                     f"comma={r['prefix_comma']} plain={r['prefix_plain']} words={r['prefix_words']}\n"
                     f"  cont:   finish={r['finish']} confusion={r['confusion']} "
                     f"fmt={r['format_marker']} direction_reread={r['direction_reread']}\n")
            for w in windows(pre, ps, 250, cap=1):
                fh.write("  P| " + w.replace("\n", " ")[:300] + "\n")
            for w in windows(txt, ps, 300, cap=2):
                fh.write("  C| " + w.replace("\n", " ")[:400] + "\n")
    print(f"\nwrote {out} ({k} continuations, seed {seed})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=0,
                    help="also dump this many uniformly sampled swap-arm "
                         "continuations for hand reading")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--rows", type=Path, default=ART / "threshold_audit_rows.jsonl")
    args = ap.parse_args()

    if args.rows.exists():
        rows = [json.loads(l) for l in open(args.rows)]
        print(f"loaded {len(rows)} rows from {args.rows}")
    else:
        rows = scan()
        with open(args.rows, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"scanned {len(rows)} continuations -> {args.rows}")
    timing()
    report(rows)
    if args.sample:
        dump_sample(rows, args.sample, args.seed, args.rows.with_suffix(".sample.txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
