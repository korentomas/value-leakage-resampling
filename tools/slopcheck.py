#!/usr/bin/env python3
"""Check prose for markers of LLM authorship, tuned for Claude rather than GPT.

Why this is not the usual banned-word list
------------------------------------------
The widely circulated marker lists ("delve", "tapestry", "landscape",
"showcase") are GPT markers, derived from excess-vocabulary studies of
post-2023 abstracts (Kobak et al., arXiv:2406.07016). Claude largely avoids
those words, so scoring zero against that list says almost nothing about
whether Claude wrote the text. Claude's reported tells are structural instead:
heavy em dashes, three-item lists, negative parallelism, stacked transitions,
formal subordination, and uniform sentence rhythm.

So the lexical section here is short and Claude-specific, and the weight sits on
the structural and statistical checks.

Thresholds, from published guidance where it exists:
  em dashes      > 20 per 1,000 words
  burstiness     stdev(sentence length)/mean; AI 0.2-0.4, human 0.6-1.2, flag < 0.4
  three-item list  > 1 per 200 words
  transitions    sentence-initial connectives > 3% of sentences

Caveat worth reading before acting on the output. Detector-style metrics have
false-positive rates above 60% on non-native English (see the AI-detection
literature on perplexity/burstiness), because regular grammar and plainer
vocabulary look like low perplexity. Chasing a score can therefore strip the
exact idiom that makes writing someone's own. Treat every number below as a
prompt to reread a passage, never as a verdict.

Usage:  python tools/slopcheck.py paper.tex README.md
"""
import re
import statistics as st
import sys

# Claude-specific lexical tells. Deliberately short: the GPT vocabulary list is
# not evidence about Claude.
CLAUDE_WORDS = [
    "delve", "tapestry", "showcase", "underscore",   # kept only to confirm absence
    "quiet", "genuinely", "precisely", "essentially", "fundamentally",
    "nuanced", "compelling", "meaningful", "thoughtful", "careful",
]

PATTERNS = {
    "negative parallelism": r"(is|it'?s|that'?s|was)\s+not\s+(just|merely|only|simply)\b|not only\b[^.]{0,60}\bbut\b",
    "no X no Y just Z": r"\bNo [A-Za-z]+\.\s+No [A-Za-z]+\.\s+Just\b",
    "empty opener": r"\b(it is important to|it is worth noting|it should be noted|when it comes to|in today'?s|one of the most)\b",
    "wrap-up ending": r"(^|\.\s)(In conclusion|Ultimately|In summary|Taken together|All in all|At the end of the day)\b",
    "trailing participle": r",\s+(highlighting|underscoring|emphasizing|reflecting|showcasing|ensuring|contributing|demonstrating|illustrating|allowing|enabling|making it|cementing|solidifying)\b",
    "dodged copula": r"\b(serves|stands|acts|functions) as\b|\bboasts\b|\bplays an? [a-z]+ role\b",
    "vague attribution": r"\b(experts|observers|critics|researchers|some|many|industry reports)\s+(argue|believe|note|have noted|suggest|say)\b",
    "em dash": r"\u2014",
    "curly quote": r"[\u201c\u201d\u2018\u2019]",
    "title case heading": r"^\\(sub)?section\{(?:[A-Z][a-z]+ ){2,}[A-Z][a-z]+\}",
}

TRANSITIONS = ["however", "moreover", "furthermore", "additionally", "therefore",
               "thus", "consequently", "nevertheless", "notably", "indeed",
               "overall", "importantly", "crucially"]

HEDGES = ["may", "might", "could", "possibly", "potentially", "arguably",
          "seems", "appears", "relatively", "somewhat", "fairly", "generally",
          "typically", "often", "perhaps"]

SUBORDINATORS = ["which", "that", "while", "whereas", "although", "though",
                 "because", "since", "unless", "whether"]


def strip_markup(text, is_tex):
    if is_tex:
        text = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ", text, flags=re.S)
        text = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", " ", text, flags=re.S)
        text = re.sub(r"\$[^$]*\$", " X ", text)
        text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", text)
    else:
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"`[^`]*`", " ", text)
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) > 3]


def report(path):
    raw = open(path).read()
    is_tex = path.endswith(".tex")
    prose = strip_markup(raw, is_tex)
    sents = sentences(prose)
    words = re.findall(r"[A-Za-z']+", prose)
    n_words = len(words)
    low = prose.lower()
    per1k = lambda n: 1000.0 * n / max(n_words, 1)

    print(f"\n=== {path}  ({n_words} prose words, {len(sents)} sentences) ===")

    print("-- structural patterns --")
    flagged = False
    for name, pat in PATTERNS.items():
        hits = re.findall(pat, prose, flags=re.I | re.M)
        n = len(hits)
        if name == "em dash":
            rate = per1k(n)
            bad = rate > 20
            print(f"   {name:22s} {n:4d}  ({rate:.1f}/1k words, flag >20)  {'FLAG' if bad else 'ok'}")
            flagged |= bad
        elif n:
            print(f"   {name:22s} {n:4d}  FLAG")
            flagged = True
    if not flagged:
        print("   (no structural pattern over threshold)")

    # three-item lists: "A, B, and C"
    tri = re.findall(r"\b[\w-]+(?: [\w-]+){0,2}, [\w-]+(?: [\w-]+){0,2},? and [\w-]+(?: [\w-]+){0,2}\b", prose)
    tri_density = len(tri) / max(n_words / 200.0, 1e-9)
    print(f"   three-item lists       {len(tri):4d}  ({tri_density:.2f} per 200 words, flag >1)  "
          f"{'FLAG' if tri_density > 1 else 'ok'}")

    # transition stacking
    tr = sum(1 for s in sents if s.split() and s.split()[0].strip(",").lower() in TRANSITIONS)
    tr_pct = 100.0 * tr / max(len(sents), 1)
    print(f"   sentence-initial transitions {tr:3d}  ({tr_pct:.1f}% of sentences, flag >3%)  "
          f"{'FLAG' if tr_pct > 3 else 'ok'}")

    # hedge stacking: 2+ hedges in one sentence
    hedge_re = re.compile(r"\b(" + "|".join(HEDGES) + r")\b", re.I)
    stacked = [s for s in sents if len(hedge_re.findall(s)) >= 2]
    print(f"   hedge-stacked sentences {len(stacked):3d}  (2+ hedges in one sentence)")

    print("-- rhythm (the strongest remaining tell) --")
    lens = [len(s.split()) for s in sents]
    cv = st.pstdev(lens) / st.mean(lens)
    verdict = "FLAG (reads mechanical)" if cv < 0.4 else ("ok" if cv < 0.6 else "ok, strongly human")
    print(f"   burstiness (CV)        {cv:.2f}   AI 0.2-0.4 / human 0.6-1.2 / flag <0.4 -> {verdict}")
    print(f"   sentence words         mean {st.mean(lens):.1f}  median {st.median(lens):.0f}  "
          f"min {min(lens)}  max {max(lens)}")
    print(f"   short (<=8w) {sum(1 for l in lens if l <= 8):3d}   long (>=35w) {sum(1 for l in lens if l >= 35):3d}")

    # paragraph uniformity: another uniformity signal
    paras = [p for p in re.split(r"\n\s*\n", prose) if len(p.split()) > 25]
    if len(paras) > 3:
        plens = [len(p.split()) for p in paras]
        pcv = st.pstdev(plens) / st.mean(plens)
        print(f"   paragraph length CV    {pcv:.2f}  ({len(paras)} paragraphs; very low = assembled)")

    # subordination: Claude reportedly adds formal subordination
    sub = sum(len(re.findall(r"\b" + w + r"\b", low)) for w in SUBORDINATORS)
    print(f"   subordinators/sentence {sub / max(len(sents),1):.2f}  (>1.5 reads clause-heavy)")

    print("-- Claude-specific vocabulary (absence is weak evidence; presence is worth a look) --")
    hits = [(w, len(re.findall(r"\b" + w + r"\w*\b", low))) for w in CLAUDE_WORDS]
    hits = [(w, c) for w, c in hits if c]
    print("   " + (", ".join(f"{w}x{c}" for w, c in hits) if hits else "none"))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
    print("\nNumbers are prompts to reread, not verdicts. Detector-style metrics")
    print("false-positive above 60% on non-native English; do not sand idiom out")
    print("to chase a score.")
