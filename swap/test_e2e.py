"""End-to-end smoke test of the driver against a mock OpenAI-compatible server.

Exercises the parts unit tests cannot: the async runner, retry on transient
failures, resumability after an interruption, and the full
requests -> results -> judge -> counts -> model handoff.

The mock server plays a biased model: continuations sampled under the
``above_good`` prompt land above the threshold more often than those under
``below_good``, and the bias shrinks as the spliced prefix grows. So the
pipeline should recover a positive dep that decays with cut position.

Run: ``python test_e2e.py``.
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from driver import (
    Source,
    aggregate_counts,
    build_judge_requests,
    build_requests,
    completed_ids,
    iter_jsonl,
    run_requests,
)
from template import THINK_OPEN

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{': ' + detail if detail else ''}")
        _FAILURES.append(label)


THRESHOLD = 1000.0


class MockHandler(BaseHTTPRequestHandler):
    """Serves /v1/completions and /v1/chat/completions with a scripted bias."""

    fail_first_n = 0
    _seen: dict[str, int] = {}
    _lock = threading.Lock()

    def log_message(self, *args):  # silence the default stderr logging
        pass

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        with MockHandler._lock:
            key = self.path
            MockHandler._seen[key] = MockHandler._seen.get(key, 0) + 1
            count = MockHandler._seen[key]
        if count <= MockHandler.fail_first_n:
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b"transient")
            return

        if self.path.endswith("/chat/completions"):
            payload = self._judge(body)
        else:
            payload = self._complete(body)
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _complete(self, body):
        prompt = body["prompt"]
        prefix = prompt.split(f"{THINK_OPEN}\n", 1)[1]
        # Longer spliced prefix -> the prompt matters less.
        decay = max(0.0, 1.0 - len(prefix) / 400.0)
        base = 0.5
        lean = 0.35 * decay
        p_above = base + (lean if "ABOVE" in prompt else -lean)
        rng = random.Random(hash(prompt) & 0xFFFF)
        choices = []
        for _ in range(body.get("n", 1)):
            value = 1500 if rng.random() < p_above else 500
            choices.append(
                {
                    "text": f" more reasoning.</think>\n\nMy estimate is {value}.",
                    "finish_reason": "stop",
                }
            )
        return {"choices": choices, "usage": {"completion_tokens": 12 * len(choices)}}

    def _judge(self, body):
        text = body["messages"][0]["content"]
        value = "1500" if "is 1500" in text else "500"
        return {
            "choices": [
                {
                    "message": {
                        "content": f"<final_estimate>{value}</final_estimate>"
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 8},
        }


def _sources(n: int) -> list[Source]:
    out = []
    for i in range(n):
        direction = "above_good" if i % 2 == 0 else "below_good"
        marker = "ABOVE" if direction == "above_good" else "BELOW"
        counter = "BELOW" if direction == "above_good" else "ABOVE"
        out.append(
            Source(
                source_id=f"s{i}",
                prompt_key="v1_giraffes",
                direction=direction,
                threshold=THRESHOLD,
                prompt=f"{marker} prompt, threshold 1000",
                counter_prompt=f"{counter} prompt, threshold 1000",
                reasoning=" ".join(
                    f"Reasoning step {j} for rollout {i}." for j in range(12)
                ),
            )
        )
    return out


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            results = tmp / "results.jsonl"
            judge_results = tmp / "judge_results.jsonl"

            print("request building")
            sources = _sources(8)
            reqs = build_requests(
                sources, model="mock", n_cuts=6, n_continuations=20, chunk=10
            )
            check("8 sources x 5 cuts x 2 arms x 2 chunks = 160 requests",
                  len(reqs) == 160, str(len(reqs)))

            print("partial run then resume")
            first_half = reqs[:60]
            import asyncio

            stats1 = asyncio.run(
                run_requests(first_half, results, base_url=base_url, concurrency=16)
            )
            check("first pass completed its slice",
                  stats1["ok"] == 60 and stats1["failed"] == 0, json.dumps(stats1))

            stats2 = asyncio.run(
                run_requests(reqs, results, base_url=base_url, concurrency=16)
            )
            check("resume skips the already-done requests",
                  stats2["already_done"] == 60, json.dumps(stats2))
            check("resume finishes the rest",
                  stats2["ok"] == 100 and stats2["failed"] == 0, json.dumps(stats2))

            stats3 = asyncio.run(
                run_requests(reqs, results, base_url=base_url, concurrency=16)
            )
            check("a third pass is a no-op",
                  stats3["attempted"] == 0, json.dumps(stats3))
            check("results hold one line per request",
                  len(completed_ids(results)) == 160,
                  str(len(completed_ids(results))))
            check("token usage was accumulated",
                  stats2["completion_tokens"] > 0, str(stats2["completion_tokens"]))

            print("retry on transient failure")
            MockHandler.fail_first_n = 3
            MockHandler._seen.clear()
            retry_results = tmp / "retry.jsonl"
            stats4 = asyncio.run(
                run_requests(
                    reqs[:8], retry_results, base_url=base_url,
                    concurrency=1, max_retries=5,
                )
            )
            MockHandler.fail_first_n = 0
            check("503s were retried rather than recorded as failures",
                  stats4["ok"] == 8 and stats4["failed"] == 0, json.dumps(stats4))

            print("judging")
            judge_reqs = build_judge_requests(
                results, judge_model="mock-judge",
                judge_path="/v1/chat/completions",
            )
            check("judge requests were built", len(judge_reqs) > 0, str(len(judge_reqs)))
            check("identical answers are judged once",
                  len(judge_reqs) < 160 * 10, str(len(judge_reqs)))
            stats5 = asyncio.run(
                run_requests(judge_reqs, judge_results, base_url=base_url,
                             concurrency=16)
            )
            check("judging completed", stats5["failed"] == 0, json.dumps(stats5))

            print("aggregation")
            counts = aggregate_counts(results, judge_results)
            check("every continuation was scored",
                  all(c["n_unparsed"] == 0 for c in counts),
                  str([c["n_unparsed"] for c in counts][:5]))
            total = sum(c["n"] for c in counts)
            # 8 sources x 5 paid cuts x 2 arms x 20 continuations.
            check("1600 scored continuations (8 x 5 x 2 x 20)",
                  total == 1600, str(total))
            check("counts carry both arms",
                  {c["arm"] for c in counts} == {"orig", "swap"})
            check("counts are per (source, cut, arm)",
                  len(counts) == 8 * 5 * 2, str(len(counts)))

            print("dep decays with cut position (mock ground truth)")
            by_frac = {}
            for c in counts:
                by_frac.setdefault(round(c["frac_sentences"], 1), {"orig": [0, 0],
                                                                   "swap": [0, 0]})
                slot = by_frac[round(c["frac_sentences"], 1)][c["arm"]]
                slot[0] += c["k"]
                slot[1] += c["n"]
            deps = {}
            for frac, arms in sorted(by_frac.items()):
                p_o = arms["orig"][0] / arms["orig"][1]
                p_s = arms["swap"][0] / arms["swap"][1]
                deps[frac] = p_o - p_s
            print("   dep by CoT fraction: "
                  + ", ".join(f"{f}:{d:+.2f}" for f, d in sorted(deps.items())))
            fracs = sorted(deps)
            check("dep is positive at the earliest cut", deps[fracs[0]] > 0.2,
                  f"{deps[fracs[0]]:.3f}")
            check("dep is smaller at the last cut than the first",
                  deps[fracs[-1]] < deps[fracs[0]],
                  f"{deps[fracs[-1]]:.3f} vs {deps[fracs[0]]:.3f}")

            print("model handoff")
            from model import prepare_data

            data = prepare_data(counts)
            check("prepare_data indexes every rollout",
                  int(data.unit_is_rollout.sum()) == 8,
                  str(int(data.unit_is_rollout.sum())))
            check("observations reach the model",
                  len(data.obs_k) == 8 * 5 * 2, str(len(data.obs_k)))

            print("corrupted-tail tolerance")
            with open(results, "a") as fh:
                fh.write('{"req_id": "truncated line, no clos')
            check("a truncated final line is skipped, not fatal",
                  len(list(iter_jsonl(results))) == 160,
                  str(len(list(iter_jsonl(results)))))
    finally:
        server.shutdown()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} failure(s): {_FAILURES}")
        return 1
    print("end-to-end smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
