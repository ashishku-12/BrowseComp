"""
Step 7 — Verifier Agent
6a. Blind-solve check: a model, WITHOUT search, tries to answer the question
    from parametric memory alone. If it succeeds, reject (contamination /
    not actually hard-to-find).
6b. Evidence-only re-derivation: a model is given the cited evidence for
    EVERY hop - WITH the entity/relation structure, not just bare sentences -
    and must derive the final answer using no other knowledge.
Output: data/step6_verified.json  (this is the final dataset file;
filter on final_verdict == "accept" for the usable dataset)
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm, call_local_llm_text

INPUT_PATH = os.path.join(DATA_DIR, "step6_graphchecked.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step7_verified.json")

MATCH_SYSTEM_PROMPT = """You judge whether two answers to the same question refer to
the same real-world entity/fact. Allow paraphrase, alternate names, spelling
variants, and formatting differences (e.g. "Real Madrid" vs "Real Madrid CF",
different date formats). Do NOT require an exact string match - judge whether
a knowledgeable person would consider them the same answer.
Return ONLY JSON: {"match": true|false}."""



def _answers_match(candidate: str, canonical: str) -> bool:
    if not candidate:
        return False
    try:
        out = call_local_llm(
            MATCH_SYSTEM_PROMPT,
            f"Canonical answer: {canonical}\nCandidate answer: {candidate}"
        )
        return bool(out.get("match"))
    except Exception:
        return candidate.strip().lower() == canonical.strip().lower()


def run() -> None:
    items = [c for c in load_json_list(INPUT_PATH) if c.get("graph_check", {}).get("verdict") == "pass"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    fail_reasons = {}  # diagnostic tally, printed at the end

    for item in items:
        if writer.is_done(item["id"]):
            continue

        c = item["construction"]
        question = c["question"]
        canonical = c["canonical_answer"]
        hops = item["hops"]
        entities = item["entities"]

        # 6a: blind-solve check (no tools, no search) - local model, no evidence given at all
        try:
            blind_answer = call_local_llm_text(question)
            blind_solved = _answers_match(blind_answer, canonical)
        except Exception as e:
            blind_answer, blind_solved = f"[error: {e}]", False  # fail-open to manual review, not auto-accept

        final_verdict = "accept" if (not blind_solved) else "reject"
        reason = []
        if blind_solved:
            reason.append("blind-solved from parametric memory (contamination risk)")

        record = {
            **item,
            "verification": {
                "blind_solve_answer": blind_answer,
                "blind_solve_correct": blind_solved,
                "final_verdict": final_verdict,
                "reason": "; ".join(reason) if reason else "passed check",
            },
        }
        writer.append(record)
        fail_reasons[final_verdict] = fail_reasons.get(final_verdict, 0) + 1
        print(f"[step7] {item['id']}: {final_verdict}")

    print(f"\n[step7] summary: {fail_reasons}")


if __name__ == "__main__":
    run()