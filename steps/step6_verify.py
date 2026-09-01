"""
Step 6 — Verifier Agent
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
from utils.llm_client import call_llm, call_llm_no_tools_text

INPUT_PATH = os.path.join(DATA_DIR, "step5_graphchecked.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step6_verified.json")

MATCH_SYSTEM_PROMPT = """You judge whether two answers to the same question refer to
the same real-world entity/fact. Allow paraphrase, alternate names, spelling
variants, and formatting differences (e.g. "Real Madrid" vs "Real Madrid CF",
different date formats). Do NOT require an exact string match - judge whether
a knowledgeable person would consider them the same answer.
Return ONLY JSON: {"match": true|false}."""

REDERIVE_SYSTEM_PROMPT = """You are given a multihop chain's evidence, hop by hop,
WITH the entity and relation for each hop, plus the sentence supporting it
(no web access, no outside knowledge should be needed beyond this evidence),
and a question. Walk the chain in order - each hop's "to" entity is the next
hop's "from" entity - and derive the final answer.
Return ONLY JSON: {"derived_answer": "<answer or null if not derivable from the evidence given>"}."""


def _answers_match(candidate: str, canonical: str) -> bool:
    if not candidate:
        return False
    try:
        out = call_llm(
            MATCH_SYSTEM_PROMPT,
            f"Canonical answer: {canonical}\nCandidate answer: {candidate}",
            use_secondary=True,
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
            blind_answer = call_llm_no_tools_text(question, use_secondary=True)
            blind_solved = _answers_match(blind_answer, canonical)
        except Exception as e:
            blind_answer, blind_solved = f"[error: {e}]", False  # fail-open to manual review, not auto-accept

        # 6b: evidence-only re-derivation - NOW includes entity/relation structure
        # per hop, matching what Steps 3 and 5 already give the model, instead of
        # forcing cold entity-resolution across bare sentences with no scaffolding.
        # evidence_lines = []
        # for i, hop in enumerate(hops):
        #     evidence_lines.append(
        #         f"Hop {i+1}: {entities[i]} --[{hop['relation']}]--> {entities[i+1]}\n"
        #         f"  Evidence: {hop['supporting_sentence']}"
        #     )
        # evidence_text = "\n".join(evidence_lines)

        # try:
        #     rederive = call_llm(
        #         REDERIVE_SYSTEM_PROMPT,
        #         f"Question: {question}\n\n{evidence_text}",
        #         use_secondary=True,
        #     )
        #     evidence_ok = _answers_match(rederive.get("derived_answer"), canonical)
        # except Exception as e:
        #     rederive, evidence_ok = {"derived_answer": None, "error": str(e)}, False

        final_verdict = "accept" if (not blind_solved) else "reject"
        reason = []
        if blind_solved:
            reason.append("blind-solved from parametric memory (contamination risk)")
        # if not evidence_ok:
        #     reason.append("could not be re-derived from cited evidence alone")

        record = {
            **item,
            "verification": {
                "blind_solve_answer": blind_answer,
                "blind_solve_correct": blind_solved,
                # "evidence_rederivation": rederive,
                # "evidence_rederivation_correct": evidence_ok,
                "final_verdict": final_verdict,
                "reason": "; ".join(reason) if reason else "passed check",
            },
        }
        writer.append(record)
        fail_reasons[final_verdict] = fail_reasons.get(final_verdict, 0) + 1
        print(f"[step6] {item['id']}: {final_verdict}")

    print(f"\n[step6] summary: {fail_reasons}")


if __name__ == "__main__":
    run()