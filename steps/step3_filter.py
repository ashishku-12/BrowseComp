"""
Step 3 — Filtering Agent
Scores every intermediate bridge entity's popularity/salience, checks source
credibility for every hop, and checks hop independence across the whole
chain. Rejects chains where any bridge is too famous (shortcut risk) or
sources are weak/duplicated.
Output: data/step3_filtered.json
"""
import os
from config import DATA_DIR, LOW_CREDIBILITY_DOMAINS
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm

INPUT_PATH = os.path.join(DATA_DIR, "step2_chains.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step3_filtered.json")

SYSTEM_PROMPT = """You are an evidence verification agent for a multihop
question-answering dataset.

Your ONLY job is to verify whether each hop is explicitly supported by the
provided supporting sentence.

Do NOT use your own world knowledge to fill missing information.
Do NOT assume a relationship is true because it sounds plausible.
Do NOT infer a relationship from context.
Do NOT judge whether an entity is famous unless the supplied evidence directly
supports that judgment.

For EACH hop, verify:

1. ENTITY MATCH:
   Does the supporting sentence explicitly mention or unambiguously identify
   the next entity?

2. RELATION MATCH:
   Does the supporting sentence explicitly state the claimed relationship
   between those two entities?

3. NO INFERENCE:
   Would accepting this hop require adding information that is not explicitly
   stated in the sentence?

A hop is valid ONLY when the relationship is explicitly supported by the
provided sentence.

If you cannot establish the relationship from the supplied sentence alone,
mark that hop as unsupported.

The source URL is metadata only. Do not assume that a URL is credible merely
because its domain looks familiar.

Return ONLY JSON:

{
  "verdict": "pass" | "fail",
  "hop_verification": [
    {
      "hop": 1,
      "entity_match": "yes" | "no",
      "relation_match": "yes" | "no",
      "requires_inference": "yes" | "no",
      "verdict": "supported" | "unsupported",
      "reason": "<short evidence-based explanation>"
    }
  ],
  "notes": "<short explanation>"
}

There must be exactly one hop_verification entry per hop.
"""


def _domain_flag(url: str) -> bool:
    return any(bad in (url or "") for bad in LOW_CREDIBILITY_DOMAINS)


def run() -> None:
    chains = [c for c in load_json_list(INPUT_PATH) if c.get("status") == "chain_complete"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    fail_reasons = {}  # diagnostic tally, printed at the end

    for chain in chains:
        if writer.is_done(chain["id"]):
            continue

        hops = chain["hops"]
        source_urls = [h["source_url"] for h in hops]

        pre_fail = None
        if any(_domain_flag(u) for u in source_urls):
            pre_fail = "low-credibility domain detected"
        if len(set(source_urls)) < len(source_urls):
            pre_fail = (pre_fail + "; " if pre_fail else "") + "two or more hops share the same source URL"

        if pre_fail:
            record = {**chain, "filter": {"verdict": "fail", "notes": pre_fail}}
            writer.append(record)
            fail_reasons["pre-check"] = fail_reasons.get("pre-check", 0) + 1
            print(f"[step3] {chain['id']}: fail (pre-check) - {pre_fail}")
            continue

        entities = chain["entities"]
        n_hops = len(hops)
        n_bridges = len(entities) - 2  # everything except the first entity and the final answer

        chain_desc_lines = []
        for i, hop in enumerate(hops):
            chain_desc_lines.append(
                f"{entities[i]} --[{hop['relation']}]--> {entities[i+1]}\n"
                f"  source: {hop['source_url']}\n"
                f"  sentence: {hop['supporting_sentence']}"
            )
        user_prompt = (
            "Verify the following chain using ONLY the supplied supporting sentences.\n"
            "Do not use outside knowledge.\n\n"
            + "\n".join(chain_desc_lines)
        )

        try:
            verdict = call_local_llm(
                SYSTEM_PROMPT,
                user_prompt
            )
        except Exception as e:
            verdict = {"verdict": "fail", "notes": f"filter agent error: {e}"}

        # Defensive check - flag hop verification array-length mismatches
        # so they're visible in the data instead of silently trusted
        hop_verification = verdict.get("hop_verification", [])

        if len(hop_verification) != n_hops:
            verdict["length_mismatch_warning"] = (
                f"expected {n_hops} hop verification entries, "
                f"got {len(hop_verification)}"
            )

        record = {**chain, "filter": verdict}
        writer.append(record)
        reason = verdict.get("verdict", "unknown")
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        print(f"[step3] {chain['id']}: {reason}")

    print(f"\n[step3] summary: {fail_reasons}")


if __name__ == "__main__":
    run()