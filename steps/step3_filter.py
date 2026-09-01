"""
Step 3 — Filtering Agent
Scores every intermediate bridge entity's popularity/salience, checks source
credibility for every hop, and checks hop independence across the whole
chain. Rejects chains where any bridge is too famous (shortcut risk) or
sources are weak/duplicated.
Output: data/step3_filtered.json
"""
import os
from config import DATA_DIR, DEEPSEEK_MODEL_CHEAP, LOW_CREDIBILITY_DOMAINS
from utils.io_utils import ResumableWriter, load_json_list
from utils.llm_client import call_llm

INPUT_PATH = os.path.join(DATA_DIR, "step2_chains.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step3_filtered.json")

SYSTEM_PROMPT = """You are a Filtering Agent for a multihop question dataset.
Given a chain of entities connected by a sequence of relations, with one
source sentence per hop, evaluate:

1. Bridge entity salience: for each INTERMEDIATE entity (every entity except
   the first and the final answer), is it a globally famous / "household name"
   entity? Only a TRULY famous, front-page-recognizable entity should count
   as "high" salience and cause a FAIL - a moderately well-known professional,
   local official, or niche-but-documented organization is "low" or "medium"
   and should NOT fail on salience alone.
2. Source credibility: is every hop's source reasonably reputable (official
   sites, established news, institutional pages, verifiable databases)
   rather than low-quality (forums, unsourced wikis, content farms, spam)?
   "medium" credibility sources are acceptable and should not by themselves
   cause a FAIL - only "low" credibility sources should.
3. Hop independence: do the source URLs differ across ALL hops, and does no
   single source already state multiple hops of the connection together?

This chain has exactly {n_hops} hops and {n_bridges} intermediate (bridge)
entities. Your response arrays MUST have EXACTLY these lengths:
- "source_credibility": exactly {n_hops} entries, one per hop, in order.
- "bridge_entity_salience": exactly {n_bridges} entries, one per intermediate
  entity, in order (empty list [] if there are zero intermediate entities).

Return ONLY JSON:
{{
  "verdict": "pass" | "fail",
  "bridge_entity_salience": ["low"|"medium"|"high", ...],
  "source_credibility": ["high"|"medium"|"low", ...],
  "hop_independence": "confirmed" | "violated",
  "notes": "<short explanation, especially if failing>"
}}
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
        user_prompt = "Chain:\n" + "\n".join(chain_desc_lines)

        try:
            verdict = call_llm(
                SYSTEM_PROMPT.format(n_hops=n_hops, n_bridges=n_bridges),
                user_prompt, model=DEEPSEEK_MODEL_CHEAP, use_secondary=True,
            )
        except Exception as e:
            verdict = {"verdict": "fail", "notes": f"filter agent error: {e}"}

        # Defensive check - flag (don't force-fail) array-length mismatches so
        # they're visible in the data instead of silently trusted
        sal = verdict.get("bridge_entity_salience", [])
        cred = verdict.get("source_credibility", [])
        if len(sal) != n_bridges or len(cred) != n_hops:
            verdict["length_mismatch_warning"] = (
                f"expected {n_bridges} salience / {n_hops} credibility entries, "
                f"got {len(sal)} / {len(cred)}"
            )

        record = {**chain, "filter": verdict}
        writer.append(record)
        reason = verdict.get("verdict", "unknown")
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        print(f"[step3] {chain['id']}: {reason}")

    print(f"\n[step3] summary: {fail_reasons}")


if __name__ == "__main__":
    run()