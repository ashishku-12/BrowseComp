"""
Step 5 — Reasoning-Graph / Redundancy Check Agent
Formalizes the question into nodes/edges across the WHOLE chain and checks
every edge is load-bearing, no shortcut path exists, and no node is
ambiguous.
Output: data/step5_graphchecked.json
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm

INPUT_PATH = os.path.join(DATA_DIR, "step4_questions.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step5_graphchecked.json")

SYSTEM_PROMPT = """You are a Reasoning-Graph / Redundancy-Check Agent.
Given a constructed multihop question and its underlying chain of entities
and relations, formalize it as a graph (nodes = entities/attributes,
edges = relations) and check:

1. Every edge is load-bearing: removing it should make the question
   unanswerable or ambiguous. Flag a clue as redundant ONLY if you are
   confident it adds nothing - don't flag clues just because they seem
   generic.
2. No shortcut path: check whether the given clues let a solver skip one or
   more intermediate nodes entirely and jump ahead. Only flag this if you
   are confident a SPECIFIC, real bypass exists - not a hypothetical or
   weak possibility.
3. No ambiguous nodes: could the obfuscated description of the first entity,
   or the description used to find any intermediate entity, plausibly match
   more than one real entity? Only flag genuine, likely ambiguity.

This chain has exactly {n_hops} hops connecting {n_nodes} entities in
sequence. Report your graph's node/edge counts for reference, but a count
mismatch in your own formalization is NOT by itself a reason to fail -
base "verdict" only on items 1-3 above (a genuine load-bearing violation,
shortcut, or ambiguity).

Return ONLY JSON:
{{
  "verdict": "pass" | "fail",
  "graph": {{"nodes": ["entity1(obfuscated)", "entity2", "..."], "edges": ["relation1", "relation2", "..."]}},
  "issues": ["<any shortcut or ambiguity found, else empty list>"],
  "redundant_clues": ["<any clue found unnecessary, else empty list>"]
}}
"""


def run() -> None:
    items = [c for c in load_json_list(INPUT_PATH) if c.get("construction_status") == "constructed"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    fail_reasons = {}  # diagnostic tally, printed at the end

    for item in items:
        if writer.is_done(item["id"]):
            continue

        c = item["construction"]
        entities = item["entities"]
        hops = item["hops"]
        n_hops = len(hops)
        n_nodes = len(entities)

        chain_desc = " --> ".join(
            f"{entities[i]} --[{hops[i]['relation']}]-->" for i in range(n_hops)
        ) + f" {entities[-1]}"

        user_prompt = (
            f"Question: {c['question']}\n"
            f"Canonical answer (final entity): {c['canonical_answer']}\n"
            f"Underlying chain: {chain_desc}\n"
            f"Obfuscation used for first entity: {c.get('obfuscation_map', {}).get('first_entity')}\n"
        )
        try:
            verdict = call_local_llm(
                SYSTEM_PROMPT.format(n_hops=n_hops, n_nodes=n_nodes),
                user_prompt
            )
        except Exception as e:
            verdict = {"verdict": "fail", "issues": [f"redundancy agent error: {e}"]}

        # Defensive check - flag (don't force-fail) a structural count mismatch
        # so it's visible in the data instead of silently trusted either way
        reported_nodes = len(verdict.get("graph", {}).get("nodes", []))
        reported_edges = len(verdict.get("graph", {}).get("edges", []))
        if reported_nodes and reported_nodes != n_nodes:
            verdict["node_count_mismatch_warning"] = f"expected {n_nodes} nodes, model reported {reported_nodes}"
        if reported_edges and reported_edges != n_hops:
            verdict["edge_count_mismatch_warning"] = f"expected {n_hops} edges, model reported {reported_edges}"

        record = {**item, "graph_check": verdict}
        writer.append(record)
        reason = verdict.get("verdict", "unknown")
        fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
        print(f"[step5] {item['id']}: {reason}")

    print(f"\n[step5] summary: {fail_reasons}")


if __name__ == "__main__":
    run()