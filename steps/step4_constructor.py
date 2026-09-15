"""
Step 4 — Question Constructor Agent

Constructs a natural multihop research question from the chain, using ONLY
each hop's broad relation CATEGORY (relation_type_used) as the clue - not
the specific extracted relation phrase, and not the supporting sentence.

Rationale: relation_type_used, anchored to a specific entity, requires a
solver to do the SAME discovery search Step 2 originally performed to find
that hop - genuine re-discovery. The specific extracted relation phrase and
supporting sentence are already-digested results of that search, and handing
them to a solver short-circuits the work rather than requiring it.

Entity 1 is the only entity named directly (it's a legitimate, vetted-obscure
starting point, never the answer). Every entity from Entity 2 through the
Final Entity is described ONLY through its relation-category clue, never by
name - this closes a real gap where an intermediate entity, if named, could
let a solver skip straight past the chain that was built to find it.

Output: data/step4_questions.json
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm


INPUT_PATH = os.path.join(DATA_DIR, "step3_filtered.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step4_questions.json")


SYSTEM_PROMPT = """You are a Question Constructor Agent for a difficult multihop
web-research question dataset.

You are given a chain of entities:

Entity 1 -> Entity 2 -> ... -> Final Entity

connected hop by hop, where each hop has a RELATION CATEGORY describing what
kind of connection links the two entities (e.g. "leadership", "election",
"institution_affiliation"). The LAST entity is always the answer.

RULES:

1. Name Entity 1 directly and describe it using its known attributes, so the
   solver has a legitimate, verifiable starting point for research.

2. NEVER name any entity from Entity 2 through the Final Entity. Each must be
   identified ONLY by describing the relation category that connects it to
   the entity immediately before it in the chain - phrased as natural
   language, not as the raw category label (e.g. turn "leadership" into
   something like "a person who took on a leadership role connected to
   [the previous entity]" - never the bare word "leadership" itself).

3. Do NOT use any specific extracted fact, date, or detail beyond the
   relation category - the solver should have to search using the entity
   and the category type, the same way this chain was originally
   discovered, not be handed an already-extracted specific that shortcuts
   that search.

4. Chain the clues in forward reading order - Entity 1 through the final
   category-clue pointing at the Final Entity - so the question reads as a
   single coherent research trail.

5. Do not mention the final answer directly, by name or by unique
   description.

6. Make the question read as a natural research puzzle, not a list of
   category labels or graph edges.

Return ONLY JSON:

{
  "question": "<final multihop research question, presented forward>",
  "canonical_answer": "<final entity>",
  "obfuscation_map": {
    "first_entity": "<description of how Entity 1 was presented>"
  }
}
"""


def run() -> None:
    chains = [
        c for c in load_json_list(INPUT_PATH)
        if c.get("filter", {}).get("verdict") == "pass"
    ]

    writer = ResumableWriter(OUTPUT_PATH, key="id")

    for chain in chains:
        if writer.is_done(chain["id"]):
            continue

        entities = chain["entities"]
        hops = chain["hops"]

        chain_lines = [
            f"Entity 1 (name this directly): {entities[0]}",
            f"Entity 1 attributes: {chain.get('entity_A_attributes', {})}",
        ]

        for i, hop in enumerate(hops):
            chain_lines.append(f"\nHop {i + 1}:")
            chain_lines.append(f"From entity: {entities[i]}")
            chain_lines.append(f"Relation category: {hop.get('relation_type_used', 'unknown')}")
            chain_lines.append(f"To entity (do NOT name this): {entities[i + 1]}")

        chain_lines.append(f"\nFinal answer (do NOT name or uniquely describe this): {entities[-1]}")

        user_prompt = "\n".join(chain_lines)

        try:
            out = call_local_llm(
                SYSTEM_PROMPT,
                user_prompt,
                sample=True,
            )

            out["canonical_answer"] = entities[-1]

            if "obfuscation_map" not in out:
                out["obfuscation_map"] = {
                    "first_entity": ""
                }

            status = "constructed"

        except Exception as e:
            out = {
                "question": None,
                "canonical_answer": None,
                "obfuscation_map": {
                    "first_entity": ""
                },
                "error": str(e),
            }
            status = "failed"

        record = {
            **chain,
            "construction": out,
            "construction_status": status
        }

        writer.append(record)

        print(f"[step4] {chain['id']}: {status}")


if __name__ == "__main__":
    run()