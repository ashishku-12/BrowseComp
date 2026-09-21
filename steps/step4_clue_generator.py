"""
Step 4 — Clue Generator

Builds ONE short, obscured clue per hop, in REVERSE order (last hop first),
so each clue can anchor to the next hop's already-built clue text instead
of a name.

FIX: previously grounded primarily on `relation` - often a short, under-
descriptive phrase that doesn't explain WHY the target entity was picked,
causing weird/mismatched clues. Now grounds PRIMARILY on supporting_sentence
(the actual evidence explaining the real connection - the "terminal fact"),
with relation kept only as a secondary/backup label. This is safe here
specifically because Step 4's whole job is compressing/obscuring evidence
into a short clue - Step 5 (the final user-facing question) never sees
supporting_sentence directly anymore, only this step's already-obscured
output, so the earlier over-hinting risk doesn't reapply.

Output: data/step4_clues.json
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm

INPUT_PATH = os.path.join(DATA_DIR, "step3_filtered.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step4_clues.json")

CLUE_SYSTEM_PROMPT = """You write ONE short clue phrase identifying a target
entity, for a multihop research puzzle. The solver must find this entity via
genuine search - never by having its name stated.

You are given:
- source_entity: the entity the chain is currently at.
- target_entity (PRIVATE - for locating it in the evidence only, NEVER state
  this name in your output clue).
- target_entity_type / target_entity_identity: what the target is and which
  specific real-world sense of its name is meant (private context - never
  state directly).
- supporting_sentence: evidence text that may describe MULTIPLE facts or
  entities, not just the connection between source_entity and target_entity.
- relation: a short backup label for the connection type, only if the
  supporting_sentence alone leaves the connection type unclear.
- downstream_clue: if given, how the NEXT entity in the chain (a DIFFERENT
  entity, possibly a different type) is already described.

Do this in two steps internally:
1. Using target_entity's name to locate it precisely within
   supporting_sentence, identify SPECIFICALLY why target_entity is
   connected to source_entity - the one particular fact that links them.
   Ignore any other facts or entities in supporting_sentence not part of
   this specific connection.
2. Write a SHORT clue (roughly one sentence) based ONLY on that isolated
   connecting fact.

IMPORTANT - grammar must match THIS hop's own target_entity_type, not
downstream_clue's type: downstream_clue describes a DIFFERENT entity that
may have a different type than your own target. Never borrow pronouns
(he/she/they) or phrasing style from downstream_clue - use only what is
grammatically correct for YOUR OWN target_entity_type. Connect into
downstream_clue's CONTENT (what it says), never its GRAMMAR (how it's
phrased).

The clue must:
- Never name the target entity.
- Be grammatically and semantically correct for its OWN type only.
- Be a generalized, paraphrased version of the isolated fact - never quote
  supporting_sentence directly, never state its specific extracted details
  (exact dates, numbers, names).
- Stay SHORT and appropriately vague while remaining true to the isolated
  connecting fact.

Return ONLY JSON:
{"clue": "<one short natural-language clue phrase>"}
"""


def _clue_contains_entity_name(clue: str, entity_name: str) -> bool:
    """Checks whether the target entity's actual name leaked into the clue
    text - the prompt already instructs 'never name the target entity', but
    we've seen local-model instruction-following fail elsewhere in this
    pipeline, so this verifies it programmatically instead of trusting it."""
    if not clue or not entity_name:
        return False
    return entity_name.strip().lower() in clue.strip().lower()


def _build_clue(source_ref: str, hop: dict, downstream_clue: str = None, max_retries: int = 2) -> str:
    prompt = (
        f"Source entity (already established): {source_ref}\n"
        f"Target entity (PRIVATE - locate it in the evidence, never state this name): {hop.get('next_entity')}\n"
        f"Target entity type: {hop.get('next_entity_type')}\n"
        f"Target entity identity (private): {hop.get('next_entity_identity')}\n"
        f"Supporting sentence (may contain multiple facts - isolate the specific source->target connection first): {hop.get('supporting_sentence')}\n"
        f"Relation (secondary backup label only): {hop.get('relation')}\n"
    )
    if downstream_clue:
        prompt += f"Downstream clue (how the NEXT entity is already described - a DIFFERENT entity, do not borrow its grammar/type): {downstream_clue}\n"

    target_name = hop.get("next_entity", "")

    for attempt in range(max_retries + 1):
        try:
            out = call_local_llm(CLUE_SYSTEM_PROMPT, prompt, sample=True)
        except Exception:
            return ""

        clue = out.get("clue", "")
        if not clue:
            continue

        if _clue_contains_entity_name(clue, target_name):
            # NEW: the target's actual name leaked into the clue - retry the
            # same hop instead of accepting a clue that defeats the whole
            # point of not naming intermediate entities
            continue

        return clue

    return ""  # exhausted retries - still leaking the name every time, give up on this hop

def _build_clues_for_chain(entities: list, hops: list) -> list:
    n_hops = len(hops)
    clues = [None] * n_hops
    downstream_clue = None

    for i in range(n_hops - 1, -1, -1):
        source_ref = entities[0] if i == 0 else (
            f"{entities[i]} ({hops[i - 1].get('next_entity_identity', '')})"
        )
        hop = hops[i]

        clue = _build_clue(source_ref, hop, downstream_clue, max_retries=4)
        if not clue:
            return None

        clues[i] = clue
        downstream_clue = clue

    return clues


def run() -> None:
    chains = [c for c in load_json_list(INPUT_PATH) if c.get("filter", {}).get("verdict") == "pass"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    for chain in chains:
        if writer.is_done(chain["id"]):
            continue

        entities = chain["entities"]
        hops = chain["hops"]

        clues = _build_clues_for_chain(entities, hops)

        if clues is None:
            record = {**chain, "clue_status": "failed", "clues": None}
        else:
            record = {**chain, "clue_status": "built", "clues": clues}

        writer.append(record)
        print(f"[step4] {chain['id']}: {record['clue_status']}")


if __name__ == "__main__":
    run()