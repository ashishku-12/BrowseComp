"""
Step 5 — Question Constructor Agent

Constructs a natural multihop research question ENTIRELY from clues - no
entity is named directly anywhere in this step, including Entity 1.

CHANGE: Entity 1 was previously handed to the assembler as a bare name +
attributes, the only entity in the whole chain treated this way. Now Entity
1 gets its own clue, built from its known attributes (the same obscuring
discipline every other hop already goes through) - the assembler receives
ONLY clue text, entity types, and NOTHING that names any entity, including
the final answer (removed in the previous fix). This makes the whole chain
structurally uniform: N+1 entities, N+1 clues (Entity 1's attribute-clue +
one clue per hop), zero names anywhere in what the model sees.

Output: data/step5_questions.json
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm


INPUT_PATH = os.path.join(DATA_DIR, "step4_clues.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step5_questions.json")


ENTITY_A_CLUE_SYSTEM_PROMPT = """You write ONE short clue phrase identifying
a starting entity for a research puzzle, using ONLY its known attributes -
never its name.

You are given:
- entity_type: what kind of thing it is (person/organization/place/event/work)
- known_attributes: a set of facts about this entity (occupation/role,
  nationality or location, active time period, domain, etc.)

Write a SHORT clue (roughly one sentence) that:
- Never names the entity.
- Uses ONLY the given attributes - do not invent any detail not present in
  known_attributes.
- Is grammatically and semantically correct for entity_type.
- Is specific enough, combining the given attributes together, that a
  solver doing real research could identify this one entity - not so vague
  it could be almost anyone/anything of that type.

Return ONLY JSON:
{"clue": "<one short natural-language clue phrase>"}
"""


ASSEMBLER_SYSTEM_PROMPT = """You are a Question Assembler for a difficult multihop
web-research question dataset.

You are given a sequence of ALREADY-WRITTEN clues, in order - the FIRST clue
describes the starting entity, each following clue describes the next
entity in the chain, and the LAST clue describes the Final Entity (the
answer). NO entity is named anywhere in what you are given, including the
final answer - only clue text and each entity's type.

Your ONLY job is to stitch these clues into ONE single, coherent, natural-
reading research question, in forward order. You must NOT:
- Introduce any new fact, detail, date, or description not already present
  in the given clues.
- Change what any clue means or add specificity to it.
- Name ANY entity in the chain, including the starting entity or the final
  answer - none of them are named in your input, and none should appear
  named in your output.
- Describe the final answer so specifically that it becomes effectively
  obvious without being named - that defeats the purpose the same as
  naming it outright.
- Produce MORE THAN ONE question. The clues describe a single chain leading
  to ONE final answer - do not ask separate questions about intermediate
  entities in the chain.

You MUST:
- Match the question's final interrogative wording to final_entity_type:
  use "Who" ONLY if final_entity_type is "person" - for organization,
  place, event, or work, use "What" or "Which" instead, never "who".

You MAY:
- Adjust connecting words/phrasing so the clues read as one smooth,
  natural research puzzle instead of a mechanical list.
- Reorder minor phrasing within a clue for readability, without altering
  its content or specificity.

Return ONLY JSON:

{
  "question": "<ONE final multihop research question, presented forward>"
}
"""


def _clue_contains_entity_name(clue: str, entity_name: str) -> bool:
    if not clue or not entity_name:
        return False
    return entity_name.strip().lower() in clue.strip().lower()


def _build_entity_A_clue(entity_A: str, entity_A_type: str, entity_A_attributes: dict, max_retries: int = 4) -> str:
    prompt = (
        f"entity_type: {entity_A_type}\n"
        f"known_attributes: {entity_A_attributes}\n"
    )

    for attempt in range(max_retries + 1):
        try:
            out = call_local_llm(ENTITY_A_CLUE_SYSTEM_PROMPT, prompt, sample=True)
        except Exception:
            return ""

        clue = out.get("clue", "")
        if not clue:
            continue
        if _clue_contains_entity_name(clue, entity_A):
            continue  # name leaked despite instruction - retry, same as Step 4's clue check

        return clue

    return ""


def run() -> None:
    chains = [
        c for c in load_json_list(INPUT_PATH)
        if c.get("clue_status") == "built"
    ]

    writer = ResumableWriter(OUTPUT_PATH, key="id")

    for chain in chains:
        if writer.is_done(chain["id"]):
            continue

        entities = chain["entities"]
        hop_clues = chain["clues"]
        hops = chain["hops"]

        entity_A = entities[0]
        entity_A_type = chain.get("entity_A_type", "unknown")
        entity_A_attributes = chain.get("entity_A_attributes", {})
        final_entity_type = hops[-1].get("next_entity_type", "unknown") if hops else "unknown"

        entity_A_clue = _build_entity_A_clue(entity_A, entity_A_type, entity_A_attributes)

        if not entity_A_clue:
            record = {**chain, "construction": None, "construction_status": "failed",
                      "construction_error": "could not build a name-free clue for entity 1"}
            writer.append(record)
            print(f"[step5] {chain['id']}: failed (entity 1 clue)")
            continue

        # Full clue sequence: entity 1's own clue first, then every hop clue
        all_clues = [entity_A_clue] + hop_clues

        chain_lines = [f"final_entity_type: {final_entity_type}"]
        for i, clue in enumerate(all_clues):
            role = "starting entity" if i == 0 else ("final answer" if i == len(all_clues) - 1 else f"hop {i}")
            chain_lines.append(f"\nClue {i + 1} ({role}): {clue}")

        user_prompt = "\n".join(chain_lines)

        try:
            out = call_local_llm(ASSEMBLER_SYSTEM_PROMPT, user_prompt, sample=True)
            question_text = (out.get("question") or "").strip()

            # defensive check: neither entity 1's name nor the final answer's
            # name should appear anywhere in the assembled question
            answer_name = entities[-1].strip().lower()
            q_lower = question_text.lower()
            if not question_text or answer_name in q_lower or entity_A.strip().lower() in q_lower:
                out = {"question": None, "error": "name leak or empty question detected"}
                status = "failed"
            else:
                out["question"] = question_text
                out["canonical_answer"] = entities[-1]
                out["obfuscation_map"] = {"first_entity": entity_A_clue}
                status = "constructed"

        except Exception as e:
            out = {
                "question": None,
                "canonical_answer": None,
                "obfuscation_map": {"first_entity": entity_A_clue},
                "error": str(e),
            }
            status = "failed"

        record = {
            **chain,
            "construction": out,
            "construction_status": status
        }

        writer.append(record)
        print(f"[step5] {chain['id']}: {status}")


if __name__ == "__main__":
    run()