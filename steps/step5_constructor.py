"""
Step 5 — Question Constructor Agent

Builds the question from clues. The starting entity (Entity A) is passed
directly by name into the assembler; every other entity in the chain,
including intermediate hops and the final answer, is never named.

FIX (this revision):
1. The LAST clue in the chain was previously being appended as just
   another fact, so the assembled question never actually asked the
   solver to identify the final answer - it stated something true about
   it and stopped. The assembler is now explicitly instructed to convert
   the LAST clue into the question's interrogative target ("...which/what/
   who is the entity that <last clue's relation>?"), not restate it as a
   flat sentence.
2. Leak-checking previously only verified entity_A's name was present and
   the final answer's name was absent. It never checked whether an
   INTERMEDIATE entity (any hop between A and the answer) leaked into the
   assembled question - even though Step 4 now guards against this at the
   clue level, the assembler is a second LLM call that can reintroduce a
   name while "smoothing" phrasing, so this is now checked independently
   here as well (defense in depth).
3. The assembler was allowed to "adjust phrasing," which in practice let
   it quietly generalize a specific clue into a vaguer one while smoothing
   transitions (the same failure mode Step 4 had with over-vague
   obscuring). The prompt now explicitly forbids loosening/generalizing
   any clue's specificity during assembly.

Output: data/step5_questions.json
"""
import os
from config import DATA_DIR
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm


INPUT_PATH = os.path.join(DATA_DIR, "step4_clues.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step5_questions.json")


ASSEMBLER_SYSTEM_PROMPT = """You are a Question Assembler for a difficult multihop
web-research question dataset.

You are given a sequence of ALREADY-WRITTEN clues, in order, each with its
own entity type - the FIRST clue NAMES the starting entity directly, each
following clue describes the next entity in the chain without naming it,
and the LAST clue describes the Final Entity (the answer) without naming
it. Only the starting entity is named in what you are given; no other
entity - not an intermediate hop, not the final answer - is named, and
none but the starting entity should appear named in your output.

Your job has TWO parts:

PART 1 - CHAIN THE CLUES (all clues except the last):
Stitch these clues into one coherent forward-reading sequence, connecting
each entity to the next via its clue ("starting from [Entity A, named],
identify the [type] that [clue 1], then the [type] that [clue 2], ...").

PART 2 - TURN THE LAST CLUE INTO THE QUESTION:
The LAST clue describes the Final Entity - this is the answer the solver
must produce. Do NOT append the last clue as a flat statement of fact.
Instead, rephrase it as the actual interrogative the solver must answer,
using the connective built up from the earlier clues as the subject
("...what/which/who is the [final entity type] that <last clue's
relation, rephrased as what is being asked for>?"). The question must end
by clearly asking the solver to name/identify this final entity - if a
correct solver could answer your question without ever stating what the
final entity is, the question is wrong.

You must NOT:
- Introduce any new fact, detail, date, or description not already present
  in the given clues.
- Change what any clue means, loosen it, or make it more generic/vaguer
  than it was given to you - preserve each clue's exact level of
  specificity, only rephrase its grammar to fit the sentence.
- Add specificity to any clue that isn't already there either.
- Name ANY entity in the chain OTHER than the starting entity - in
  particular, never name any intermediate entity or the final answer.
- Describe the final answer so specifically that it becomes effectively
  obvious without being named.
- Produce MORE THAN ONE question - the clues describe a single chain
  leading to ONE final answer, do not ask separate questions about
  intermediate entities.

You MUST:
- Keep the starting entity's given name exactly as provided.
- Match the question's final interrogative wording to the LAST clue's type:
  use "Who" ONLY if that type is "person" - for organization, place, event,
  or work, use "What" or "Which" instead, never "who".
- Keep each clue's own phrasing consistent with ITS OWN stated type when
  smoothing transitions - never let pronouns or phrasing meant for one
  entity's type bleed into how a different-typed clue is connected.

You MAY:
- Adjust connecting words/phrasing so the clues read as one smooth,
  natural research puzzle instead of a mechanical list.
- Reorder minor phrasing within a clue for readability, without altering
  its content, meaning, or specificity.

Return ONLY JSON:
{"question": "<ONE final multihop research question, presented forward, ending in the interrogative that asks the solver to identify the final entity>"}
"""


def _text_leaks_entity(text: str, entity_name: str) -> bool:
    """Programmatic check for an entity name leaking into the assembled
    question - used for the final answer AND every intermediate entity,
    since the assembler is a second LLM call that can reintroduce a name
    Step 4 had already scrubbed."""
    if not text or not entity_name:
        return False
    return entity_name.strip().lower() in text.strip().lower()


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
        final_entity_type = hops[-1].get("next_entity_type", "unknown") if hops else "unknown"

        # Entity A is passed directly (named) - no name-free clue needed.
        all_clues = [entity_A] + hop_clues
        all_clue_types = [entity_A_type] + [h.get("next_entity_type", "unknown") for h in hops]

        chain_lines = []
        for i, clue in enumerate(all_clues):
            role = "starting entity (named)" if i == 0 else ("final answer - must become the question" if i == len(all_clues) - 1 else f"hop {i}")
            chain_lines.append(f"Clue {i + 1} ({role}, type: {all_clue_types[i]}): {clue}")

        user_prompt = "\n".join(chain_lines)

        try:
            out = call_local_llm(ASSEMBLER_SYSTEM_PROMPT, user_prompt, sample=True)
            question_text = (out.get("question") or "").strip()
            q_lower = question_text.lower()

            answer_name = entities[-1].strip().lower()
            # every entity strictly between A and the answer must not appear
            intermediate_names = entities[1:-1]

            leaked = (
                not question_text
                or (answer_name and answer_name in q_lower)
                or any(_text_leaks_entity(question_text, name) for name in intermediate_names)
                or (entity_A and entity_A.strip().lower() not in q_lower)
                or "?" not in question_text  # must actually be a question
            )

            if leaked:
                out = {"question": None, "error": "name leak, missing entity A, missing '?', or empty question detected"}
                status = "failed"
            else:
                out["question"] = question_text
                out["canonical_answer"] = entities[-1]
                out["obfuscation_map"] = {"first_entity": entity_A}
                status = "constructed"

        except Exception as e:
            out = {
                "question": None,
                "canonical_answer": None,
                "obfuscation_map": {"first_entity": entity_A},
                "error": str(e),
            }
            status = "failed"

        record = {
            **chain,
            "construction": out,
            "construction_status": status,
        }

        writer.append(record)
        print(f"[step5] {chain['id']}: {status}")


if __name__ == "__main__":
    run()
