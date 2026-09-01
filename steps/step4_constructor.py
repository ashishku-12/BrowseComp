"""
Step 4 — Question Constructor Agent

Constructs a natural multihop research question from the chain.

The last entity in the chain is always the final answer.

The question should use the chain of relationships as clues and guide the
solver from the first entity toward the final entity.

Output: data/step4_questions.json
"""
import os
from config import DATA_DIR, DEEPSEEK_MODEL_STRONG
from utils.io_utils import ResumableWriter, load_json_list
from utils.llm_client import call_llm


INPUT_PATH = os.path.join(DATA_DIR, "step3_filtered.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step4_questions.json")


SYSTEM_PROMPT = """You are a Question Constructor Agent for a difficult multihop
web-research question dataset.

You are given a chain of entities:

Entity 1 -> Entity 2 -> ... -> Final Entity

The LAST entity is always the answer.

Your task is to construct ONE natural-language research question using the
relationships between the entities as a chain of clues.

Rules:

1. The question should begin with clues based on Entity 1 and its known
   attributes rather than simply presenting the chain mechanically.

2. Use the relationship between Entity 1 and Entity 2 as the first reasoning
   step.

3. Continue using each subsequent relationship as another clue that guides
   the solver through the chain.

4. The LAST entity must be the answer to the question.

5. Do not mention the final answer directly in the question.

6. Make the question creative and natural. It should feel like a research
   puzzle rather than a list of graph relations.

7. Do not simply copy relation labels. Convert the relationships and
   supporting facts into natural-language clues.

8. The question should require following the intended chain to discover
   the final entity.

Return ONLY JSON:

{
  "question": "<final multihop research question>",
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
            f"Entity 1: {entities[0]}",
            f"Entity 1 attributes: {chain.get('entity_A_attributes', {})}",
        ]

        for i, hop in enumerate(hops):
            chain_lines.append(
                f"\nHop {i + 1}:"
            )
            chain_lines.append(
                f"From entity: {entities[i]}"
            )
            chain_lines.append(
                f"Relation: {hop['relation']}"
            )
            chain_lines.append(
                f"To entity: {entities[i + 1]}"
            )
            chain_lines.append(
                f"Supporting fact: {hop.get('supporting_sentence', '')}"
            )

        chain_lines.append(
            f"\nFinal answer: {entities[-1]}"
        )

        user_prompt = "\n".join(chain_lines)

        try:
            out = call_llm(
                SYSTEM_PROMPT,
                user_prompt,
                model=DEEPSEEK_MODEL_STRONG,
                use_secondary=True,
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