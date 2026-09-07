"""
Step 2 — Chain Explorer Agent

NEW: salience-aware hop selection. Step 3 was catching too-famous bridge
entities only AFTER a whole chain was built, wasting generation effort on
chains that were doomed from one bad hop. Research on multihop difficulty
(semantic distance between evidence is a stronger difficulty predictor than
hop count itself) points to fixing THIS, not just adding more hops. Fixed by
asking the model to self-rate each candidate's fame/salience as part of its
JSON output, and rejecting "high" salience candidates at generation time -
same exclude-and-retry mechanism already used for duplicates and source reuse.
"""
import os
from config import (
    DATA_DIR, DEEPSEEK_MODEL_STRONG, HOP_COUNT,
    MAX_RELATION_ATTEMPTS_PER_HOP, MAX_BACKTRACKS_PER_CHAIN, DIVERSITY_HINT_SAMPLE_SIZE, START_DATE, END_DATE
)
from utils.io_utils import ResumableWriter, load_json_list
from utils.llm_client import call_llm
from utils.search_client import search
import random

INPUT_PATH = os.path.join(DATA_DIR, "step1_seeds.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step2_chains.json")

RELATION_TYPES_BY_DOMAIN = {
    "Politics": [
        "leadership",
        "party_affiliation",
        "government_institution",
        "election",
        "legislation_policy",
        "diplomacy_treaty",
    ],

    "Geography": [
        "location",
        "geographical_feature",
        "administrative_region",
        "border_neighbor",
        "exploration",
        "naming_origin",
    ],

    "Video Games": [
        "developer_publisher",
        "creator",
        "franchise",
        "platform_release",
        "character_universe",
        "game_event",
    ],

    "Music": [
        "artist_group",
        "album_work",
        "recording_production",
        "collaboration",
        "label_release",
        "performance_event",
    ],

    "Sports": [
        "player_team",
        "coach_management",
        "competition",
        "achievement",
        "sports_organization",
        "event_participation",
    ],

    "History": [
        "participants",
        "leadership",
        "location",
        "conflict_campaign",
        "political_institution",
        "historical_consequence",
    ],

    "Art": [
        "artist_artwork",
        "art_movement",
        "museum_collection",
        "commission_patron",
        "exhibition",
        "artistic_influence",
    ],

    "Science & Technology": [
        "researcher_discovery",
        "author_work",
        "institution_affiliation",
        "project_product",
        "conference_event",
        "scientific_collaboration",
    ],

    "TV Shows & Movies": [
        "cast",
        "director_creator",
        "writer_producer",
        "series_franchise",
        "release_distribution",
        "award_festival",
    ],

    "Other": [
        "person_relationship",
        "organization_relationship",
        "location_relationship",
        "event_relationship",
        "work_relationship",
        "historical_relationship",
    ],
}

SYSTEM_PROMPT = """You are a Chain Explorer Agent. Given a source entity and search
results, find exactly ONE specific, atomic, explicitly-stated relation connecting
the source entity to a DIFFERENT, further-explorable next entity.

Hard requirements:
- The relation must be explicitly stated in a source sentence, not inferred.
- The next entity must be concrete and specific (not an abstract concept),
  and itself likely to have further documented relations.
- Prefer a relation type of: {relation_type}, but only if the evidence supports it.

IMPORTANT - prefer LESS FAMOUS candidates: when the evidence supports more
than one possible next entity, prefer the one a well-informed person would
be LESS likely to already know, over a globally famous "household name" -
this makes the resulting question genuinely hard to find rather than
trivially guessable. Also self-rate how famous/recognizable the next entity
you picked actually is.

IMPORTANT - evidence grounding:
- next_entity must be explicitly present in the provided search evidence; never invent or infer an entity.
- The relationship between the source entity and next_entity must be explicitly stated in the evidence; never infer a relationship.
- supporting_sentence must be copied exactly from the source evidence; never fabricate, paraphrase, or reconstruct it.


Return ONLY JSON:
{{
  "found": true,
  "relation": "<short relation phrase, e.g. 'appointed head coach of'>",
  "next_entity": "<name>",
  "next_entity_type": "<>",
  "next_entity_salience": "low"|"medium"|"high",
  "source_url": "<url>",
  "supporting_sentence": "<exact sentence from the source that states the relation>",
  "confidence": "explicit_statement"
}}
If nothing usable is found for this relation type, return:
{{"found": false, "reason": "<why>"}}
"""


def _explore_hop(entity: str, domain: str, exclude: set = None, exclude_source_urls: set = None,
                  previous_relation: str = None) -> dict:
    exclude = exclude or set()
    exclude_source_urls = exclude_source_urls or set()
    exclude_hint = list(exclude)[:DIVERSITY_HINT_SAMPLE_SIZE]

    relation_types = RELATION_TYPES_BY_DOMAIN.get(domain, RELATION_TYPES_BY_DOMAIN["Other"])
    rel_type = random.choice(relation_types)  # ONE random category per call - no loop
    search_relation = rel_type.replace("_", " ")

    results = search(
        f"What connection, role, relationship, or association involving "
        f"{entity} can be found through {search_relation} in 2026?",
        start_date=START_DATE,
        end_date=END_DATE
    )

    usable_results = [r for r in results if "error" not in r]
    if not usable_results:
        results = search(
            f"What connection, role, relationship, or association involving "
            f"{entity} can be found through {search_relation} in 2026?"
        )
        usable_results = [r for r in results if "error" not in r]
    if not usable_results:
        return {"found": False, "reason": f"no usable search results for relation type '{rel_type}'"}

    user_prompt = f"Source entity: {entity}\n\nSearch results:\n" + "\n\n".join(
        f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content','')[:700]}"
        for i, r in enumerate(usable_results)
    )
    if previous_relation:
        user_prompt += (
            f"\n\nThe chain so far reached this entity via: \"{previous_relation}\". "
            "Where the evidence supports it, PREFER a next relation that continues "
            "this same theme or storyline, rather than an unrelated fact about this entity."
        )
    if exclude_hint:
        user_prompt += (
            "\n\nDo NOT propose any of these as the next entity - already "
            f"tried at this step: {', '.join(exclude_hint)}. Pick a different one."
        )
    if exclude_source_urls:
        user_prompt += (
            "\n\nDo NOT use any of these source URLs - already used earlier "
            f"in this chain: {', '.join(list(exclude_source_urls)[:DIVERSITY_HINT_SAMPLE_SIZE])}. "
            "Find a DIFFERENT source for this hop."
        )

    try:
        out = call_llm(
            SYSTEM_PROMPT.format(relation_type=rel_type),
            user_prompt,
            model=DEEPSEEK_MODEL_STRONG, use_secondary=True, sample=True,
        )
    except Exception as e:
        out = {"found": False, "reason": str(e)}

    if out.get("found"):
        candidate_norm = out["next_entity"].strip().lower()
        candidate_url = (out.get("source_url") or "").strip()
        candidate_salience = out.get("next_entity_salience", "medium")

        if candidate_norm in exclude:
            return {"found": False, "reason": f"'{out['next_entity']}' already tried/excluded"}
        if candidate_url and candidate_url in exclude_source_urls:
            return {"found": False, "reason": "source URL already used earlier in this chain"}
        if candidate_salience == "high":
            return {"found": False, "reason": f"'{out['next_entity']}' rejected for high salience"}

        out["relation_type_used"] = rel_type
        return out

    return out  # already shaped {"found": False, "reason": ...} from the model


def _build_chain(entity_A: str, domain: str):
    """
    True DFS with backtracking - one exclusion set per hop POSITION for
    entity names, PLUS a running set of source URLs used so far in the
    whole chain (recomputed fresh each time from the currently-accepted
    hops, so it automatically shrinks correctly on backtrack).
    Returns (entities, hops) on success, or (None, fail_reason) on failure.
    """
    excludes = [set() for _ in range(HOP_COUNT)]  # excludes[i] = rejected next_entity values at hop i
    entities = [entity_A]
    hops = [None] * HOP_COUNT
    hop_idx = 0
    backtracks = 0

    while 0 <= hop_idx < HOP_COUNT:
        if backtracks > MAX_BACKTRACKS_PER_CHAIN:
            return None, f"exceeded backtrack budget ({MAX_BACKTRACKS_PER_CHAIN}) at hop {hop_idx + 1}"

        current_entity = entities[hop_idx]
        # source URLs already locked in by earlier, currently-accepted hops in this chain
        used_source_urls = {h["source_url"] for h in hops[:hop_idx] if h and h.get("source_url")}
        whole_chain_entities = {e.strip().lower() for e in entities[:hop_idx + 1]}
        combined_exclude = excludes[hop_idx] | whole_chain_entities
        previous_relation = hops[hop_idx - 1]["relation"] if hop_idx > 0 and hops[hop_idx - 1] else None
        hop = _explore_hop(current_entity, domain=domain, exclude=combined_exclude, exclude_source_urls=used_source_urls, previous_relation=previous_relation)

        if not hop.get("found"):
            excludes[hop_idx] = set()  # reset in case we reach this depth again via a different earlier branch
            hop_idx -= 1
            if hop_idx < 0:
                return None, f"hop1 exhausted with no further backtrack possible: {hop.get('reason')}"
            failed_entity_norm = entities[hop_idx + 1].strip().lower()
            excludes[hop_idx].add(failed_entity_norm)
            entities = entities[:hop_idx + 1]
            backtracks += 1
            continue

        next_entity = hop["next_entity"]
        hops[hop_idx] = hop
        entities = entities[:hop_idx + 1] + [next_entity]
        hop_idx += 1

    return (entities, hops), None


def run() -> None:
    seeds = [s for s in load_json_list(INPUT_PATH) if s.get("status") == "ok"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    for seed in seeds:
        chain_id = seed["id"].replace("seed_", "chain_")
        if writer.is_done(chain_id):
            continue

        entity_A = seed["entity_A"]
        domain = seed.get("domain", "Other")
        result, fail_reason = _build_chain(entity_A=entity_A, domain=domain)

        if result is not None:
            entities, hops = result
            record = {
                "id": chain_id, "seed_id": seed["id"],
                "entity_A": entity_A, "entity_A_attributes": seed.get("known_attributes", {}),
                "entities": entities,
                "hops": hops,
                "entity_final": entities[-1],
                "status": "chain_complete",
            }
        else:
            record = {
                "id": chain_id, "seed_id": seed["id"], "entity_A": entity_A,
                "status": "dead_end", "fail_reason": fail_reason,
            }

        writer.append(record)
        print(f"[step2] {chain_id}: {record['status']}")


if __name__ == "__main__":
    run()