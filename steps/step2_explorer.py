"""
Step 2 — Chain Explorer Agent
Finds a chain of HOP_COUNT atomic, sourced relations starting from entity A:
A -> B -> C -> ... (HOP_COUNT hops, HOP_COUNT + 1 entities total).

Diversity fixes (entity + hop dedup, real DFS backtracking) - see previous
version's docstring for details, unchanged here.

NEW: source-URL independence fix. Step 3 was correctly rejecting chains
where two different hops pulled their evidence from the SAME source page -
that's a genuine hop-independence violation (a single search could partly
solve the question). That was never checked at generation time, so Step 2
could - and did - happily build a chain that Step 3 would always reject.
Fixed by tracking every source_url already used earlier in the SAME chain
and rejecting any candidate hop that reuses one, at generation time, before
the chain is ever assembled - same exclude-and-retry pattern as entity dedup.
Output: data/step2_chains.json
"""
import os
from config import (
    DATA_DIR, DEEPSEEK_MODEL_STRONG, HOP_COUNT,
    MAX_RELATION_ATTEMPTS_PER_HOP, MAX_BACKTRACKS_PER_CHAIN, DIVERSITY_HINT_SAMPLE_SIZE, RECENCY_START_DATE
)
from utils.io_utils import ResumableWriter, load_json_list
from utils.llm_client import call_llm
from utils.search_client import search

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

Return ONLY JSON:
{{
  "found": true,
  "relation": "<short relation phrase, e.g. 'appointed head coach of'>",
  "next_entity": "<name>",
  "next_entity_type": "<person|organization|place|event|work>",
  "source_url": "<url>",
  "supporting_sentence": "<exact sentence from the source that states the relation>",
  "confidence": "explicit_statement"
}}
If nothing usable is found for this relation type, return:
{{"found": false, "reason": "<why>"}}
"""


def _explore_hop(entity: str, domain: str, exclude: set = None, exclude_source_urls: set = None) -> dict:
    """
    Try up to MAX_RELATION_ATTEMPTS_PER_HOP relation types before declaring
    dead end.
    `exclude` - normalized next_entity names already tried/rejected at THIS
    hop position (dedup + real backtracking, from the previous fix).
    `exclude_source_urls` - source URLs already used by EARLIER hops in this
    SAME chain - a candidate reusing one is rejected, since two hops sharing
    a source is exactly the "hop independence violated" failure Step 3 flags.
    """
    exclude = exclude or set()
    exclude_source_urls = exclude_source_urls or set()
    exclude_hint = list(exclude)[:DIVERSITY_HINT_SAMPLE_SIZE]  # fixed size, never grows prompt

    relation_types = RELATION_TYPES_BY_DOMAIN.get(domain, RELATION_TYPES_BY_DOMAIN["Other"])

    for attempt, rel_type in enumerate(relation_types[:MAX_RELATION_ATTEMPTS_PER_HOP]):
        search_relation = rel_type.replace("_", " ")
        results = search(f"{entity} {search_relation} relation OR role OR connection", start_date=RECENCY_START_DATE)
        usable_results = [r for r in results if "error" not in r]
        if not usable_results:
            results = search(f"{entity} {search_relation} relation OR role OR connection")
            usable_results = [r for r in results if "error" not in r]
        if not usable_results:
            continue
        user_prompt = f"Source entity: {entity}\n\nSearch results:\n" + "\n\n".join(
            f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content','')[:500]}"
            for i, r in enumerate(results) if "error" not in r
        )
        if exclude_hint:
            user_prompt += (
                "\n\nDo NOT propose any of these as the next entity - already "
                f"tried at this step: {', '.join(exclude_hint)}. Pick a different one."
            )
        if exclude_source_urls:
            user_prompt += (
                "\n\nDo NOT use any of these source URLs - already used earlier "
                f"in this chain, so reusing one breaks source independence: "
                f"{', '.join(list(exclude_source_urls)[:DIVERSITY_HINT_SAMPLE_SIZE])}. "
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
            if candidate_norm in exclude:
                continue  # duplicate entity - reject, try next relation type
            if candidate_url and candidate_url in exclude_source_urls:
                continue  # reused source from an earlier hop - reject, try next relation type
            out["relation_type_used"] = rel_type
            out["attempts_used"] = attempt + 1
            return out
    return {"found": False, "reason": "exhausted relation types (including exclusions)",
            "attempts_used": MAX_RELATION_ATTEMPTS_PER_HOP}


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

        hop = _explore_hop(current_entity, domain=domain, exclude=excludes[hop_idx], exclude_source_urls=used_source_urls)

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