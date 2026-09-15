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
    DATA_DIR, HOP_COUNT,
    MAX_RELATION_ATTEMPTS_PER_HOP, MAX_BACKTRACKS_PER_CHAIN, DIVERSITY_HINT_SAMPLE_SIZE, START_DATE, END_DATE
)
from utils.io_utils import ResumableWriter, load_json_list
from utils.local_llm_client import call_local_llm
from utils.search_client import search
import random

INPUT_PATH = os.path.join(DATA_DIR, "step1_seeds.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "step2_chains.json")

RELATION_TYPES_BY_DOMAIN = {
    "Politics": [
        "specific_appointment_with_date",       
        "party_membership_with_date",           
        "committee_or_institution_role",       
        "specific_election_result",             
        "named_bill_or_policy_sponsorship",     
        "named_treaty_or_agreement",            
    ],

    "Geography": [
        "site_specific_incident",               
        "named_geographical_feature",           
        "administrative_boundary_change",      
        "specific_border_dispute_or_treaty",    
        "named_expedition_or_survey",        
        "documented_naming_event",       
    ],

    "Video Games": [
        "specific_title_development_credit",  
        "named_creator_credit",                 
        "specific_franchise_entry",             
        "dated_platform_release",               
        "named_character_appearance",           
        "specific_tournament_or_convention",    
    ],

    "Music": [
        "named_group_membership_with_dates",    
        "specific_album_or_track_credit",       
        "named_session_or_production_credit",   
        "credited_feature_or_session_work",     
        "specific_label_signing_with_date",     
        "named_dated_performance",              
    ],

    "Sports": [
        "specific_team_tenure_with_dates",       
        "specific_coaching_tenure_with_dates",   
        "named_competition_result",              
        "specific_dated_achievement",            
        "specific_organizational_role",          
        "named_event_participation_with_date",   
    ],

    "History": [
        "named_participant_role",               
        "specific_command_or_office",           
        "site_specific_historical_event",       
        "named_campaign_or_battle",             
        "specific_institutional_role",          
        "documented_direct_consequence",        
    ],

    "Art": [
        "specific_artwork_attribution",         
        "named_movement_affiliation_with_dates",
        "specific_acquisition_or_collection_entry", 
        "named_commission_with_date",           
        "specific_named_exhibition",            
        "documented_direct_influence",          
    ],

    "Science & Technology": [
        "specific_discovery_credit",            
        "specific_publication_credit",          
        "dated_institutional_affiliation",      
        "named_project_role",                   
        "specific_named_conference_presentation",
        "named_coauthorship_or_joint_project",  
    ],

    "TV Shows & Movies": [
        "specific_named_role_credit",           
        "specific_title_directing_credit",      
        "specific_title_writing_credit",       
        "specific_franchise_installment",       
        "dated_release_or_distribution_deal",   
        "specific_named_award_or_festival",     
    ],

    "Other": [
        "documented_specific_incident",         
        "named_joint_credit_or_appearance",
        "dated_formal_agreement",
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
- next_entity_type must be one of: person, organization, place, event, work.
- next_entity should not be same as the source entity

Return ONLY JSON:
{{
  "found": true,
  "relation": "<relation phrase, e.g. 'appointed head coach of'>",
  "next_entity": "<name>",
  "next_entity_type": "<person|organization|place|event|work>",
  "next_entity_salience": "low"|"medium"|"high",
  "source_url": "<url>",
  "supporting_sentence": "<exact sentence from the source that states the relation and next entity>",
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
        out = call_local_llm(
            SYSTEM_PROMPT.format(relation_type=rel_type),
            user_prompt,
            sample=True
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

    return out  


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
        hop = _explore_hop(current_entity, domain=domain)

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