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

{terminal_uniqueness_block}

IMPORTANT - prefer LESS FAMOUS candidates obscure: when the evidence supports more
than one possible next entity, prefer the one a well-informed person would
be LESS likely to already know, over a globally famous "household name" -
this makes the resulting question genuinely hard to find rather than
trivially guessable. Also self-rate how famous/recognizable the next entity
you picked actually is.

IMPORTANT - entity identity: many entity names are ambiguous (e.g. "Manchester"
could mean the city OR the football club). Alongside next_entity, provide
next_entity_identity: a short phrase disambiguating EXACTLY which sense of
that name your evidence is about (e.g. "the football club" vs "the city in
England"). This identity will be carried forward so the next hop searches
for the correct sense, not a different entity that happens to share the name.

IMPORTANT - evidence grounding:
- next_entity must be explicitly present in the provided search evidence; never invent or infer an entity.
- The relationship between the source entity and next_entity must be explicitly stated in the evidence; never infer a relationship.
- supporting_sentence must be copied exactly from the source evidence; never fabricate, paraphrase, or reconstruct it.
- next_entity_type must be one of: person, organization, place, event, work.
- next_entity should not be same as the source entity
- next_entity should be stated in the suppporting_sentence.
- The supporting sentence must assert this relation for this one target, not
    merely mention the target inside a list, roster, group, count, or set of
    people/entities. If the sentence says that multiple people received the
    same thing, it does not identify one of them.
- The source entity must participate in the claimed relation as an explicit
    subject, object, possessor, organizer, employer, host, or other relational
    argument. A source name appearing only in a venue, location, title,
    modifier, or surrounding context is not a connection.

Return ONLY JSON:
{{
  "found": true,
  "relation": "<relation phrase, e.g. 'appointed head coach of'>",
  "next_entity": "<name>",
  "next_entity_identity": "<short phrase disambiguating which sense of this name is meant>",
  "next_entity_type": "<person|organization|place|event|work>",
  "next_entity_salience": "low"|"medium"|"high",
  "next_entity_unique": true|false,
    "competing_entities": ["<other entity in the sentence that could satisfy the same relation>"],
  "source_url": "<url>",
  "supporting_sentence": "<exact sentence from the source that states the relation and next entity>",
  "confidence": "explicit_statement"
}}
If nothing usable is found for this relation type, return:
{{"found": false, "reason": "<why>"}}
"""

EVIDENCE_UNIQUENESS_PROMPT = """You are a strict evidence adjudicator.

Decide three separate questions using only the supplied text:

1. relation_supported: Does the evidence explicitly state the claimed
    relation between the exact source entity and exact target entity? Do not
    replace the source with an associated person, event, venue, or institution.
2. target_explicit: Does the evidence explicitly identify the proposed target,
    rather than merely mentioning it in surrounding context?
3. target_unique: Does the evidence identify that target as the one entity
    satisfying the relation, rather than one member of a list, roster, group,
    count, or set of candidates?
4. source_participates: Is the source an actual argument of that relation?
    Reject cases where the source appears only as a venue, location, title,
    modifier, or contextual reference. For example, a school named in "a
    school football game" is not necessarily the entity that recognized an
    award.

All four answers must be true for the hop to be accepted.

A target is NOT unique when it is one name in a list, when the text says that
multiple people/entities received or did the same thing, or when an ellipsis
indicates omitted names. For example, "29 principals received renewed
contracts" followed by "Steve Kappenman" does not uniquely identify Steve.

Return ONLY JSON:
{
    "relation_supported": true|false,
    "target_explicit": true|false,
    "target_unique": true|false,
    "source_participates": true|false,
    "reason": "short explanation grounded in the evidence"
}
"""

TERMINAL_UNIQUENESS_BLOCK = """
IMPORTANT - this is the FINAL hop, and next_entity will become the answer.
The supporting sentence must identify exactly one entity satisfying the stated
relation from the source entity. A sentence that names a group, list, lineup,
multiple winners, multiple honorees, multiple performers, or several entities
that could satisfy the same relation is NOT unique, even if next_entity is one
of those names. For example, a sentence naming three people honored at one
event cannot uniquely identify any one of those people.

Before returning found=true, enumerate every other entity in the supporting
sentence that could satisfy the same source-to-target relation. Put those
competitors in competing_entities, excluding the source entity and
next_entity. Set next_entity_unique to true ONLY when competing_entities is an
empty list and the sentence itself supports one unambiguous target. If unsure,
return found=false instead of guessing.
"""


def _terminal_hop_is_unique(hop: dict) -> bool:
    if hop.get("next_entity_unique") is not True:
        return False
    if hop.get("competing_entities") != []:
        return False
    sentence = (hop.get("supporting_sentence") or "").strip()
    target = (hop.get("next_entity") or "").strip()
    return bool(
        sentence
        and target
        and target.lower() in sentence.lower()
        and sentence[-1] in ".?!\u201d\u2019"
    )


def _verify_single_target(source_entity: str, hop: dict) -> tuple[bool, str]:
    prompt = (
        f"Source entity: {source_entity}\n"
        f"Relation claimed: {hop.get('relation')}\n"
        f"Target entity: {hop.get('next_entity')}\n"
        f"Supporting sentence: {hop.get('supporting_sentence')}"
    )
    try:
        verdict = call_local_llm(EVIDENCE_UNIQUENESS_PROMPT, prompt, sample=False)
    except Exception as exc:
        return False, f"evidence adjudication failed: {exc}"
    checks = (
        "relation_supported",
        "target_explicit",
        "target_unique",
        "source_participates",
    )
    if not all(verdict.get(check) is True for check in checks):
        return False, verdict.get("reason", "evidence did not prove the relation and target")
    return True, verdict.get("reason", "relation and target are explicitly supported")


def _explore_hop(
    entity: str,
    domain: str,
    entity_identity: str = None,
    is_terminal_hop: bool = False,
    relation_type: str = None,
) -> dict:
    relation_types = RELATION_TYPES_BY_DOMAIN.get(domain, RELATION_TYPES_BY_DOMAIN["Other"])
    rel_type = relation_type or random.choice(relation_types)
    search_relation = rel_type.replace("_", " ")

    entity_ref = f"{entity} ({entity_identity})" if entity_identity else entity

    results = search(
        f"What connection, role, relationship, or association involving "
        f"{entity_ref} can be found through {search_relation}?",
        start_date=START_DATE, end_date=END_DATE
    )
    usable_results = [r for r in results if "error" not in r]
    if not usable_results:
        results = search(
            f"What connection, role, relationship, or association involving "
            f"{entity_ref} can be found through {search_relation}?"
        )
        usable_results = [r for r in results if "error" not in r]
    if not usable_results:
        return {"found": False, "reason": f"no usable search results for relation type '{rel_type}'"}

    user_prompt = f"Source entity: {entity_ref}\n\nSearch results:\n" + "\n\n".join(
        f"[{i}] {r.get('title')} ({r.get('url')})\n{r.get('content','')[:2000]}"
        for i, r in enumerate(usable_results)
    )

    terminal_block = TERMINAL_UNIQUENESS_BLOCK if is_terminal_hop else ""

    try:
        out = call_local_llm(SYSTEM_PROMPT.format(relation_type=rel_type, terminal_uniqueness_block=terminal_block), user_prompt, sample=True)
    except Exception as e:
        out = {"found": False, "reason": str(e)}

    if out.get("found"):
        evidence_is_unique, evidence_reason = _verify_single_target(entity_ref, out)
        if not evidence_is_unique:
            return {"found": False, "reason": f"evidence rejected: {evidence_reason}"}
        out["evidence_unique"] = True
        out["evidence_grounded_v2"] = True

        candidate_salience = out.get("next_entity_salience", "medium")
        if candidate_salience == "high":
            return {"found": False, "reason": f"'{out['next_entity']}' rejected for high salience"}

        if is_terminal_hop and not _terminal_hop_is_unique(out):
            return {"found": False, "reason": f"'{out.get('next_entity')}' not supported as a unique final answer"}

        out["relation_type_used"] = rel_type
        return out

    return out


def _explore_hop_with_retries(
    entity: str,
    domain: str,
    entity_identity: str = None,
    is_terminal_hop: bool = False,
) -> dict:
    relation_types = RELATION_TYPES_BY_DOMAIN.get(domain, RELATION_TYPES_BY_DOMAIN["Other"])
    candidates = relation_types[:]
    random.shuffle(candidates)
    attempts = min(MAX_RELATION_ATTEMPTS_PER_HOP, len(candidates))
    reasons = []

    for rel_type in candidates[:attempts]:
        hop = _explore_hop(
            entity,
            domain,
            entity_identity=entity_identity,
            is_terminal_hop=is_terminal_hop,
            relation_type=rel_type,
        )
        if hop.get("found"):
            return hop
        if hop.get("reason"):
            reasons.append(f"{rel_type}: {hop['reason']}")

    return {
        "found": False,
        "reason": f"all {attempts} relation attempts failed; " + " | ".join(reasons),
    }


def _build_chain(entity_A: str, domain: str, entity_A_identity: str = None):
    entities = [entity_A]
    identities = [entity_A_identity]
    hops = [None] * HOP_COUNT
    hop_idx = 0
    backtracks = 0
    entity_types = ["person", "organization", "place", "event", "work"]

    while 0 <= hop_idx < HOP_COUNT:
        if backtracks > MAX_BACKTRACKS_PER_CHAIN:
            return None, f"exceeded backtrack budget ({MAX_BACKTRACKS_PER_CHAIN}) at hop {hop_idx + 1}"

        current_entity = entities[hop_idx]
        current_identity = identities[hop_idx]

        visited_entities = {e.strip().lower() for e in entities[:hop_idx + 1]}
        is_terminal_hop = hop_idx == HOP_COUNT - 1
        hop = _explore_hop_with_retries(
            current_entity,
            domain=domain,
            entity_identity=current_identity,
            is_terminal_hop=is_terminal_hop,
        )

        if not hop.get("found"):
            hop_idx -= 1
            if hop_idx < 0:
                return None, f"hop1 exhausted with no further backtrack possible: {hop.get('reason')}"
            entities = entities[:hop_idx + 1]
            identities = identities[:hop_idx + 1]
            backtracks += 1
            continue

        if hop.get("next_entity", "").strip().lower() in visited_entities:
            hop_idx -= 1
            if hop_idx < 0:
                return None, f"hop1 exhausted with no further backtrack possible: duplicate next_entity '{hop.get('next_entity')}'"
            entities = entities[:hop_idx + 1]
            identities = identities[:hop_idx + 1]
            backtracks += 1
            continue

        if hop.get("next_entity_type", "").strip().lower() not in entity_types:
            hop_idx -= 1
            if hop_idx < 0:
                return None, f"hop1 exhausted with no further backtrack possible: invalid next_entity_type '{hop.get('next_entity_type')}'"
            entities = entities[:hop_idx + 1]
            identities = identities[:hop_idx + 1]
            backtracks += 1
            continue
        
        if hop.get("next_entity_identity", "")=="":
            hop_idx -= 1
            if hop_idx < 0:
                return None, f"hop1 exhausted with no further backtrack possible: missing next_entity_identity for '{hop.get('next_entity')}'"
            entities = entities[:hop_idx + 1]
            identities = identities[:hop_idx + 1]
            backtracks += 1
            continue

        next_entity = hop["next_entity"]
        next_identity = hop.get("next_entity_identity")
        hops[hop_idx] = hop
        entities = entities[:hop_idx + 1] + [next_entity]
        identities = identities[:hop_idx + 1] + [next_identity]
        hop_idx += 1

    return (entities, hops), None


def run() -> None:
    seeds = [s for s in load_json_list(INPUT_PATH) if s.get("status") == "ok"]
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    for seed in seeds:
        chain_id = seed["id"].replace("seed_", "chain_")
        existing = next((r for r in writer.all() if r.get("id") == chain_id), None)
        existing_terminal = (existing or {}).get("hops", [{}])[-1]
        if (
            existing
            and existing.get("status") == "chain_complete"
            and all(
                h.get("evidence_unique") is True
                and h.get("evidence_grounded_v2") is True
                for h in existing.get("hops", [])
            )
            and _terminal_hop_is_unique(existing_terminal)
        ):
            continue

        entity_A = seed["entity_A"]
        domain = seed.get("domain", "Other")
        entity_A_identity = seed.get("entity_identity")
        result, fail_reason = _build_chain(entity_A=entity_A, domain=domain, entity_A_identity=entity_A_identity)

        if result is not None:
            entities, hops = result
            record = {
                "id": chain_id, "seed_id": seed["id"],
                "entity_A": entity_A, "entity_A_attributes": seed.get("known_attributes", {}),
                "entity_A_identity": entity_A_identity, "entity_A_type": seed.get("entity_A_type"),
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