"""
Step 1 — Seed Agent
Picks entity A: real, verifiable, moderately-obscure (not over-exposed).
Output: data/step1_seeds.json

Diversity fix: previously, identical (domain, greedy-decoded) calls produced
identical seeds. Fixed with three layers, all domain-agnostic:
  1. Programmatic exact-dedup - a normalized-name set per domain, checked
     in Python, zero token cost, always enforced regardless of what the
     model does.
  2. A FIXED-size hint of recently-used entities in the prompt (bounded by
     DIVERSITY_HINT_SAMPLE_SIZE - never grows as the dataset grows).
  3. Sampled decoding (sample=True) so retries genuinely explore different
     candidates instead of regenerating the same "most likely" answer.
"""
import os
from config import DATA_DIR, DIVERSITY_HINT_SAMPLE_SIZE, MAX_DEDUP_RETRIES, START_DATE, END_DATE
from utils.io_utils import ResumableWriter
from utils.local_llm_client import call_local_llm
from utils.search_client import search

OUTPUT_PATH = os.path.join(DATA_DIR, "step1_seeds.json")

ENTITY_TYPE_ROTATION = ["person", "organization", "place", "event", "work"]

SYSTEM_PROMPT = """You are a Seed Agent for a multihop question-generation pipeline.
Given a domain/topic hint and a batch of web search results about that domain,
pick exactly ONE real, verifiable entity to use as a starting point (entity A)
for a multihop trivia chain.

Requirements for a good seed:
- Must be a real, documented entity with a clear source.
- Should be moderately obscure: has a dedicated profile/page and a few
  independent mentions, but is NOT a globally famous, "front page" entity.
  (Avoid picking things so famous a model could reason about them from memory alone.)
- Must have several distinguishing attributes (occupation/type, nationality/location,
  active time period, domain) that could later be used to describe it WITHOUT naming it.

Also provide entity_identity: a short phrase disambiguating EXACTLY which
real-world entity this name refers to, in case the name alone is ambiguous
(e.g. for "Manchester", specify "the football club" vs "the city in
England"; for a common person's name, specify their distinguishing role,
e.g. "the marine biologist", not just "person"). This must be specific
enough that a search using the name plus this identity would find the
correct entity, not a different one sharing the same name.

Return ONLY a JSON object with this exact shape:
{
  "entity_A": "<name>",
  "entity_A_type": "<person|organization|place|event|work>",
  "entity_identity": "<short phrase disambiguating which specific entity this name refers to>",
  "source_url": "<best source url from the provided results>",
  "known_attributes": {"...": "..."},
  "salience_note": "<why this is a good, moderately-obscure seed>",
  "status": "ok"
}
If none of the results are usable, return {"status": "failed", "reason": "..."}.
"""


def _parse_domain(domain_hint: str) -> tuple[str, str]:
    """
    Parse:
        'History > Military History'
    into:
        ('History', 'Military History')
    """
    if ">" in domain_hint:
        domain, subdomain = domain_hint.split(">", 1)
        return domain.strip(), subdomain.strip()

    return domain_hint.strip(), "General"


def _build_user_prompt(domain: str, results: list, exclude_hint: list, target_type: str) -> str:
    lines = [f"Domain hint: {domain}", "", "Search results:"]
    for i, r in enumerate(results):
        if "error" in r:
            continue
        lines.append(f"[{i}] {r['title']} ({r['url']})\n{r['content'][:500]}")
    if exclude_hint:
        lines.append(
            "\nEntities already used for this domain - pick a DIFFERENT one, "
            f"do not repeat any of: {', '.join(exclude_hint)}"
        )
    lines.append(
        f"\nPreferred entity type for this seed: {target_type}. Only use this "
        "type if the search results genuinely support a good, moderately-obscure "
        "entity of that type - if they don't, pick whichever type the results DO "
        "support well rather than forcing a poor fit."
    )
    return "\n\n".join(lines)


def run(domains: list, n_per_domain: int = 3) -> None:
    """
    domains: list of topic strings to seed from, e.g.
        ["regional sports coaches", "independent film composers", "niche academic awards"]
    Resumable: seed ids are deterministic (domain + index), so a re-run
    skips domains/indices already present in the output file.
    """
    writer = ResumableWriter(OUTPUT_PATH, key="id")

    # Build per-domain exclusion state from whatever's already in the output
    # file (handles resumed runs) - a set for O(1) exact-dedup checks, and an
    # ordered list to source the bounded prompt hint from.
    used_norm_by_domain = {}
    used_list_by_domain = {}
    for r in writer.all():
        if r.get("status") == "ok" and r.get("entity_A"):
            d = r.get("domain", "")
            norm = r["entity_A"].strip().lower()
            used_norm_by_domain.setdefault(d, set()).add(norm)
            used_list_by_domain.setdefault(d, []).append(r["entity_A"])

    for domain_hint in domains:
        primary_domain, subdomain = _parse_domain(domain_hint)
        used_norm = used_norm_by_domain.setdefault(domain_hint, set())
        used_list = used_list_by_domain.setdefault(domain_hint, [])

        for i in range(n_per_domain):
            seed_id = f"seed_{domain_hint.replace(' ', '_')}_{i}"
            if writer.is_done(seed_id):
                continue
            target_type = ENTITY_TYPE_ROTATION[i % len(ENTITY_TYPE_ROTATION)]
            record = None
            for attempt in range(MAX_DEDUP_RETRIES + 1):
                exclude_hint = used_list[-DIVERSITY_HINT_SAMPLE_SIZE:]  # fixed size, never grows prompt
                results = search(
                    f"What {target_type} related to {primary_domain}-{subdomain} can be identified in 2026?",
                    start_date=START_DATE,
                    end_date=END_DATE
                )
                try:
                    out = call_local_llm(
                        SYSTEM_PROMPT,
                        _build_user_prompt(primary_domain+"-"+subdomain, results, exclude_hint, target_type),
                        sample=True
                    )
                except Exception as e:
                    out = {"status": "failed", "reason": str(e)}

                if out.get("status") == "ok" and out.get("entity_A"):
                    norm = out["entity_A"].strip().lower()
                    if norm in used_norm:
                        continue
                    used_norm.add(norm)
                    used_list.append(out["entity_A"])
                    record = {"id": seed_id, "domain": primary_domain, "subdomain": subdomain, **out}
                    break
                else:
                    record = {"id": seed_id, "domain": primary_domain, "subdomain": subdomain, **out}
                    break

            if record is None:
                record = {"id": seed_id, "domain": primary_domain, "subdomain": subdomain, "status": "failed",
                           "reason": f"only produced duplicates after {MAX_DEDUP_RETRIES + 1} attempts"}

            writer.append(record)
            print(f"[step1] {seed_id}: {record.get('status')} - {record.get('entity_A', record.get('reason'))}")


if __name__ == "__main__":
    run(domains=["sports > regional coaches", "film > independent documentary directors"], n_per_domain=2)