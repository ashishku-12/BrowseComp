"""
Orchestrates all six steps in order. Every step is independently resumable
(reads its own output file first and skips finished ids), so re-running
this script after an interruption - or after adding more seed domains -
only does the remaining work, at whichever step it was left at.
"""
import os
from config import DATA_DIR
from steps import step1_seed, step2_explorer, step3_filter, step4_clue_generator, step5_constructor, step6_redundancy, step7_verify


def run_pipeline(domains: list, n_per_domain: int = 3) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("\n=== Step 1: Seed Agent ===")
    step1_seed.run(domains=domains, n_per_domain=n_per_domain)

    print("\n=== Step 2: Chain Explorer ===")
    step2_explorer.run()

    print("\n=== Step 3: Filtering Agent ===")
    step3_filter.run()

    print("\n=== Step 4: Clue Generator ===")
    step4_clue_generator.run()

    print("\n=== Step 5: Question Constructor ===")
    step5_constructor.run()

    print("\n=== Step 6: Redundancy / Reasoning-Graph Check ===")
    step6_redundancy.run()

    print("\n=== Step 7: Verifier ===")
    step7_verify.run()

    _print_summary()


def _print_summary() -> None:
    from utils.io_utils import load_json_list
    final = load_json_list(os.path.join(DATA_DIR, "step7_verified.json"))
    accepted = [r for r in final if r["verification"]["final_verdict"] == "accept"]
    print(f"\n=== Summary ===")
    print(f"Final accepted questions: {len(accepted)} / {len(final)} verified candidates")
    print(f"Full trace at each stage: see files in {DATA_DIR}/")


if __name__ == "__main__":
    # Running pipeline.py directly still asks for domains interactively -
    # this just delegates to main.py's CLI/prompt handling.
    from main import main as _cli_main
    _cli_main()