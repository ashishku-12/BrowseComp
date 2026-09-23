"""
CLI entry point.

By default this ASKS for domain(s) and sub-domain(s) interactively, since
seed quality (Step 1) depends heavily on picking a good domain/sub-domain
combination. Pass --domains explicitly (or --non-interactive) to skip the
prompts, e.g. for scripted/CI runs.

Examples:
  # Interactive (default) - prompts for domain + sub-domains
  python main.py

  # Non-interactive, domains passed directly - resumable, safe to re-run
  python main.py --domains "sports > regional coaches" "film > indie composers" --n-per-domain 5

  # Run just one step (useful while debugging a specific stage)
  python main.py --only-step 3
"""
import argparse
from pipeline import run_pipeline
from steps import step1_seed, step2_explorer, step3_filter, step4_clue_generator, step5_constructor, step6_redundancy, step7_verify

STEP_FUNCS = {
    1: lambda args: step1_seed.run(domains=args.domains, n_per_domain=args.n_per_domain),
    2: lambda args: step2_explorer.run(),
    3: lambda args: step3_filter.run(),
    4: lambda args: step4_clue_generator.run(),
    5: lambda args: step5_constructor.run(),
    6: lambda args: step6_redundancy.run(),
    7: lambda args: step7_verify.run(),
}


def prompt_for_domains() -> list:
    """
    Interactively collects one or more (domain, sub-domain) pairs.
    Each pair is combined into a single "domain > sub-domain" string, which
    is what Step 1's Seed Agent uses as its topic hint - keeping domain and
    sub-domain together gives the seed search a much more specific target
    than a domain alone (e.g. "sports > regional youth coaches" instead of
    just "sports").
    """
    print("Let's set up the seed topics for this run.\n")
    domains = []
    while True:
        domain = input("Domain (e.g. 'sports', 'film', 'academia'): ").strip()
        if not domain:
            print("  Domain can't be empty, try again.")
            continue

        sub_raw = input(
            f"Sub-domain(s) within '{domain}', comma-separated "
            f"(e.g. 'regional youth coaches, retired Olympians'): "
        ).strip()
        sub_domains = [s.strip() for s in sub_raw.split(",") if s.strip()]
        if not sub_domains:
            # allow a bare domain with no sub-domain if the user really wants that
            sub_domains = [""]

        for sub in sub_domains:
            combined = f"{domain} > {sub}" if sub else domain
            domains.append(combined)

        again = input("Add another domain? [y/N]: ").strip().lower()
        if again != "y":
            break

    print(f"\nUsing {len(domains)} domain/sub-domain seed topic(s):")
    for d in domains:
        print(f"  - {d}")
    print()
    return domains


def main():
    parser = argparse.ArgumentParser(description="Multihop BrowseComp-style dataset pipeline")
    parser.add_argument("--domains", nargs="+", default=None,
                         help="Skip the interactive prompt; pass 'domain > sub-domain' strings directly.")
    parser.add_argument("--n-per-domain", type=int, default=3)
    parser.add_argument("--non-interactive", action="store_true",
                         help="Never prompt; falls back to a small default domain list if --domains is also omitted.")
    parser.add_argument("--only-step", type=int, choices=range(1, 8), default=None,
                         help="Run a single step instead of the full pipeline (still resumable).")
    args = parser.parse_args()

    if args.domains is None:
        if args.non_interactive:
            args.domains = [
                "sports > regional youth coaches",
                "film > independent documentary directors",
                "academia > niche research awards",
            ]
        else:
            args.domains = prompt_for_domains()

    if args.only_step:
        STEP_FUNCS[args.only_step](args)
    else:
        run_pipeline(domains=args.domains, n_per_domain=args.n_per_domain)


if __name__ == "__main__":
    main()