#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def find_latest_date_dir(base_dir: Path) -> Optional[str]:
    if not base_dir.exists():
        return None
    candidates = [p.name for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Sort by mtime, newest first
    candidates_sorted = sorted(
        candidates,
        key=lambda name: (base_dir / name).stat().st_mtime,
        reverse=True,
    )
    return candidates_sorted[0]


def parse_log_for_perf_and_errors(log_text: str) -> Tuple[Optional[float], Optional[float], bool, bool]:
    # Performance line can appear multiple times; use the last one.
    perf_matches = re.findall(r"Performance:\s*([0-9]+(?:\.[0-9]+)?)\s+([0-9]+(?:\.[0-9]+)?)", log_text)
    if perf_matches:
        last = perf_matches[-1]
        ns_per_day = float(last[0])
        hours_per_ns = float(last[1])
    else:
        ns_per_day = None
        hours_per_ns = None

    # CUDA 700 and related illegal access patterns
    cuda700 = bool(
        re.search(
            r"(CUDA[^\n]*700|error\s*700|cudaErrorIllegalAddress|illegal memory access)",
            log_text,
            re.IGNORECASE,
        )
    )

    finished = "Finished mdrun" in log_text
    return ns_per_day, hours_per_ns, cuda700, finished


def summarize(method: str, molecule: str, date: Optional[str], max_seed: int, base: Path) -> None:
    run_dir = base / molecule / method
    if date is None:
        date = find_latest_date_dir(run_dir)
        if date is None:
            print("No simulation directories found to summarize.")
            return

    sim_dir = run_dir / date
    if not sim_dir.exists():
        print(f"Simulation directory not found: {sim_dir}")
        return

    print(f"Scanning logs under: {sim_dir}")

    results: List[Dict] = []
    for seed in range(max_seed + 1):
        log_file = sim_dir / f"{seed}.log"
        # Fallback: if per-seed log isn't present, try md.log under the seed folder
        if not log_file.exists():
            alt = sim_dir / str(seed) / "md.log"
            log_file = alt if alt.exists() else log_file

        if not log_file.exists():
            results.append({
                "seed": seed,
                "log": None,
                "ns_per_day": None,
                "hours_per_ns": None,
                "cuda700": False,
                "finished": False,
                "status": "missing",
            })
            continue

        try:
            text = log_file.read_text(errors="ignore")
        except Exception:
            results.append({
                "seed": seed,
                "log": str(log_file),
                "ns_per_day": None,
                "hours_per_ns": None,
                "cuda700": False,
                "finished": False,
                "status": "unreadable",
            })
            continue

        ns_per_day, hours_per_ns, cuda700, finished = parse_log_for_perf_and_errors(text)
        results.append({
            "seed": seed,
            "log": str(log_file),
            "ns_per_day": ns_per_day,
            "hours_per_ns": hours_per_ns,
            "cuda700": cuda700,
            "finished": finished,
            "status": "ok",
        })

    # Print per-seed summary
    print("\nPer-seed results:")
    print("seed\tstatus\tcuda700\tfinished\tns/day\thours/ns\tlog")
    for r in results:
        print(
            f"{r['seed']}\t{r['status']}\t{str(r['cuda700']).lower()}\t"
            f"{str(r['finished']).lower()}\t{r['ns_per_day'] if r['ns_per_day'] is not None else ''}\t"
            f"{r['hours_per_ns'] if r['hours_per_ns'] is not None else ''}\t{r['log'] if r['log'] else ''}"
        )

    # Aggregate
    valid_hours = [r["hours_per_ns"] for r in results if isinstance(r.get("hours_per_ns"), (int, float))]
    valid_nsday = [r["ns_per_day"] for r in results if isinstance(r.get("ns_per_day"), (int, float))]
    cuda_issues = sum(1 for r in results if r["cuda700"]) if results else 0

    print("\nAggregate:")
    print(f"date\t{date}")
    print(f"seeds_processed\t{len(results)}")
    print(f"cuda700_count\t{cuda_issues}")
    if valid_hours:
        avg_hpn = sum(valid_hours) / len(valid_hours)
        min_hpn = min(valid_hours)
        max_hpn = max(valid_hours)
        print(f"hours/ns_avg\t{avg_hpn:.3f}")
        print(f"hours/ns_min\t{min_hpn:.3f}")
        print(f"hours/ns_max\t{max_hpn:.3f}")
    else:
        print("hours/ns_avg\t")

    if valid_nsday:
        avg_nsd = sum(valid_nsday) / len(valid_nsday)
        min_nsd = min(valid_nsday)
        max_nsd = max(valid_nsday)
        print(f"ns/day_avg\t{avg_nsd:.1f}")
        print(f"ns/day_min\t{min_nsd:.1f}")
        print(f"ns/day_max\t{max_nsd:.1f}")
    else:
        print("ns/day_avg\t")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize CUDA errors and performance from GROMACS logs.")
    parser.add_argument("--method", default="tica")
    parser.add_argument("--molecule", default="1fme")
    parser.add_argument("--date", default=None)
    parser.add_argument("--max-seed", type=int, default=0)
    parser.add_argument("--base", default="./simulations")
    args = parser.parse_args()

    summarize(
        method=args.method,
        molecule=args.molecule,
        date=args.date,
        max_seed=args.max_seed,
        base=Path(args.base),
    )


if __name__ == "__main__":
    main()


