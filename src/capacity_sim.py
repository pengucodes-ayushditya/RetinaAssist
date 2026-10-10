"""Scenario simulation for retinal-screening clinic queues.

All timings and model operating characteristics are assumptions for planning
experiments, not validated operational or clinical forecasts.
"""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np


def queue_jobs(arrivals, service_times, servers):
    """FCFS multiserver queue. Return start and finish times in input order."""
    count = len(arrivals)
    starts = np.zeros(count, dtype=float)
    finishes = np.zeros(count, dtype=float)
    if count == 0:
        return starts, finishes

    available = [0.0] * servers
    heapq.heapify(available)
    order = sorted(range(count), key=lambda i: arrivals[i])

    for i in order:
        server_free = heapq.heappop(available)
        start = max(float(arrivals[i]), server_free)
        finish = start + float(service_times[i])
        starts[i] = start
        finishes[i] = finish
        heapq.heappush(available, finish)

    return starts, finishes


def simulate_one_day(args, seed):
    rng = np.random.default_rng(seed)
    horizon = args.open_hours * 60.0

    # Poisson arrivals during clinic opening hours.
    arrivals = []
    time = 0.0
    mean_gap = horizon / args.patients_per_day
    while True:
        time += float(rng.exponential(mean_gap))
        if time >= horizon:
            break
        arrivals.append(time)
    arrivals = np.asarray(arrivals, dtype=float)
    n = len(arrivals)

    if n == 0:
        return None

    disease = rng.random(n) < args.prevalence
    initial_quality_failure = rng.random(n) < args.quality_failure_rate
    recaptured_successfully = (
        rng.random(n) < args.recapture_success_rate
    )
    ungradable = initial_quality_failure & ~recaptured_successfully
    capture_times = arrivals + initial_quality_failure * args.recapture_minutes

    clinician_service = rng.gamma(
        shape=4.0,
        scale=args.clinician_minutes / 4.0,
        size=n,
    )

    manual_start, manual_finish = queue_jobs(
        arrivals, clinician_service, args.clinicians
    )
    manual_wait = manual_start - arrivals

    usable_ids = np.flatnonzero(~ungradable)
    ai_arrivals = capture_times[usable_ids]
    ai_service = rng.exponential(
        args.ai_seconds / 60.0, size=len(usable_ids)
    )
    ai_start, ai_finish = queue_jobs(
        ai_arrivals, ai_service, args.ai_workers
    )

    # Simulated threshold decisions using the supplied sensitivity/specificity.
    random_values = rng.random(len(usable_ids))
    usable_disease = disease[usable_ids]
    flagged = np.where(
        usable_disease,
        random_values < args.sensitivity,
        random_values >= args.specificity,
    )

    # Positive AI results and images still ungradable after recapture go to review.
    review_ids = list(usable_ids[flagged]) + list(np.flatnonzero(ungradable))
    review_arrivals = [
        float(ai_finish[j])
        for j in range(len(usable_ids))
        if flagged[j]
    ] + [
        float(capture_times[i])
        for i in np.flatnonzero(ungradable)
    ]

    review_ids = np.asarray(review_ids, dtype=int)
    review_arrivals = np.asarray(review_arrivals, dtype=float)
    assisted_services = clinician_service[review_ids]
    assisted_start, assisted_finish = queue_jobs(
        review_arrivals, assisted_services, args.clinicians
    )

    model_tp = int(np.count_nonzero(flagged & usable_disease))
    model_fn = int(np.count_nonzero(~flagged & usable_disease))
    model_tn = int(np.count_nonzero(~flagged & ~usable_disease))
    model_fp = int(np.count_nonzero(flagged & ~usable_disease))

    def summarize(starts, finishes, queue_arrivals, service_times, original_ids):
        waits = starts - queue_arrivals
        total_time = finishes - arrivals[original_ids]
        return {
            "clinician_reviews": int(len(original_ids)),
            "review_fraction": float(len(original_ids) / n),
            "clinician_workload_hours": float(service_times.sum() / 60.0),
            "mean_clinician_queue_wait_min": (
                float(waits.mean()) if len(waits) else 0.0
            ),
            "p90_clinician_queue_wait_min": (
                float(np.percentile(waits, 90)) if len(waits) else 0.0
            ),
            "mean_time_to_review_min": (
                float(total_time.mean()) if len(total_time) else 0.0
            ),
            "minutes_past_clinic_close": (
                float(max(0.0, finishes.max() - horizon))
                if len(finishes) else 0.0
            ),
        }

    manual = summarize(
        manual_start,
        manual_finish,
        arrivals,
        clinician_service,
        np.arange(n),
    )
    assisted = summarize(
        assisted_start,
        assisted_finish,
        review_arrivals,
        assisted_services,
        review_ids,
    )

    assisted["reviews_avoided_vs_manual"] = int(n - len(review_ids))
    assisted["recaptures"] = int(initial_quality_failure.sum())
    assisted["images_referred_for_quality"] = int(ungradable.sum())
    assisted["simulated_false_negatives"] = model_fn
    assisted["model_sensitivity_on_gradable_images"] = (
        float(model_tp / (model_tp + model_fn))
        if model_tp + model_fn else None
    )
    assisted["model_specificity_on_gradable_images"] = (
        float(model_tn / (model_tn + model_fp))
        if model_tn + model_fp else None
    )
    return {"patients": n, "manual": manual, "ai_assisted": assisted}


def average_runs(runs, scenario):
    keys = [
        "clinician_reviews",
        "review_fraction",
        "clinician_workload_hours",
        "mean_clinician_queue_wait_min",
        "p90_clinician_queue_wait_min",
        "mean_time_to_review_min",
        "minutes_past_clinic_close",
    ]
    result = {}
    for key in keys:
        values = [run[scenario][key] for run in runs]
        result[key] = float(np.mean(values))
    if scenario == "ai_assisted":
        extra = [
            "reviews_avoided_vs_manual",
            "recaptures",
            "images_referred_for_quality",
            "simulated_false_negatives",
            "model_sensitivity_on_gradable_images",
            "model_specificity_on_gradable_images",
        ]
        for key in extra:
            values = [
                run[scenario][key]
                for run in runs
                if run[scenario][key] is not None
            ]
            result[key] = float(np.mean(values)) if values else None
    result["average_patients"] = float(np.mean([r["patients"] for r in runs]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients-per-day", type=int, default=150)
    parser.add_argument("--open-hours", type=float, default=8.0)
    parser.add_argument("--prevalence", type=float, default=0.18)
    parser.add_argument("--sensitivity", type=float, default=0.855)
    parser.add_argument("--specificity", type=float, default=0.559)
    parser.add_argument("--quality-failure-rate", type=float, default=0.04)
    parser.add_argument("--recapture-success-rate", type=float, default=0.75)
    parser.add_argument("--recapture-minutes", type=float, default=3.0)
    parser.add_argument("--ai-seconds", type=float, default=2.0)
    parser.add_argument("--clinician-minutes", type=float, default=8.0)
    parser.add_argument("--clinicians", type=int, default=1)
    parser.add_argument("--ai-workers", type=int, default=2)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/capacity_simulation.json")
    args = parser.parse_args()

    if args.patients_per_day <= 0 or args.open_hours <= 0:
        parser.error("patients-per-day and open-hours must be positive")
    if args.clinicians <= 0 or args.ai_workers <= 0 or args.runs <= 0:
        parser.error("clinicians, ai-workers, and runs must be positive")
    for name in ("prevalence", "sensitivity", "specificity",
                 "quality_failure_rate", "recapture_success_rate"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"{name.replace('_', '-')} must be between 0 and 1")

    runs = [
        simulate_one_day(args, args.seed + i)
        for i in range(args.runs)
    ]
    runs = [run for run in runs if run is not None]

    report = {
        "simulation": "RetinaAssist clinic capacity scenario",
        "replications": len(runs),
        "assumptions": {
            "patients_per_day_target": args.patients_per_day,
            "clinic_open_hours": args.open_hours,
            "assumed_disease_prevalence": args.prevalence,
            "sensitivity_assumption": args.sensitivity,
            "specificity_assumption": args.specificity,
            "quality_failure_rate": args.quality_failure_rate,
            "recapture_success_rate": args.recapture_success_rate,
            "recapture_minutes": args.recapture_minutes,
            "mean_ai_processing_seconds": args.ai_seconds,
            "mean_clinician_review_minutes": args.clinician_minutes,
            "clinicians": args.clinicians,
            "ai_workers": args.ai_workers,
            "seed": args.seed,
        },
        "manual_review_every_image": average_runs(runs, "manual"),
        "ai_assisted_triage": average_runs(runs, "ai_assisted"),
        "caution": (
            "Scenario simulation only. Model sensitivity/specificity are "
            "assumptions based on a small external research evaluation, not "
            "validated clinical performance or a staffing recommendation."
        ),
    }

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved simulation: {output}")


if __name__ == "__main__":
    main()
