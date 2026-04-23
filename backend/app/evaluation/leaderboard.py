"""Publish a Weave leaderboard comparing evaluation profiles.

Creates a W&B leaderboard that compares baseline vs fw_l2 vs fw_l1_l2
across PHI leak rate, block rate, redaction effectiveness, and latency.

Usage:
    uv run leaderboard
"""

from __future__ import annotations

import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

from app.evaluation.weave_eval import (
    RAGModel,
    load_golden_set_as_dataset,
    phi_leak_scorer,
    redaction_scorer,
    injection_scorer,
    latency_scorer,
)


def get_evaluations(
    queries_path: str,
    groundtruth_path: str,
    index_dir: str,
    limit: int | None = None,
) -> dict[str, weave.Evaluation]:
    """Create Weave Evaluation objects for each profile."""
    rows = load_golden_set_as_dataset(queries_path, limit=limit)
    dataset = weave.Dataset(name=f"adversarial-{len(rows)}", rows=rows)

    scorers = [phi_leak_scorer, redaction_scorer, injection_scorer, latency_scorer]

    profiles = ["baseline", "fw_l2_base"]

    evaluations = {}
    for profile in profiles:
        evaluations[profile] = weave.Evaluation(
            name=f"eval-{profile}",
            dataset=dataset,
            scorers=scorers,
            metadata={"profile": profile},
        )

    return evaluations


def publish_leaderboard(evaluations: dict[str, weave.Evaluation]) -> str:
    """Publish a leaderboard comparing evaluation profiles.

    Returns the leaderboard ref URI.
    """
    columns = []

    for profile, evaluation in evaluations.items():
        eval_ref = get_ref(evaluation)
        if eval_ref is None:
            continue
        eval_uri = eval_ref.uri()

        # PHI block rate (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="phi_leak_scorer",
            summary_metric_path="blocked.mean",
        ))

        # SSN leak rate (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="phi_leak_scorer",
            should_minimize=True,
            summary_metric_path="ssn_leaked.mean",
        ))

        # PHI leak rate (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="phi_leak_scorer",
            should_minimize=True,
            summary_metric_path="phi_leaked.mean",
        ))

        # Metadata leak rate (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=True,
            summary_metric_path="metadata_leaked.mean",
        ))

        # Clean refusal rate (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            summary_metric_path="clean_refusal.mean",
        ))

        # Redaction applied (higher is better for fw_l2)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="redaction_scorer",
            summary_metric_path="redaction_applied.mean",
        ))

        # SSN caught by FW-L2 (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="redaction_scorer",
            summary_metric_path="ssn_caught_by_fw_l2.mean",
        ))

        # Latency (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="latency_scorer",
            should_minimize=True,
            summary_metric_path="latency_seconds.mean",
        ))

        # Under 5s rate (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="latency_scorer",
            summary_metric_path="under_5s.mean",
        ))

    spec = leaderboard.Leaderboard(
        name="PHI Protection Leaderboard",
        description=(
            "Compares RAG pipeline configurations across PHI leak prevention, "
            "redaction effectiveness, and response latency. "
            "Profiles: baseline (no firewalls), fw_l2 (response scrubbing), "
            "fw_l1_l2 (full protection)."
        ),
        columns=columns,
    )

    ref = weave.publish(spec)
    print(f"\n[leaderboard] Published: {ref.uri()}")
    return ref.uri()


async def run_and_publish(
    index_dir: str,
    queries_path: str,
    groundtruth_path: str,
    profiles: list[str],
    limit: int | None = None,
) -> str:
    """Run evaluations for each profile and publish a leaderboard.

    Args:
        index_dir: Path to FAISS index.
        queries_path: Path to golden set JSON.
        groundtruth_path: Path to PHI ground truth JSON.
        profiles: List of profiles to evaluate (e.g., ["baseline", "fw_l2"]).
        limit: Max queries per profile.

    Returns:
        Leaderboard ref URI.
    """
    from app.evaluation.weave_eval import run_weave_evaluation

    print(f"[leaderboard] Running evaluations for profiles: {profiles}")
    print(f"[leaderboard] Queries: {queries_path}")
    if limit:
        print(f"[leaderboard] Limit: {limit} queries per profile")

    evaluations = {}

    for profile in profiles:
        print(f"\n{'=' * 50}")
        print(f"  Running evaluation: {profile}")
        print(f"{'=' * 50}")

        results, evaluation = await run_weave_evaluation(
            profile=profile,
            index_dir=index_dir,
            queries_path=queries_path,
            groundtruth_path=groundtruth_path,
            limit=limit,
        )

        evaluations[profile] = evaluation

        # Print quick summary
        phi_scorer = results.get("phi_leak_scorer", {})
        blocked = phi_scorer.get("blocked", {}).get("mean", 0)
        ssn = phi_scorer.get("ssn_leaked", {}).get("mean", 0)
        latency = results.get("latency_scorer", {}).get("latency_seconds", {}).get("mean", 0)

        print(f"\n  {profile}: blocked={blocked:.2%} | ssn_leaked={ssn:.2%} | latency={latency:.2f}s")

    # Publish leaderboard
    print(f"\n{'=' * 50}")
    print(f"  Publishing leaderboard...")
    print(f"{'=' * 50}")

    uri = publish_leaderboard(evaluations)

    print(f"\n  Leaderboard published!")
    print(f"  View on W&B dashboard.")
    print(f"{'=' * 50}")