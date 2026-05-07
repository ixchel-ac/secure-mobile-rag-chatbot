"""Publish a Weave leaderboard comparing evaluation profiles.

Creates a W&B leaderboard that compares profiles across:
- PHI value leakage (per PII type: SSN, DOB, phone, email, name, address)
- Metadata leakage (redaction tags, field mentions, refusal wording)
- FW-L2 redaction effectiveness (caught vs missed, per type)
- Injection detection
- Response latency

Usage:
    uv run leaderboard
"""

from __future__ import annotations

import wandb
import weave
from weave.flow import leaderboard
from weave.trace.ref_util import get_ref

from app.evaluation.weave_eval import (
    RAGModel,
    load_golden_set_as_dataset,
    phi_leak_scorer,
    metadata_leak_scorer,
    redaction_scorer,
    injection_scorer,
    latency_scorer,
    classification_scorer,
    classification_summary_scorer,
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

    scorers = [phi_leak_scorer, metadata_leak_scorer, redaction_scorer,
               injection_scorer, latency_scorer]

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

        # ── PHI Value Leakage (phi_leak_scorer) ─────────────────────

        # Primary metric: no PHI values leaked (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="phi_leak_scorer",
            should_minimize=False,
            summary_metric_path="no_phi_leaked.mean",
        ))

        # Per-type value leaks (lower is better)
        for phi_type in ["ssn", "dob", "phone", "email", "name", "address"]:
            columns.append(leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_uri,
                scorer_name="phi_leak_scorer",
                should_minimize=True,
                summary_metric_path=f"{phi_type}_leaked.mean",
            ))

        # Any ground truth PHI leaked (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="phi_leak_scorer",
            should_minimize=True,
            summary_metric_path="phi_leaked.mean",
        ))

        # ── Metadata Leakage (metadata_leak_scorer) ─────────────────

        # Overall metadata revealed (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=True,
            summary_metric_path="metadata_revealed.mean",
        ))

        # Redaction tags leaked — [SSN], [NAME], etc. (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=True,
            summary_metric_path="redaction_tags_leaked.mean",
        ))

        # Field mention leaked (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=True,
            summary_metric_path="field_mention_leaked.mean",
        ))

        # Refusal leaked (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=True,
            summary_metric_path="refusal_leaked.mean",
        ))

        # Clean refusal rate (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="metadata_leak_scorer",
            should_minimize=False,
            summary_metric_path="clean_refusal.mean",
        ))

        # ── FW-L2 Redaction Effectiveness (redaction_scorer) ────────

        # Redaction applied (higher means FW-L2 is active)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="redaction_scorer",
            should_minimize=False,
            summary_metric_path="redaction_applied.mean",
        ))

        # Per-type caught by FW-L2 (higher is better)
        for phi_type in ["ssn", "dob", "phone", "email", "name", "address"]:
            columns.append(leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_uri,
                scorer_name="redaction_scorer",
                should_minimize=False,
                summary_metric_path=f"{phi_type}_caught_by_fw_l2.mean",
            ))

        # Per-type missed by FW-L2 (lower is better)
        for phi_type in ["ssn", "dob", "phone", "email", "name", "address"]:
            columns.append(leaderboard.LeaderboardColumn(
                evaluation_object_ref=eval_uri,
                scorer_name="redaction_scorer",
                should_minimize=True,
                summary_metric_path=f"{phi_type}_missed_by_fw_l2.mean",
            ))

        # ── Classification Metrics (classification_scorer) ──────────

        # Accuracy (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="classification_summary_scorer",
            should_minimize=False,
            summary_metric_path="accuracy.mean",
        ))

        # TP rate — adversarial correctly blocked (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="classification_scorer",
            should_minimize=False,
            summary_metric_path="tp.mean",
        ))

        # FP rate — benign incorrectly refused (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="classification_scorer",
            should_minimize=True,
            summary_metric_path="fp.mean",
        ))

        # FN rate — adversarial that leaked (lower is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="classification_scorer",
            should_minimize=True,
            summary_metric_path="fn.mean",
        ))

        # TN rate — benign correctly answered (higher is better)
        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="classification_scorer",
            should_minimize=False,
            summary_metric_path="tn.mean",
        ))

        # ── Injection Detection (injection_scorer) ──────────────────

        columns.append(leaderboard.LeaderboardColumn(
            evaluation_object_ref=eval_uri,
            scorer_name="injection_scorer",
            should_minimize=True,
            summary_metric_path="injection_detected.mean",
        ))

        # ── Latency (latency_scorer) ────────────────────────────────

        # Avg latency (lower is better)
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
            should_minimize=False,
            summary_metric_path="under_5s.mean",
        ))

    spec = leaderboard.Leaderboard(
        name="PHI Protection Leaderboard",
        description=(
            "Compares RAG pipeline configurations across PHI value leakage "
            "(per PII type), metadata leakage (redaction tags, field mentions, "
            "refusal wording), FW-L2 redaction effectiveness, classification "
            "metrics (accuracy, TP/FP/TN/FN), injection detection, and latency."
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
    mode: str = "local",
    remote_url: str | None = None,
    benign_path: str | None = None,
) -> str:
    """Run evaluations for each profile and publish a leaderboard.

    Args:
        index_dir: Path to FAISS index (used in local mode).
        queries_path: Path to adversarial golden set JSON.
        groundtruth_path: Path to PHI ground truth JSON.
        profiles: List of profiles to evaluate (e.g., ["baseline", "fw_l2"]).
        limit: Max queries per profile.
        mode: "local" to run pipeline locally, "remote" to call /test endpoint.
        remote_url: Base URL for remote mode (e.g., "https://...run.app").
        benign_path: Path to benign queries JSON. If provided, runs combined
                     evaluation with classification metrics (accuracy, P/R/F1).

    Returns:
        Leaderboard ref URI.
    """
    from app.evaluation.weave_eval import (
        run_weave_evaluation, run_weave_evaluation_remote,
        clear_collected_rows, get_collected_rows,
    )

    print(f"[leaderboard] Mode: {mode}")
    if mode == "remote":
        print(f"[leaderboard] Remote URL: {remote_url}")
    print(f"[leaderboard] Running evaluations for profiles: {profiles}")
    print(f"[leaderboard] Adversarial queries: {queries_path}")
    if benign_path:
        print(f"[leaderboard] Benign queries: {benign_path}")
    else:
        print(f"[leaderboard] Benign queries: none (adversarial-only mode)")
    if limit:
        print(f"[leaderboard] Limit: {limit} queries per set")

    evaluations = {}

    for profile in profiles:
        print(f"\n{'=' * 50}")
        print(f"  Running evaluation: {profile} ({mode})")
        print(f"{'=' * 50}")

        # Clear per-row collector before each profile
        clear_collected_rows()

        if mode == "remote":
            results, evaluation = await run_weave_evaluation_remote(
                profile=profile,
                remote_url=remote_url,
                queries_path=queries_path,
                groundtruth_path=groundtruth_path,
                benign_path=benign_path,
                limit=limit,
            )
        else:
            results, evaluation = await run_weave_evaluation(
                profile=profile,
                index_dir=index_dir,
                queries_path=queries_path,
                groundtruth_path=groundtruth_path,
                benign_path=benign_path,
                limit=limit,
            )

        evaluations[profile] = evaluation

        # Print quick summary
        phi_scorer = results.get("phi_leak_scorer", {})
        meta_scorer = results.get("metadata_leak_scorer", {})
        cls_scorer = results.get("classification_summary_scorer", {})
        no_phi = phi_scorer.get("no_phi_leaked", {}).get("mean", 0)
        ssn = phi_scorer.get("ssn_leaked", {}).get("mean", 0)
        name = phi_scorer.get("name_leaked", {}).get("mean", 0)
        metadata = meta_scorer.get("metadata_revealed", {}).get("mean", 0)
        clean = meta_scorer.get("clean_refusal", {}).get("mean", 0)
        accuracy = cls_scorer.get("accuracy", {}).get("mean", 0)
        latency = results.get("latency_scorer", {}).get("latency_seconds", {}).get("mean", 0)

        # Compute precision/recall/F1 from aggregated TP/FP/FN
        cls_raw = results.get("classification_scorer", {})
        tp = cls_raw.get("tp", {}).get("mean", 0)
        fp = cls_raw.get("fp", {}).get("mean", 0)
        fn = cls_raw.get("fn", {}).get("mean", 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  {profile}:")
        print(f"    PHI values:     no_leak={no_phi:.2%} | ssn={ssn:.2%} | name={name:.2%}")
        print(f"    Metadata:       revealed={metadata:.2%} | clean_refusal={clean:.2%}")
        print(f"    Classification: accuracy={accuracy:.2%} | precision={precision:.2%} | recall={recall:.2%} | F1={f1:.2%}")
        print(f"    Latency:        {latency:.2f}s")

        # Log confusion matrix to W&B
        eval_meta = evaluation.metadata or {}
        total = eval_meta.get("total_queries", 0)
        if total > 0:
            log_confusion_matrix(profile, results, total)

        # Log per-category risk and generator-vs-FW-L2 protection charts
        collected = get_collected_rows()
        if collected:
            log_risk_and_protection_charts(profile, collected)

    # Publish leaderboard
    print(f"\n{'=' * 50}")
    print(f"  Publishing leaderboard...")
    print(f"{'=' * 50}")

    uri = publish_leaderboard(evaluations)

    print(f"\n  Leaderboard published!")
    print(f"  View on W&B dashboard.")
    print(f"{'=' * 50}")


def log_confusion_matrix(
    profile: str,
    results: dict,
    total_queries: int,
) -> None:
    """Log a confusion matrix to W&B for a single profile evaluation.

    Computes a 2x2 matrix (block vs allow) from the classification scorer's
    aggregated TP/FP/FN/TN means, and logs it as a W&B chart.

    Args:
        profile: The evaluation profile name.
        results: The Weave evaluation results dict.
        total_queries: Total number of queries in the evaluation.
    """
    cls_raw = results.get("classification_scorer", {})
    tp_mean = cls_raw.get("tp", {}).get("mean", 0)
    fp_mean = cls_raw.get("fp", {}).get("mean", 0)
    fn_mean = cls_raw.get("fn", {}).get("mean", 0)
    tn_mean = cls_raw.get("tn", {}).get("mean", 0)

    # Convert means to counts
    tp = round(tp_mean * total_queries)
    fp = round(fp_mean * total_queries)
    fn = round(fn_mean * total_queries)
    tn = round(tn_mean * total_queries)

    # Build prediction and ground truth lists for wandb confusion matrix
    # Use integer labels: 0=allow, 1=block
    y_true = []
    y_pred = []

    # TP: expected=block, predicted=block
    y_true.extend([1] * tp)
    y_pred.extend([1] * tp)

    # FN: expected=block, predicted=allow (leaked)
    y_true.extend([1] * fn)
    y_pred.extend([0] * fn)

    # FP: expected=allow, predicted=block (false alarm)
    y_true.extend([0] * fp)
    y_pred.extend([1] * fp)

    # TN: expected=allow, predicted=allow
    y_true.extend([0] * tn)
    y_pred.extend([0] * tn)

    if not y_true:
        print(f"  [confusion] No classification data for {profile}, skipping")
        return

    # Compute metrics for the table
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_queries if total_queries > 0 else 0

    # Log to W&B
    run = wandb.init(
        project="mobile-rag-firewall",
        name=f"confusion-{profile}",
        job_type="confusion-matrix",
        tags=["leaderboard", "confusion-matrix", profile],
        reinit=True,
    )

    # Confusion matrix chart
    run.log({
        f"confusion_matrix/{profile}": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=["allow", "block"],
            title=f"Confusion Matrix — {profile}",
        ),
    })

    # Summary metrics table
    metrics_table = wandb.Table(
        columns=["Profile", "TP", "FP", "FN", "TN",
                 "Accuracy", "Precision", "Recall", "F1"],
        data=[[profile, tp, fp, fn, tn,
               round(accuracy, 4), round(precision, 4),
               round(recall, 4), round(f1, 4)]],
    )
    run.log({f"classification_metrics/{profile}": metrics_table})

    # Also log as summary scalars for easy comparison
    run.summary[f"accuracy"] = accuracy
    run.summary[f"precision"] = precision
    run.summary[f"recall"] = recall
    run.summary[f"f1"] = f1
    run.summary[f"tp"] = tp
    run.summary[f"fp"] = fp
    run.summary[f"fn"] = fn
    run.summary[f"tn"] = tn

    run.finish()
    print(f"  [confusion] Logged confusion matrix for {profile} to W&B")


def log_risk_and_protection_charts(
    profile: str,
    collected_rows: list[dict],
) -> None:
    """Log per-category risk chart and generator-vs-FW-L2 protection chart to W&B.

    Chart 1 — Per-category risk:
        Grouped bar chart showing data leak rate and metadata leak rate
        for each category (safe, C1-C5). Identifies which attack categories
        pose the highest risk to the system.

    Chart 2 — Generator vs FW-L2 protection:
        Stacked bar chart showing, for each category, what percentage of
        queries were protected by:
        - Generator alone (LLM refused/didn't leak)
        - FW-L2 (generator leaked but FW-L2 caught it)
        - Neither (both failed, PHI reached the user)

    Args:
        profile: The evaluation profile name.
        collected_rows: Per-row data from detail_collector_scorer.
    """
    from collections import defaultdict

    if not collected_rows:
        print(f"  [charts] No collected data for {profile}, skipping")
        return

    # ── Aggregate by category ───────────────────────────────────────

    category_stats = defaultdict(lambda: {
        "total": 0,
        "data_leaked": 0,
        "metadata_leaked": 0,
        "generator_protected": 0,
        "fw_l2_saved": 0,
        "both_failed": 0,
        "raw_data_leaked": 0,
    })

    for row in collected_rows:
        cat = row["category"]
        stats = category_stats[cat]
        stats["total"] += 1
        if row["final_data_leaked"]:
            stats["data_leaked"] += 1
        if row["final_metadata_leaked"]:
            stats["metadata_leaked"] += 1
        if row["generator_protected"]:
            stats["generator_protected"] += 1
        if row["fw_l2_saved"]:
            stats["fw_l2_saved"] += 1
        if row["both_failed"]:
            stats["both_failed"] += 1
        if row["raw_data_leaked"]:
            stats["raw_data_leaked"] += 1

    # ── Build W&B tables ────────────────────────────────────────────

    # Chart 1: Per-category risk
    risk_table = wandb.Table(
        columns=["Category", "Total", "Data Leak Rate", "Metadata Leak Rate",
                 "Data Leaked", "Metadata Leaked"],
    )
    for cat in ["safe", "C1", "C2", "C3", "C4", "C5"]:
        stats = category_stats.get(cat)
        if not stats or stats["total"] == 0:
            continue
        total = stats["total"]
        risk_table.add_data(
            cat,
            total,
            round(stats["data_leaked"] / total, 4),
            round(stats["metadata_leaked"] / total, 4),
            stats["data_leaked"],
            stats["metadata_leaked"],
        )

    # Chart 2: Generator vs FW-L2 protection
    protection_table = wandb.Table(
        columns=["Category", "Total",
                 "Generator Protected", "FW-L2 Saved", "Both Failed",
                 "Generator Rate", "FW-L2 Rate", "Failure Rate"],
    )
    for cat in ["safe", "C1", "C2", "C3", "C4", "C5"]:
        stats = category_stats.get(cat)
        if not stats or stats["total"] == 0:
            continue
        total = stats["total"]
        protection_table.add_data(
            cat,
            total,
            stats["generator_protected"],
            stats["fw_l2_saved"],
            stats["both_failed"],
            round(stats["generator_protected"] / total, 4),
            round(stats["fw_l2_saved"] / total, 4),
            round(stats["both_failed"] / total, 4),
        )

    # ── Log to W&B ──────────────────────────────────────────────────

    run = wandb.init(
        project="mobile-rag-firewall",
        name=f"analysis-{profile}",
        job_type="risk-analysis",
        tags=["leaderboard", "risk-analysis", profile],
        reinit=True,
    )

    # Chart 1: Per-category risk (bar chart)
    run.log({
        f"risk_by_category/{profile}": wandb.plot.bar(
            risk_table,
            "Category",
            "Data Leak Rate",
            title=f"Data Leak Rate by Category — {profile}",
        ),
        f"metadata_risk_by_category/{profile}": wandb.plot.bar(
            risk_table,
            "Category",
            "Metadata Leak Rate",
            title=f"Metadata Leak Rate by Category — {profile}",
        ),
    })

    # Full risk table (for custom W&B dashboard charts)
    run.log({f"risk_table/{profile}": risk_table})

    # Chart 2: Generator vs FW-L2 protection (stacked via table)
    run.log({
        f"generator_protection/{profile}": wandb.plot.bar(
            protection_table,
            "Category",
            "Generator Rate",
            title=f"Generator Protection Rate by Category — {profile}",
        ),
        f"fw_l2_protection/{profile}": wandb.plot.bar(
            protection_table,
            "Category",
            "FW-L2 Rate",
            title=f"FW-L2 Save Rate by Category — {profile}",
        ),
        f"failure_rate/{profile}": wandb.plot.bar(
            protection_table,
            "Category",
            "Failure Rate",
            title=f"Neither Protected (Failure) by Category — {profile}",
        ),
    })

    # Full protection table (for custom dashboards)
    run.log({f"protection_table/{profile}": protection_table})

    # Summary scalars — overall rates across all adversarial queries
    adv_rows = [r for r in collected_rows if r["expected_action"] == "block"]
    if adv_rows:
        total_adv = len(adv_rows)
        run.summary["overall_generator_protection_rate"] = sum(
            1 for r in adv_rows if r["generator_protected"]) / total_adv
        run.summary["overall_fw_l2_save_rate"] = sum(
            1 for r in adv_rows if r["fw_l2_saved"]) / total_adv
        run.summary["overall_failure_rate"] = sum(
            1 for r in adv_rows if r["both_failed"]) / total_adv
        run.summary["overall_data_leak_rate"] = sum(
            1 for r in adv_rows if r["final_data_leaked"]) / total_adv
        run.summary["overall_metadata_leak_rate"] = sum(
            1 for r in adv_rows if r["final_metadata_leaked"]) / total_adv

    run.finish()
    print(f"  [charts] Logged risk and protection charts for {profile} to W&B")