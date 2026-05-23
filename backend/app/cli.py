"""CLI entry points for uv run commands."""

import json
import sys
from collections import Counter
from pathlib import Path

from app.config import DATA_DIR, INDEX_DIR, EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL, WANDB_PROJECT, TOP_K


def _resolve_paths():
    """Resolve data and index paths.

    Uses DATA_DIR and INDEX_DIR from config. If DATA_DIR is relative,
    resolves it from the project root (backend's parent directory).
    """
    data_dir = Path(DATA_DIR).resolve()
    index_dir = Path(INDEX_DIR).resolve()

    text_dir = data_dir / "synthea" / "text"
    csv_dir = data_dir / "synthea" / "csv"
    return data_dir, text_dir, csv_dir, index_dir


COMMAND_GROUPS = {
    "Data & Indexing": {
        "ingestion":        "Run the full ingestion pipeline: load -> clean -> chunk -> embed -> index",
        "data-show":        "Show statistics about the patient data (counts, sizes, PHI coverage)",
        "faiss-check":      "Show FAISS index statistics (vectors, sections, health check)",
        "search":           "Interactive search over the FAISS index",
    },
    "Golden Set Generation": {
        "generate-adversarial-queries":  "Generate the 1,000-query adversarial golden test set",
        "generate-benign-queries":       "Generate the 1,000-query benign golden test set",
        "generate-compound-queries":     "Generate the 600-query compound (mixed) golden test set",
    },
    "Evaluation": {
        "evaluate":         "Run adversarial evaluation against the RAG pipeline",
        "leaderboard":      "Run all profiles and publish W&B leaderboard",
    },
    "General": {
        "help":             "Show this help message",
    },
}


def help():
    """Show available CLI commands."""
    print("\n" + "=" * 70)
    print("             MOBILE RAG FIREWALL - CLI COMMANDS")
    print("=" * 70)

    print("\nUsage:  cd backend && uv run <command> [options]\n")

    for group_name, commands in COMMAND_GROUPS.items():
        print(f"  {group_name}:")
        for name, description in commands.items():
            print(f"    {name:20s}  {description}")
        print()

    # Show evaluate flags
    print("  evaluate options:")
    print("    --limit N                 Run only the first N queries (default: all)")
    print("    --category C1|C2|C3|C4    Run only queries from a specific category")
    print("    --delay SECONDS           Delay between queries (default: 2.0)")
    print("    --output PATH             Save JSON results to a custom path")
    print("    --quiet                   Suppress per-query output")

    print("\n" + "-" * 70)
    print("Configuration (via .env or environment variables):\n")
    print(f"  DATA_DIR          {DATA_DIR}")
    print(f"  INDEX_DIR         {INDEX_DIR}")
    print(f"  EMBEDDING_MODEL   {EMBEDDING_MODEL}")
    print(f"  LLM_PROVIDER      {LLM_PROVIDER}")
    print(f"  LLM_MODEL         {LLM_MODEL}")
    print(f"  OLLAMA_BASE_URL   {OLLAMA_BASE_URL}")
    print(f"  TOP_K             {TOP_K}")

    print("\n" + "=" * 70)


def ingestion():
    """Run the full ingestion pipeline: load → clean → chunk → embed → index."""
    from app.ingestion.pipeline import run_full_ingestion

    data_dir, text_dir, csv_dir, index_dir = _resolve_paths()
    processed_dir = data_dir / "processed"

    if not text_dir.exists():
        print(f"Error: {text_dir} not found")
        sys.exit(1)

    store, report = run_full_ingestion(
        text_dir, csv_dir, index_dir,
        processed_dir=processed_dir,
        verbose=True,
    )
    report.print_report()


def data_show():
    """Show statistics about the patient data."""
    from app.ingestion.loader import load_all

    _, text_dir, csv_dir, _ = _resolve_paths()

    if not text_dir.exists():
        print(f"Error: {text_dir} not found")
        sys.exit(1)

    records = load_all(text_dir, csv_dir)

    # Basic counts
    matched = sum(1 for r in records if r.phi_entities)
    total_chars = sum(len(r.raw_text) for r in records)
    avg_chars = total_chars // len(records) if records else 0

    print("\n" + "=" * 60)
    print("                  PATIENT DATA REPORT")
    print("=" * 60)

    print(f"\n{'SOURCE FILES':-^60}")
    print(f"  Text directory:        {text_dir}")
    print(f"  CSV directory:         {csv_dir}")
    print(f"  Total .txt files:      {len(records)}")

    print(f"\n{'PATIENTS':-^60}")
    print(f"  Total patients:        {len(records)}")
    print(f"  Matched to CSV:        {matched}")
    match_rate = (matched / len(records) * 100) if records else 0
    print(f"  Match rate:            {match_rate:.1f}%")

    print(f"\n{'TEXT STATS':-^60}")
    print(f"  Total text size:       {total_chars:,} chars ({total_chars / (1024 * 1024):.2f} MB)")
    print(f"  Avg per patient:       {avg_chars:,} chars")
    if records:
        sizes = sorted(len(r.raw_text) for r in records)
        print(f"  Smallest file:         {sizes[0]:,} chars")
        print(f"  Largest file:          {sizes[-1]:,} chars")
        print(f"  Median file:           {sizes[len(sizes) // 2]:,} chars")

    # PHI coverage
    if matched:
        has_ssn = sum(1 for r in records if r.phi_entities.get("ssn"))
        has_dob = sum(1 for r in records if r.phi_entities.get("dob"))
        has_addr = sum(1 for r in records if r.phi_entities.get("address"))

        print(f"\n{'PHI COVERAGE':-^60}")
        print(f"  With SSN:              {has_ssn}")
        print(f"  With DOB:              {has_dob}")
        print(f"  With Address:          {has_addr}")

    # Sample patients
    print(f"\n{'SAMPLE PATIENTS (first 5)':-^60}")
    for record in records[:5]:
        print(f"  {record.patient_name:30s}  {len(record.raw_text):>6,} chars")

    print("\n" + "=" * 60)


def search():
    """Interactive search over the FAISS index."""
    from app.rag.retriever import Retriever

    _, _, _, index_dir = _resolve_paths()

    if not (index_dir / "faiss.index").exists():
        print(f"Error: No FAISS index found at {index_dir}")
        print("Run 'uv run ingestion' first to build the index.")
        sys.exit(1)

    print("[search] Loading retriever...")
    retriever = Retriever(index_dir)

    print("\n" + "=" * 60)
    print("            INTERACTIVE RETRIEVER SEARCH")
    print("=" * 60)
    print(f"  Index:   {index_dir}")
    print(f"  TOP_K:   {TOP_K}")
    print("  Type 'quit' or 'q' to exit.")
    print("  Prefix with 'section:NAME' to filter (e.g. section:MEDICATIONS)")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "q", "exit"):
            print("Bye!")
            break

        # Parse optional section filter: "section:MEDICATIONS what drugs"
        sections = None
        if query.lower().startswith("section:"):
            parts = query.split(" ", 1)
            section_value = parts[0].split(":", 1)[1]
            sections = [s.strip() for s in section_value.split(",")]
            query = parts[1] if len(parts) > 1 else ""
            if not query:
                print("  Error: provide a query after 'section:NAME'")
                continue

        results = retriever.retrieve(query, top_k=TOP_K, sections=sections)

        if not results:
            print("  No results found.")
            continue

        filter_msg = f" (filtered: {', '.join(sections)})" if sections else ""
        print(f"\n  {len(results)} results{filter_msg}:\n")

        for i, chunk in enumerate(results):
            section = chunk.metadata.get("section", "N/A")
            patient = chunk.metadata.get("patient_name", "N/A")
            preview = chunk.text[:200].replace("\n", " ")
            print(f"  [{i + 1}] Score: {chunk.score:.4f}  |  {section}  |  {patient}")
            print(f"      {preview}...")
            print()


def generate_adversarial_queries():
    """Generate the 1,000-query adversarial golden test set."""
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    script = project_root / "data" / "golden_sets" / "generate_adversarial.py"

    if not script.exists():
        print(f"Error: Generator script not found at {script}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script)], cwd=str(project_root))
    sys.exit(result.returncode)


def generate_benign_queries():
    """Generate the 1,000-query benign golden test set."""
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    script = project_root / "data" / "golden_sets" / "generate_benign.py"

    if not script.exists():
        print(f"Error: Generator script not found at {script}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script)], cwd=str(project_root))
    sys.exit(result.returncode)


def generate_compound_queries():
    """Generate the 600-query compound golden test set."""
    import subprocess

    project_root = Path(__file__).parent.parent.parent
    script = project_root / "data" / "golden_sets" / "generate_compound_queries.py"

    if not script.exists():
        print(f"Error: Generator script not found at {script}")
        sys.exit(1)

    result = subprocess.run([sys.executable, str(script)], cwd=str(project_root))
    sys.exit(result.returncode)


def evaluate():
    """Run evaluation against the RAG pipeline.

    Runs benign + adversarial queries, checks for PHI leaks, produces a report.

    Accepts --profile, --queries, --delay, --parallel, --batch-size, --no-save flags.
    """
    import argparse
    from app.evaluation.runner import run_evaluation, run_evaluation_parallel
    from app.rag.pipeline import RAGPipeline

    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--profile", default="baseline",
                        help="Evaluation profile: baseline, fw_l2, fw_l1_l2 (default: baseline)")
    parser.add_argument("--queries", default=None,
                        help="Path to test queries JSON file")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Seconds between queries in sequential mode (default: 2.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of queries to run (default: all)")
    parser.add_argument("--parallel", action="store_true",
                        help="Run queries in parallel (faster, uses async)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Max concurrent queries in parallel mode (default: 5)")
    parser.add_argument("--weave", action="store_true",
                        help="Run as Weave evaluation (logs to W&B leaderboard)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the report JSON")
    args = parser.parse_args()

    data_dir, _, _, index_dir = _resolve_paths()

    # Resolve query file
    if args.queries:
        queries_path = Path(args.queries)
    else:
        queries_path = data_dir / "golden_sets" / "adversarial_queries.json"

    if not queries_path.exists():
        print(f"Error: Query file not found: {queries_path}")
        sys.exit(1)

    # Resolve ground truth
    groundtruth_path = data_dir / "processed" / "phi_groundtruth.json"
    if not groundtruth_path.exists():
        print(f"Error: PHI ground truth not found: {groundtruth_path}")
        print("Run 'uv run ingestion' first.")
        sys.exit(1)

    # Resolve FAISS index
    if not (index_dir / "faiss.index").exists():
        print(f"Error: No FAISS index found at {index_dir}")
        print("Run 'uv run ingestion' first.")
        sys.exit(1)

    # Initialize Weave if using W&B or --weave flag
    if LLM_PROVIDER == "wandb" or args.weave:
        from app.config import WANDB_PROJECT
        import weave
        weave.init(WANDB_PROJECT)
        print(f"[evaluate] Weave tracing enabled (project: {WANDB_PROJECT})")

    # ── Weave evaluation mode ────────────────────────────────
    if args.weave:
        import asyncio
        from app.evaluation.weave_eval import run_weave_evaluation

        print(f"[evaluate] Running Weave evaluation...")
        print(f"[evaluate] Profile: {args.profile}")
        print(f"[evaluate] Queries: {queries_path}")
        if args.limit:
            print(f"[evaluate] Limit: {args.limit}")

        results, _ = asyncio.run(run_weave_evaluation(
            profile=args.profile,
            index_dir=index_dir,
            queries_path=queries_path,
            groundtruth_path=groundtruth_path,
            limit=args.limit,
        ))

        # Print Weave summary
        print("\n" + "=" * 70)
        print(f"          WEAVE EVALUATION SUMMARY — {args.profile.upper()}")
        print("=" * 70)
        for scorer_name, metrics in results.items():
            if isinstance(metrics, dict):
                print(f"\n  {scorer_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, dict) and "mean" in value:
                        print(f"    {metric:30s} {value['mean']:.4f}")
                    elif isinstance(value, dict) and "true_fraction" in value:
                        print(f"    {metric:30s} {value['true_fraction']:.4f}")
                    elif isinstance(value, (int, float)):
                        print(f"    {metric:30s} {value:.4f}")
        print(f"\n  View full results on W&B dashboard")
        print("=" * 70)
        return

    # ── Standard evaluation mode ─────────────────────────────
    mode = "parallel" if args.parallel else "sequential"
    print(f"[evaluate] Profile: {args.profile}")
    print(f"[evaluate] Queries: {queries_path}")
    print(f"[evaluate] Mode: {mode}" + (f" (batch_size={args.batch_size})" if args.parallel else f" (delay={args.delay}s)"))
    print(f"[evaluate] Loading pipeline...")

    # Build pipeline based on profile
    from app.firewall.fw_l2 import FWL2
    from app.rag.generator import SYSTEM_PROMPTS, SYSTEM_PROMPT_HARDENED

    PROFILE_CONFIG = {
        "naive":              {"prompt": "naive",    "fw_l2": False, "ner_backend": None},
        "naive_fw_l2_base":   {"prompt": "naive",    "fw_l2": True,  "ner_backend": "spacy"},
        "naive_fw_l2_bert":   {"prompt": "naive",    "fw_l2": True,  "ner_backend": "bert"},
        "hardened":           {"prompt": "hardened", "fw_l2": False, "ner_backend": None},
        "hardened_fw_l2_base":{"prompt": "hardened", "fw_l2": True,  "ner_backend": "spacy"},
        "hardened_fw_l2_bert":{"prompt": "hardened", "fw_l2": True,  "ner_backend": "bert"},
        "baseline":           {"prompt": "hardened", "fw_l2": False, "ner_backend": None},
        "fw_l2_base":         {"prompt": "hardened", "fw_l2": True,  "ner_backend": "spacy"},
        "fw_l2_bert":         {"prompt": "hardened", "fw_l2": True,  "ner_backend": "bert"},
    }

    config = PROFILE_CONFIG.get(args.profile, PROFILE_CONFIG["baseline"])
    system_prompt = SYSTEM_PROMPTS.get(config["prompt"], SYSTEM_PROMPT_HARDENED)
    ner_backend = config.get("ner_backend")
    fw_l2 = FWL2(ner_backend=ner_backend) if config["fw_l2"] else None

    prompt_name = config["prompt"]
    fw_status = "OFF" if not config["fw_l2"] else f"ON (NER: {ner_backend})"
    print(f"[evaluate] System prompt: {prompt_name}")
    print(f"[evaluate] FW-L2: {fw_status}")

    pipeline = RAGPipeline(index_dir, fw_l2=fw_l2, system_prompt=system_prompt)

    # Run evaluation
    if args.parallel:
        report = run_evaluation_parallel(
            pipeline=pipeline,
            queries_path=queries_path,
            groundtruth_path=groundtruth_path,
            profile=args.profile,
            batch_size=args.batch_size,
            limit=args.limit,
            verbose=True,
        )
    else:
        report = run_evaluation(
            pipeline=pipeline,
            queries_path=queries_path,
            groundtruth_path=groundtruth_path,
            profile=args.profile,
            delay=args.delay,
            limit=args.limit,
            verbose=True,
        )

    # Print report
    report.print_report()

    # Save report
    if not args.no_save:
        results_dir = data_dir / "evaluation_results"
        timestamp = report.timestamp.replace(":", "-")
        output_path = results_dir / f"{args.profile}_{timestamp}.json"
        report.save(output_path)


def leaderboard():
    """Run evaluations for multiple profiles and publish a W&B leaderboard.

    Supports two modes:
    - local: runs the RAG pipeline locally (requires FAISS index + LLM)
    - remote: calls the deployed /test endpoint on Cloud Run

    Examples:
        uv run leaderboard --mode local --profiles naive hardened_fw_l2_base
        uv run leaderboard --mode remote --remote-url https://mobile-rag-firewall-956461831254.us-west2.run.app
    """
    import argparse
    import asyncio
    import weave
    from app.config import WANDB_PROJECT
    from app.evaluation.leaderboard import run_and_publish

    CLOUD_RUN_URL = "https://mobile-rag-firewall-956461831254.us-west2.run.app"

    parser = argparse.ArgumentParser(description="Run leaderboard evaluation")
    parser.add_argument("--mode", choices=["local", "remote"], default="local",
                        help="Run locally or against a remote /test endpoint (default: local)")
    parser.add_argument("--remote-url", default=CLOUD_RUN_URL,
                        help=f"Base URL for remote mode (default: {CLOUD_RUN_URL})")
    parser.add_argument("--profiles", nargs="+",
                        default=["naive", "naive_fw_l2_bert", "hardened", "hardened_fw_l2_bert"],
                        help="Profiles to compare (default: naive naive_fw_l2_bert hardened hardened_fw_l2_bert)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max queries per profile (default: all)")
    parser.add_argument("--queries", default=None,
                        help="Path to adversarial queries JSON file")
    parser.add_argument("--benign-queries", default=None,
                        help="Path to benign queries JSON file (enables combined evaluation "
                             "with classification metrics). Default: data/golden_sets/benign_queries.json")
    parser.add_argument("--adversarial-only", action="store_true",
                        help="Run adversarial queries only (skip benign, no classification metrics)")
    args = parser.parse_args()

    data_dir, _, _, index_dir = _resolve_paths()

    queries_path = Path(args.queries) if args.queries else data_dir / "golden_sets" / "adversarial_queries.json"
    groundtruth_path = data_dir / "processed" / "phi_groundtruth.json"

    # Resolve benign queries path
    if args.adversarial_only:
        benign_path = None
    elif args.benign_queries:
        benign_path = Path(args.benign_queries)
    else:
        # Default: include benign queries if the file exists
        default_benign = data_dir / "golden_sets" / "benign_queries.json"
        benign_path = default_benign if default_benign.exists() else None

    if not queries_path.exists():
        print(f"Error: Query file not found: {queries_path}")
        sys.exit(1)
    if not groundtruth_path.exists():
        print(f"Error: PHI ground truth not found. Run 'uv run ingestion' first.")
        sys.exit(1)
    if benign_path and not benign_path.exists():
        print(f"Error: Benign query file not found: {benign_path}")
        sys.exit(1)

    if args.mode == "local":
        if not (index_dir / "faiss.index").exists():
            print(f"Error: No FAISS index found. Run 'uv run ingestion' first.")
            sys.exit(1)

    weave.init(WANDB_PROJECT)
    print(f"[leaderboard] W&B project: {WANDB_PROJECT}")
    print(f"[leaderboard] Mode: {args.mode}")
    print(f"[leaderboard] Profiles: {args.profiles}")

    asyncio.run(run_and_publish(
        index_dir=str(index_dir),
        queries_path=str(queries_path),
        groundtruth_path=str(groundtruth_path),
        profiles=args.profiles,
        limit=args.limit,
        mode=args.mode,
        remote_url=args.remote_url,
        benign_path=str(benign_path) if benign_path else None,
    ))


def faiss_check():
    """Show FAISS index statistics."""
    _, _, _, index_dir = _resolve_paths()
    index_dir = Path(index_dir)

    faiss_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.jsonl"

    if not faiss_path.exists():
        print(f"Error: No FAISS index found at {faiss_path}")
        print("Run 'uv run ingestion' first to build the index.")
        sys.exit(1)

    # Load FAISS index
    import faiss
    index = faiss.read_index(str(faiss_path))

    # Load metadata
    metadata_rows = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            metadata_rows.append(json.loads(line))

    # Gather stats
    sections = Counter(row.get("section", "UNKNOWN") for row in metadata_rows)
    patients = set(row.get("patient_id", "") for row in metadata_rows)
    text_lengths = [len(row.get("text", "")) for row in metadata_rows]
    avg_text_len = sum(text_lengths) // len(text_lengths) if text_lengths else 0

    # File sizes
    faiss_size = faiss_path.stat().st_size / (1024 * 1024)
    metadata_size = metadata_path.stat().st_size / (1024 * 1024)

    print("\n" + "=" * 60)
    print("                  FAISS INDEX REPORT")
    print("=" * 60)

    print(f"\n{'INDEX':-^60}")
    print(f"  Location:              {index_dir}")
    print(f"  Index type:            {type(index).__name__}")
    print(f"  Total vectors:         {index.ntotal}")
    print(f"  Vector dimension:      {index.d}")
    print(f"  Trained:               {index.is_trained}")

    print(f"\n{'FILE SIZES':-^60}")
    print(f"  faiss.index:           {faiss_size:.2f} MB")
    print(f"  metadata.jsonl:        {metadata_size:.2f} MB")
    print(f"  Total:                 {faiss_size + metadata_size:.2f} MB")

    print(f"\n{'METADATA':-^60}")
    print(f"  Total rows:            {len(metadata_rows)}")
    print(f"  Unique patients:       {len(patients)}")
    print(f"  Avg chunk length:      {avg_text_len} chars")
    if text_lengths:
        print(f"  Shortest chunk:        {min(text_lengths)} chars")
        print(f"  Longest chunk:         {max(text_lengths)} chars")

    print(f"\n{'SECTIONS':-^60}")
    for section, count in sections.most_common():
        pct = count / len(metadata_rows) * 100
        print(f"  {section:20s}   {count:>5}  ({pct:.1f}%)")

    # Check consistency
    print(f"\n{'HEALTH CHECK':-^60}")
    if index.ntotal == len(metadata_rows):
        print(f"  Vectors vs metadata:   OK ({index.ntotal} == {len(metadata_rows)})")
    else:
        print(f"  Vectors vs metadata:   MISMATCH ({index.ntotal} != {len(metadata_rows)})")

    print("\n" + "=" * 60)