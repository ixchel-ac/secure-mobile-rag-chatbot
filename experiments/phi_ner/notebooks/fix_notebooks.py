import json, pathlib

def fix_notebook(path):
    nb_path = pathlib.Path(path)
    nb = json.loads(nb_path.read_text())
    w = (nb.get("metadata", {})
           .get("widgets", {})
           .get("application/vnd.jupyter.widget-state+json", {}))
    if w and "state" not in w:
        w["state"] = {}
        nb_path.write_text(json.dumps(nb, indent=1))
        print(f"Fixed: {path}")
    else:
        print(f"No fix needed: {path}")

fix_notebook("01_training.ipynb")
fix_notebook("02_evaluation.ipynb")
fix_notebook("03_error_analysis.ipynb")
fix_notebook("04_ablation.ipynb")
fix_notebook("05_baselines.ipynb")