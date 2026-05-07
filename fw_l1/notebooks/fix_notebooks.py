import json, pathlib

HERE = pathlib.Path(__file__).parent

def fix_notebook(path):
    nb_path = HERE / path
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    w = (nb.get("metadata", {})
           .get("widgets", {})
           .get("application/vnd.jupyter.widget-state+json", {}))
    if w and "state" not in w:
        w["state"] = {}
        nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print(f"Fixed: {path}")
    else:
        print(f"No fix needed: {path}")

fix_notebook("01_training.ipynb")