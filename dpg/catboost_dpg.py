import json
import os
import tempfile
from typing import Any, Dict


def export_catboost_json(model: Any) -> Dict[str, Any]:
    """
    Exporting a trained CatBoost model as its internal JSON structure.
    """
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        model.save_model(path, format="json")
        with open(path) as f:
            return json.load(f)
    finally:
        os.remove(path)


if __name__ == "__main__":
    from catboost import CatBoostClassifier
    from sklearn.datasets import load_iris

    data = load_iris()
    model = CatBoostClassifier(
        iterations=5,
        depth=3,
        grow_policy="SymmetricTree",
        verbose=False,
    )
    model.fit(data.data, data.target)

    structure = export_catboost_json(model)
    print("Top-level keys:", list(structure.keys()))
    print(json.dumps(structure, indent=2)[:2000])
