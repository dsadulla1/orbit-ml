"""Write small example CSVs from datasets bundled with scikit-learn (no download needed).

    uv run python examples/make_datasets.py
"""
from pathlib import Path

from sklearn import datasets

OUT = Path(__file__).resolve().parent / "data"


def main() -> None:
    OUT.mkdir(exist_ok=True)

    bc = datasets.load_breast_cancer(as_frame=True)
    df = bc.frame.rename(columns={"target": "diagnosis"})
    df["diagnosis"] = df["diagnosis"].map(dict(enumerate(bc.target_names)))
    df.to_csv(OUT / "breast_cancer.csv", index=False)          # binary -> roc_auc

    wine = datasets.load_wine(as_frame=True)
    df = wine.frame.rename(columns={"target": "cultivar"})
    df["cultivar"] = df["cultivar"].map(dict(enumerate(wine.target_names)))
    df.to_csv(OUT / "wine.csv", index=False)                   # multiclass -> accuracy

    diabetes = datasets.load_diabetes(as_frame=True, scaled=False)
    diabetes.frame.rename(columns={"target": "progression"}).to_csv(
        OUT / "diabetes.csv", index=False)                     # regression -> rmse

    for f in sorted(OUT.glob("*.csv")):
        print(f"wrote {f.relative_to(OUT.parent.parent)}")


if __name__ == "__main__":
    main()
