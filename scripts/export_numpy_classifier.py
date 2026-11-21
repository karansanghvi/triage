"""Export the live classifier (or any named one) to .npz, so the server runs it as plain numpy maths without
importing scikit-learn. Training and retraining already do this; run it by hand only for older model files.

    python -m scripts.export_numpy_classifier                    # the classifier named in config/routing.yaml
    python -m scripts.export_numpy_classifier --name classifier_c_logreg_cv --version 2

Each .npz records the SHA-256 of the .joblib it came from; the server ignores an .npz that doesn't match.
"""
import argparse

from app.router.predictor import MODELS_DIR, TIERS, export_numpy, load_classifier_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name")
    parser.add_argument("--version", type=int)
    args = parser.parse_args()
    name, version = load_classifier_config()
    name, version = args.name or name, args.version or version
    for tier in TIERS:
        stem = MODELS_DIR / f"{name}_{tier}_v{version}"
        out = export_numpy(stem)
        print(f"{stem.name}: " + (f"wrote {out.name}" if out else "can't run as numpy (left as .joblib only)"))


if __name__ == "__main__":
    main()
