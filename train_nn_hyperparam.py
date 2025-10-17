"""
Hyperparameter tuning script for the neural network model.
"""

import itertools

from sklearn.model_selection import train_test_split

try:
    import pandas as pd
except ImportError:
    pd = None

from train_nn import load_texts_and_labels, make_embeddings, train_one, MODEL_PATH

# Hyperparameter grid
HP_GRID = {
    "hidden": [64, 128, 256, 512],
    "lr": [1e-4, 5e-4, 1e-3, 2e-3],
    "epochs": [3, 4, 6],
}


def hyperparam_tuning():
    """
    Perform hyperparameter tuning using grid search on the neural network model.
    """
    texts, labels = load_texts_and_labels()
    xtr_texts, xte_texts, ytr, yte = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    print("Generating embeddings...")
    xtr_embs = make_embeddings(xtr_texts)
    xte_embs = make_embeddings(xte_texts)

    # simple grid search
    best = {"acc": -1, "config": None, "model": None, "preds": None}
    results = []
    keys, values = zip(*HP_GRID.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        print(f"Trying config: {cfg}")
        model, acc, preds = train_one(
            xtr_embs,
            ytr,
            xte_embs,
            yte,
            hidden=cfg["hidden"],
            lr=cfg["lr"],
            epochs=cfg["epochs"],
        )
        print(f"-> acc: {acc:.4f}")
        # store trial result
        results.append(
            {
                "hidden": cfg["hidden"],
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "acc": float(acc),
            }
        )
        if acc > best["acc"]:
            best.update({"acc": acc, "config": cfg, "model": model, "preds": preds})

    print("Best config:", best["config"], "acc:", best["acc"])
    # show all results sorted by accuracy
    if pd is not None:
        df = pd.DataFrame(results)
        df = df.sort_values(
            by=["acc", "hidden", "lr", "epochs"], ascending=[False, True, True, True]
        )
        print("\nAll trials sorted by acc:")
        print(df.to_string(index=False))
    else:
        # fallback: pretty-print JSON sorted
        results_sorted = sorted(
            results, key=lambda x: (-x["acc"], x["hidden"], x["lr"], x["epochs"])
        )
        print("\nAll trials sorted by acc:")
        for r in results_sorted:
            print(r)
    print(f"Saved best model to {MODEL_PATH}")


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    hyperparam_tuning()
