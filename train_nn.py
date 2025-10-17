"""
Train a small neural network on SentenceTransformer embeddings built from action
metadata.
- Reuses the same metadata->text flattening as `train_classifier.py`.
- Uses `sentence-transformers/all-MiniLM-L6-v2` to generate embeddings.
- Trains a tiny PyTorch MLP classifier.
- Saves the trained PyTorch model and the embedding model name used.
"""

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------ Configuration (disable parser) ------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EPOCHS = 6  # best found
BATCH_SIZE = 8
LR = 2e-3  # best found
HIDDEN = 512  # best found
FORCE_CPU = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
SEED = 42


def set_seed(seed: int):
    """Make runs reproducible across torch/numpy/random where possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # deterministic behaviour for cuDNN (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# apply seed at import time to keep behavior consistent unless caller overrides
set_seed(SEED)
# --------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "sample_actions.json"
MODEL_PATH = ROOT / "emb_nn_model.pt"

HARMFUL_KEYWORDS = [
    "racist", "harass", "harassment", "misinformation", "deepfake", "privacy",
    "deanonym", "de-anonym", "extremist", "violence", "sexist",
    "child", "exploit", "manipulat", "market_manipulation", "coordinat",
    "discriminat", "conspiracy", "hate", "abuse",
]

_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME, device=str(DEVICE))
    return _EMBED_MODEL


def is_harmful(entry):
    """Check if an action entry is harmful based on keywords."""
    label = (entry.get("label") or "").lower()
    resource = (entry.get("resource") or "").lower()
    action_meta = entry.get("action", {}) or {}
    server_label = (action_meta.get("server_label") or "").lower()
    require_approval = (action_meta.get("require_approval") or "").lower()

    text = " ".join([label, resource, server_label, require_approval])
    for k in HARMFUL_KEYWORDS:
        if k in text:
            return 1
    return 0


def flatten_action_to_text(entry):
    """Flatten action metadata to text."""
    parts = []
    parts.append(entry.get("label", ""))
    parts.append(entry.get("resource", ""))
    action_meta = entry.get("action", {}) or {}
    parts.append(action_meta.get("server_label", ""))
    server_url = action_meta.get("server_url", "")
    if server_url:
        try:
            host = re.sub(r"^https?://", "", server_url)
            host = host.split("/")[0]
            parts.append(host)
        except Exception:
            parts.append(server_url)
    parts.append(action_meta.get("require_approval", ""))
    params = action_meta.get("parameters") or {}
    if isinstance(params, dict):
        parts.extend(sorted(list(params.keys())))
    else:
        parts.append(str(type(params)))
    return " ".join([str(p) for p in parts if p])


class SimpleMLP(nn.Module):
    """Simple MLP for binary classification."""

    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        """Forward pass."""
        return self.net(x)


def load_texts_and_labels():
    """Load texts and labels from dataset."""
    with open(DATA_PATH, encoding="utf-8") as f:
        j = json.load(f)
    items = j.get("actions", [])
    texts = [flatten_action_to_text(it) for it in items]
    labels = [is_harmful(it) for it in items]
    return texts, labels


def make_embeddings(texts):
    """Generate embeddings for texts."""
    # sentence-transformers returns numpy arrays
    model = _get_embed_model()
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embs)


def train_one(Xtr, ytr, Xte, yte, hidden, lr, epochs):
    """Train the model for one configuration."""
    set_seed(SEED)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    train_ds = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleMLP(in_dim=Xtr.shape[1], hidden=hidden)
    model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            bs = xb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.4f}")

    model.eval()
    with torch.inference_mode():
        logits = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE))
        preds = torch.argmax(logits, dim=1).cpu().numpy()
    acc = accuracy_score(yte, preds)
    return model, acc, preds


def train_model():
    """Train the model with best hyperparameters."""
    # Ensure reproducible split and training
    set_seed(SEED)
    texts, labels = load_texts_and_labels()
    Xtr_texts, Xte_texts, ytr, yte = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    print("Generating embeddings...")
    # embeddings should be deterministic given seed and model
    set_seed(SEED)
    Xtr_embs = make_embeddings(Xtr_texts)
    Xte_embs = make_embeddings(Xte_texts)

    # Use best values discovered interactively
    print("Training...")
    cfg = {"hidden": HIDDEN, "lr": LR, "epochs": EPOCHS}
    model, acc, _ = train_one(
        Xtr_embs, ytr, Xte_embs, yte, hidden=HIDDEN, lr=LR, epochs=EPOCHS
    )
    print(f"-> acc: {acc:.4f}")

    # save best model and embedding info
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_dim": Xtr_embs.shape[1],
            "config": cfg,
        },
        MODEL_PATH,
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    # Run grid search with top-level configuration variables
    train_model()
