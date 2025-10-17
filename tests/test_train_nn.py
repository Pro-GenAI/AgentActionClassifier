import numpy as np
import pytest

from train_nn import flatten_action_to_text, is_harmful, make_embeddings


def test_flatten_and_label():
    sample = {
        "label": "Harassment",
        "resource": "chat",
        "action": {
            "server_label": "msg",
            "server_url": "https://a.b/c",
            "parameters": {"to": "x"},
            "require_approval": "never",
        },
    }
    txt = flatten_action_to_text(sample)
    assert "Harassment" in txt
    assert "a.b" in txt
    lbl = is_harmful(sample)
    assert lbl in (0, 1)


@pytest.mark.skipif(not hasattr(np, "zeros"), reason="numpy missing")
def test_make_embeddings_smoke():
    texts = ["hello world", "test"]
    embs = make_embeddings(texts)
    assert embs.shape[0] == 2
