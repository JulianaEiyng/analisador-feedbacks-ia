from __future__ import annotations

from typing import Any

import streamlit as st
from transformers import pipeline


MODEL_NAME = "cardiffnlp/mbert-base-tweet-sentiment-pt"
MODEL_REVISION = "605a6e533ff59d296978653dd34c556e9c9b1671"
MAX_COMMENTS = 20
MAX_COMMENT_LENGTH = 300

LABELS = {
    "negative": "Negativo",
    "neutral": "Neutro",
    "positive": "Positivo",
}


def validate_comments(comments: list[str]) -> None:
    if not comments:
        raise ValueError("Digite pelo menos um comentário não vazio.")
    if len(comments) > MAX_COMMENTS:
        raise ValueError(f"Envie no máximo {MAX_COMMENTS} comentários por vez.")
    if any(len(comment) > MAX_COMMENT_LENGTH for comment in comments):
        raise ValueError(
            "Cada comentário deve ter no máximo "
            f"{MAX_COMMENT_LENGTH} caracteres. O texto não foi cortado."
        )


@st.cache_resource(show_spinner=False)
def load_classifier() -> Any:
    return pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        revision=MODEL_REVISION,
        device=-1,
    )


def analyze_comments(classifier: Any, comments: list[str]) -> list[dict[str, Any]]:
    predictions = classifier(comments, truncation=True)
    return [
        {
            "comentário": comment,
            "sentimento": LABELS[prediction["label"].lower()],
            "pontuação": f"{float(prediction['score']):.2%}",
        }
        for comment, prediction in zip(comments, predictions)
    ]
