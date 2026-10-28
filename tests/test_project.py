from __future__ import annotations

from sentiment import analyze_comments, validate_comments


def test_validate_comments_limits() -> None:
    validate_comments(["a" * 300])

    for comments in ([], ["a" * 301], [str(index) for index in range(21)]):
        try:
            validate_comments(comments)
        except ValueError:
            continue
        raise AssertionError("Entrada inválida foi aceita")


def test_analyze_comments_maps_labels_without_model_download() -> None:
    classificador = lambda comentarios, truncation: [
        {"label": "positive", "score": 0.9} for _ in comentarios
    ]

    resultado = analyze_comments(classificador, ["Texto de teste"])

    assert resultado == [
        {
            "comentário": "Texto de teste",
            "sentimento": "Positivo",
            "pontuação": "90.00%",
        }
    ]
