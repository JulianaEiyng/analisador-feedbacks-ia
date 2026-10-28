from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment import MODEL_NAME, MODEL_REVISION, analyze_comments, load_classifier


DATASET_URL = (
    "https://raw.githubusercontent.com/americanas-tech/b2w-reviews01/"
    "4639429ec698d7821fc99a0bc665fa213d9fcd5a/B2W-Reviews01.csv"
)
DATASET_NAME = "B2W-Reviews01"
DATASET_LICENSE = "CC BY-NC-SA 4.0"
SAMPLE_SIZE = 6_000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def carregar_amostra() -> pd.DataFrame:
    partes: list[pd.DataFrame] = []
    for parte in pd.read_csv(
        DATASET_URL,
        usecols=["overall_rating", "review_text"],
        chunksize=20_000,
        encoding="utf-8",
    ):
        parte = parte.dropna(subset=["overall_rating", "review_text"])
        parte["texto_normalizado"] = (
            parte["review_text"].astype(str).str.strip().str.lower().str.replace(
                r"\s+",
                " ",
                regex=True,
            )
        )
        partes.append(parte.drop_duplicates("texto_normalizado"))

    dados = pd.concat(partes, ignore_index=True).drop_duplicates(
        "texto_normalizado"
    )
    dados["rotulo"] = dados["overall_rating"].map(
        {1: "Negativo", 2: "Negativo", 3: "Neutro", 4: "Positivo", 5: "Positivo"}
    )
    dados = dados.dropna(subset=["rotulo"])
    por_classe = SAMPLE_SIZE // 3
    amostra = pd.concat(
        [
            grupo.sample(n=por_classe, random_state=RANDOM_STATE)
            for _, grupo in dados.groupby("rotulo")
        ],
        ignore_index=True,
    )
    return amostra[["review_text", "rotulo"]].rename(
        columns={"review_text": "texto", "rotulo": "sentimento"}
    )


def criar_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "classificador",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def metricas(
    esperados: list[str],
    previstos: list[str],
) -> dict[str, Any]:
    classes = ["Negativo", "Neutro", "Positivo"]
    return {
        "acuracia": round(float(accuracy_score(esperados, previstos)), 4),
        "f1_macro": round(
            float(
                f1_score(
                    esperados,
                    previstos,
                    labels=classes,
                    average="macro",
                    zero_division=0,
                )
            ),
            4,
        ),
        "matriz_confusao": confusion_matrix(
            esperados,
            previstos,
            labels=classes,
        ).tolist(),
        "ordem_matriz_confusao": classes,
    }


def main() -> None:
    dados = carregar_amostra()
    treino, teste = train_test_split(
        dados,
        test_size=TEST_SIZE,
        stratify=dados["sentimento"],
        random_state=RANDOM_STATE,
    )

    classificador_tradicional = criar_pipeline()
    classificador_tradicional.fit(treino["texto"], treino["sentimento"])
    previsoes_tradicional = classificador_tradicional.predict(teste["texto"])
    resultado_tradicional = metricas(
        teste["sentimento"].tolist(),
        previsoes_tradicional.tolist(),
    )

    classificador_transformer = load_classifier()
    previsoes_transformer = [
        item["sentimento"]
        for item in analyze_comments(
            classificador_transformer,
            teste["texto"].tolist(),
        )
    ]
    resultado_transformer = metricas(
        teste["sentimento"].tolist(),
        previsoes_transformer,
    )

    resultado = {
        "fonte": DATASET_NAME,
        "url_fonte": DATASET_URL,
        "licenca": DATASET_LICENSE,
        "atribuicao": (
            "Real, Livy; Oshiro, Marcio; Mafra, Alexandre. "
            "B2W-reviews01: an open product reviews corpus. STIL, 2019."
        ),
        "observacao_dominio": (
            "São avaliações de produtos, não feedbacks de delivery. "
            "As estrelas são uma aproximação dos rótulos de sentimento."
        ),
        "amostra_total": len(dados),
        "treino": len(treino),
        "teste": len(teste),
        "distribuicao_amostra": dados["sentimento"].value_counts().to_dict(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "mapeamento_estrelas": {
            "1-2": "Negativo",
            "3": "Neutro",
            "4-5": "Positivo",
        },
        "modelos": {
            "tfidf_regressao_logistica": resultado_tradicional,
            "transformer_atual": resultado_transformer,
        },
        "transformer": {
            "nome": MODEL_NAME,
            "revisao": MODEL_REVISION,
        },
    }
    Path("resultados_comparacao.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
