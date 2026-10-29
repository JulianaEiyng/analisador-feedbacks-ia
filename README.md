# Analisador de sentimentos de feedbacks

[Abra o app público](https://juliana-feedbacks-ia.streamlit.app/)

Classifica comentários curtos de delivery em português.
Mostra se cada comentário é positivo, neutro ou negativo.
Exibe a pontuação retornada pelo modelo.
É um projeto pessoal para estudar NLP em uma aplicação simples.

## Aviso

Este é um projeto pessoal de estudo, sem vínculo com o iFood. O Transformer usado no app é pré-treinado e foi integrado sem treinamento ou fine-tuning.

## O que ele faz

O usuário digita comentários, um por linha. O app mostra se cada comentário é positivo, neutro ou negativo, com a pontuação do modelo.

## Tecnologias

- Python 3.12.10
- Streamlit
- Hugging Face Transformers
- PyTorch (execução em CPU)
- pandas
- scikit-learn
- Modelo `cardiffnlp/mbert-base-tweet-sentiment-pt`

## Resultados

| Conjunto de teste | Modelo | Acurácia | Macro-F1 |
|---|---|---:|---:|
| B2W-Reviews01: 1.200 textos de teste, de uma amostra de 6.000 avaliações | TF-IDF + regressão logística | 73,50% | 73,27% |
| B2W-Reviews01: 1.200 textos de teste, de uma amostra de 6.000 avaliações | Transformer atual | 55,42% | 53,66% |
| 30 exemplos fictícios de delivery | Transformer atual | 63,3% | — |

Os testes são pequenos. O B2W-Reviews01 reúne avaliações de produtos, não feedbacks de delivery. O TF-IDF foi um experimento separado e não faz parte do app. O app mantém o Transformer porque a comparação foi feita em avaliações de produtos, e uma avaliação específica de feedbacks de delivery ainda não foi feita.

## Como rodar

No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

## Testes

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Há um workflow no GitHub Actions ([`.github/workflows/testes.yml`](./.github/workflows/testes.yml)) que executa esses testes em pushes e pull requests.

## Limitações

- O modelo foi treinado em textos de redes sociais, não especificamente para atendimento de delivery.
- Negação e ironia podem causar erros de classificação.
- Na versão hospedada, os comentários enviados pelo navegador vão ao servidor para serem processados.

## Créditos e licença dos dados

Foi usada uma amostra do [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), com licença **CC BY-NC-SA 4.0**.

> Real, Livy; Oshiro, Marcio; Mafra, Alexandre. *B2W-reviews01: an open product reviews corpus*. STIL, 2019.

A fonte também orienta dar crédito à B2W Digital. A licença permite atribuição e adaptação para uso não comercial, com compartilhamento pela mesma licença; não é uma autorização para uso comercial. Consulte a referência: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[Detalhes técnicos completos](./docs/DETALHES.md)
