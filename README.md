# Analisador de sentimentos de feedbacks

[Abra o app público](https://juliana-feedbacks-ia.streamlit.app/)

Um app que lê comentários curtos sobre delivery, em português, e diz se cada um soa positivo, neutro ou negativo, com a pontuação retornada pelo modelo.

O projeto tem duas partes:

1. **O app**: integra um Transformer pré-treinado (da Hugging Face) a uma interface Streamlit.
2. **A avaliação**: mede quanto o modelo acerta e compara com um classificador tradicional (TF-IDF + regressão logística) que eu treinei em um script separado.

## Aviso

Este é um projeto pessoal de estudo, sem vínculo com o iFood. O Transformer usado no app é pré-treinado e foi integrado sem treinamento ou fine-tuning. O único modelo que eu treinei é o TF-IDF + regressão logística, que não faz parte do app.

## Como funciona

1. A pessoa cola um comentário por linha (até 20 comentários, 300 caracteres cada; textos maiores são recusados, não cortados).
2. O app valida a entrada ([`sentiment.py`](./sentiment.py)).
3. O Transformer classifica cada comentário e devolve uma pontuação.
4. O app mostra a contagem por sentimento, um gráfico e uma tabela ([`app.py`](./app.py)).

A pontuação é a confiança estimada pelo modelo, não uma garantia de acerto.

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

### Como ler esta tabela

- **A comparação não é totalmente justa.** O TF-IDF foi treinado com avaliações do B2W e testado no B2W (mesmo tipo de texto). O Transformer não foi treinado nesses dados. Por isso o resultado mostra que um modelo treinado no tipo de dado do teste supera um modelo genérico; não mostra qual seria melhor para delivery.
- O B2W-Reviews01 reúne avaliações de **produtos**, não feedbacks de delivery. As estrelas foram convertidas em sentimento (1–2 negativo, 3 neutro, 4–5 positivo), o que é uma aproximação.
- Os 30 exemplos de delivery foram escritos por mim, com rótulos definidos por uma só pessoa. É um teste pequeno e exploratório.
- O app mantém o Transformer porque uma avaliação específica de feedbacks de delivery, que permitiria escolher o melhor modelo, ainda não foi feita.
- Scripts: [`evaluate.py`](./evaluate.py) (30 exemplos) e [`train_compare.py`](./train_compare.py) (B2W). Resultados completos em [`resultados_avaliacao.json`](./resultados_avaliacao.json) e [`resultados_comparacao.json`](./resultados_comparacao.json).

## O que ainda não foi feito

- Avaliação com um conjunto maior de comentários de delivery, rotulado por mais de uma pessoa.
- Fine-tuning de um modelo com dados de delivery.
- Teste de outros modelos com licença declarada (a licença do modelo usado não consta na ficha dele).

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

São 2 testes leves, que não baixam o modelo: um confere os limites de entrada e outro confere o mapeamento dos rótulos usando um classificador simulado. Eles não testam a qualidade do modelo.

Há um workflow no GitHub Actions ([`.github/workflows/testes.yml`](./.github/workflows/testes.yml)) que executa esses testes em pushes e pull requests.

## Limitações

- O modelo foi treinado em textos de redes sociais, não especificamente para atendimento de delivery.
- Negação, ironia e sentimentos mistos podem causar erros de classificação.
- Nos 30 exemplos de delivery, o modelo errou 11, inclusive frases simples como "A sobremesa estava maravilhosa."
- Na versão hospedada, os comentários enviados pelo navegador vão ao servidor para serem processados.

## Créditos e licença dos dados

Foi usada uma amostra do [B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), com licença **CC BY-NC-SA 4.0**.

> Real, Livy; Oshiro, Marcio; Mafra, Alexandre. *B2W-reviews01: an open product reviews corpus*. STIL, 2019.

A fonte também orienta dar crédito à B2W Digital. A licença permite atribuição e adaptação para uso não comercial, com compartilhamento pela mesma licença; não é uma autorização para uso comercial. Consulte a referência: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[Detalhes técnicos completos](./docs/DETALHES.md)
