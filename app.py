from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sentiment import (
    MAX_COMMENT_LENGTH,
    MAX_COMMENTS,
    MODEL_NAME,
    analyze_comments,
    load_classifier,
    validate_comments,
)


EXAMPLE_COMMENTS = "\n".join(
    (
        "A comida chegou quente e deliciosa!",
        "O pedido chegou no horário combinado.",
        "A entrega atrasou muito e a comida veio fria.",
        "A pizza estava boa, mas a entrega foi péssima.",
    )
)


def carregar_estilos() -> None:
    css = Path(__file__).with_name("style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def carregar_exemplos() -> None:
    st.session_state.comentarios = EXAMPLE_COMMENTS
    st.session_state.resultados = []


def limpar_resultados() -> None:
    st.session_state.resultados = []


def mostrar_resumo(resultados: list[dict[str, str]]) -> None:
    tabela = pd.DataFrame(resultados)
    contagens = (
        tabela["sentimento"]
        .value_counts()
        .reindex(["Positivo", "Neutro", "Negativo"], fill_value=0)
    )

    st.markdown("#### Resumo da análise")
    colunas = st.columns(3)
    for coluna, sentimento, icone in zip(
        colunas,
        ("Positivo", "Neutro", "Negativo"),
        ("🟢", "🟡", "🔴"),
    ):
        with coluna:
            st.metric(f"{icone} {sentimento}", int(contagens[sentimento]))

    st.markdown("#### Distribuição dos sentimentos")
    dados_grafico = pd.DataFrame(
        {
            "sentimento": contagens.index,
            "comentários": contagens.values,
        }
    )
    st.vega_lite_chart(
        dados_grafico,
        {
            "mark": {"type": "bar", "cornerRadiusEnd": 4},
            "encoding": {
                "y": {
                    "field": "sentimento",
                    "type": "nominal",
                    "sort": ["Positivo", "Neutro", "Negativo"],
                    "title": None,
                },
                "x": {
                    "field": "comentários",
                    "type": "quantitative",
                    "title": "Comentários",
                },
                "color": {
                    "field": "sentimento",
                    "type": "nominal",
                    "scale": {
                        "domain": ["Positivo", "Neutro", "Negativo"],
                        "range": ["#56D364", "#F2C94C", "#FF4D6D"],
                    },
                    "legend": None,
                },
                "tooltip": [
                    {"field": "sentimento", "title": "Sentimento"},
                    {"field": "comentários", "title": "Comentários"},
                ],
            },
            "height": 150,
            "width": "container",
            "background": "transparent",
            "config": {
                "axis": {
                    "labelColor": "#C4C4CC",
                    "titleColor": "#C4C4CC",
                    "gridColor": "#2B2B34",
                }
            },
        },
    )


def main() -> None:
    st.set_page_config(
        page_title="Feedbacks de Delivery com IA",
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    carregar_estilos()

    if "comentarios" not in st.session_state:
        st.session_state.comentarios = ""
    if "resultados" not in st.session_state:
        st.session_state.resultados = []

    st.markdown(
        """
        <header class="app-header">
            <div class="brand">Feedbacks <span>IA</span></div>
            <div class="eyebrow">PROJETO DE PORTFÓLIO</div>
        </header>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">ANÁLISE DE SENTIMENTOS</div>
            <h1>Transforme comentários em insights.</h1>
            <p>Explore sentimentos em feedbacks de delivery com inteligência artificial.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    coluna_entrada, coluna_resumo = st.columns([1.08, 0.92], gap="large")
    with coluna_entrada:
        with st.container(border=True):
            st.markdown("### Analise seus feedbacks")
            st.write("Cole comentários curtos e deixe um comentário por linha.")
            st.text_area(
                "Comentários de delivery",
                height=230,
                key="comentarios",
                on_change=limpar_resultados,
                label_visibility="visible",
                placeholder="Um comentário por linha",
            )
            st.caption(
                f"Até {MAX_COMMENTS} comentários · "
                f"{MAX_COMMENT_LENGTH} caracteres por comentário"
            )
            botoes = st.columns([1, 1.35], gap="small")
            with botoes[0]:
                st.button(
                    "Carregar exemplos",
                    on_click=carregar_exemplos,
                    width="stretch",
                )
            with botoes[1]:
                analisar = st.button(
                    "Analisar feedbacks",
                    type="primary",
                    width="stretch",
                )

    if analisar:
        comentarios = [
            linha.strip()
            for linha in st.session_state.comentarios.splitlines()
            if linha.strip()
        ]
        try:
            validate_comments(comentarios)
        except ValueError as error:
            st.error(str(error))
        else:
            with st.spinner("O modelo está analisando seus feedbacks..."):
                classificador = load_classifier()
                st.session_state.resultados = analyze_comments(
                    classificador,
                    comentarios,
                )

    with coluna_resumo:
        with st.container(border=True):
            if st.session_state.resultados:
                mostrar_resumo(st.session_state.resultados)
            else:
                st.markdown("### Sua análise começa aqui")
                st.info(
                    "Carregue exemplos ou escreva seus próprios comentários. "
                    "O resumo e o gráfico aparecerão depois da análise."
                )

    if st.session_state.resultados:
        st.markdown("### Resultados detalhados")
        tabela = pd.DataFrame(st.session_state.resultados)
        st.dataframe(
            tabela,
            width="stretch",
            hide_index=True,
            column_config={
                "comentário": st.column_config.TextColumn("Comentário"),
                "sentimento": st.column_config.TextColumn("Sentimento"),
                "pontuação": st.column_config.TextColumn("Pontuação"),
            },
        )
        st.caption(
            "A pontuação é uma confiança estimada pelo modelo e não garante "
            "que a classificação esteja correta."
        )

    with st.expander("Sobre a IA"):
        st.write(
            f"Modelo de análise de sentimentos: `{MODEL_NAME}`. "
            "Ele é executado com Transformers e PyTorch. Na primeira execução, "
            "os arquivos do modelo são baixados do Hugging Face."
        )
        st.write(
            "A avaliação exploratória deste projeto teve 19 acertos em 30 "
            "exemplos fictícios. Esse resultado não comprova desempenho em produção."
        )
        st.write(
            "Negação, ironia, contexto e sentimentos mistos podem ser classificados "
            "incorretamente. A licença do modelo não está declarada na ficha/API. "
            "Na versão hospedada, os comentários são enviados ao servidor da "
            "aplicação para inferência; este código não implementa banco de dados "
            "nem registro próprio de comentários."
        )

    st.markdown(
        '<footer class="app-footer">Projeto pessoal de estudo, sem vínculo com o iFood.</footer>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
