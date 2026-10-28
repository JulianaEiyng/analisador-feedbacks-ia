from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sentiment import analyze_comments, load_classifier


EXAMPLES: list[dict[str, str]] = [
    {"comentário": "A comida chegou quente e deliciosa!", "esperado": "Positivo"},
    {"comentário": "O entregador foi muito educado.", "esperado": "Positivo"},
    {"comentário": "Adorei o sabor e a embalagem.", "esperado": "Positivo"},
    {"comentário": "Pedido perfeito, vou pedir novamente.", "esperado": "Positivo"},
    {"comentário": "A entrega foi rápida e tudo veio certo.", "esperado": "Positivo"},
    {"comentário": "Excelente experiência do começo ao fim.", "esperado": "Positivo"},
    {"comentário": "A sobremesa estava maravilhosa.", "esperado": "Positivo"},
    {"comentário": "Muito bom, chegou antes do prazo.", "esperado": "Positivo"},
    {"comentário": "Gostei bastante do atendimento.", "esperado": "Positivo"},
    {"comentário": "A comida estava saborosa e bem feita.", "esperado": "Positivo"},
    {"comentário": "O pedido chegou no horário combinado.", "esperado": "Neutro"},
    {"comentário": "Recebi exatamente o que pedi.", "esperado": "Neutro"},
    {"comentário": "O restaurante fica perto da minha casa.", "esperado": "Neutro"},
    {"comentário": "O pedido foi feito pelo aplicativo.", "esperado": "Neutro"},
    {"comentário": "A embalagem era de papel.", "esperado": "Neutro"},
    {"comentário": "A entrega aconteceu às oito da noite.", "esperado": "Neutro"},
    {"comentário": "Vieram dois pratos e uma bebida.", "esperado": "Neutro"},
    {"comentário": "O entregador deixou o pedido na portaria.", "esperado": "Neutro"},
    {"comentário": "A porção tinha o tamanho informado.", "esperado": "Neutro"},
    {"comentário": "O cardápio informa as opções disponíveis.", "esperado": "Neutro"},
    {"comentário": "A entrega atrasou muito e a comida veio fria.", "esperado": "Negativo"},
    {"comentário": "O pedido veio errado e faltaram itens.", "esperado": "Negativo"},
    {"comentário": "A comida estava horrível e sem sabor.", "esperado": "Negativo"},
    {"comentário": "Não gostei do atendimento.", "esperado": "Negativo"},
    {"comentário": "O entregador foi grosseiro.", "esperado": "Negativo"},
    {"comentário": "A embalagem chegou amassada e vazando.", "esperado": "Negativo"},
    {"comentário": "Esperei uma hora além do prazo.", "esperado": "Negativo"},
    {"comentário": "A pizza estava boa, mas a entrega foi péssima.", "esperado": "Negativo"},
    {"comentário": "Não recomendo, a comida chegou fria.", "esperado": "Negativo"},
    {"comentário": "O preço foi alto e a porção pequena.", "esperado": "Negativo"},
]


def main() -> None:
    classifier = load_classifier()
    predictions = analyze_comments(
        classifier, [example["comentário"] for example in EXAMPLES]
    )
    results: list[dict[str, Any]] = []

    for example, prediction in zip(EXAMPLES, predictions):
        results.append(
            {
                **example,
                "previsto": prediction["sentimento"],
                "pontuação": prediction["pontuação"],
                "acertou": example["esperado"] == prediction["sentimento"],
            }
        )

    acertos = sum(resultado["acertou"] for resultado in results)
    resumo = {
        "total": len(results),
        "acertos": acertos,
        "erros": len(results) - acertos,
        "acuracia": round(acertos / len(results), 4),
        "resultados": results,
    }
    Path("resultados_avaliacao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Acertos: {acertos}/{len(results)} ({resumo['acuracia']:.1%})")
    for resultado in results:
        if not resultado["acertou"]:
            print(
                f"ERRO: {resultado['comentário']} | "
                f"esperado={resultado['esperado']} "
                f"previsto={resultado['previsto']} "
                f"pontuação={resultado['pontuação']}"
            )


if __name__ == "__main__":
    main()
