# Analisador de sentimentos de feedbacks

Projeto pessoal para estudo de Engenharia de Software com IA. A aplicação
classifica feedbacks curtos de delivery escritos em português como **positivo**,
**neutro** ou **negativo**.

> Este projeto não tem vínculo oficial com o iFood. Eu integrei um modelo
> pré-treinado; não treinei uma IA.

## Objetivo

Praticar a integração de um modelo real de processamento de linguagem natural
em uma aplicação pequena, local e explicável. O usuário informa um comentário
por linha, e a aplicação exibe a classificação, a pontuação retornada pelo
modelo e a contagem por categoria.

## Tecnologias

- Python 3.12.10
- Streamlit
- Hugging Face Transformers
- PyTorch (execução em CPU)
- pandas
- scikit-learn (comparador tradicional)
- Modelo `cardiffnlp/mbert-base-tweet-sentiment-pt`

O modelo tem três rótulos de saída observados na configuração:
`negative`, `neutral` e `positive`, mapeados na aplicação para português. Ele
é voltado a classificação de sentimento em português e foi executado
localmente. Na primeira execução, é necessária uma conexão com a internet para
baixar seus arquivos do Hugging Face. Depois do download, a inferência não
envia os comentários a um serviço externo.

O model card informa que ele é uma versão ajustada de
`bert-base-multilingual-cased` usando o dataset multilíngue de sentimentos da
CardiffNLP, incluindo português. A ficha consultada reporta 61,26% de acurácia
no teste desse dataset. A licença não aparece declarada na ficha/API do modelo;
por isso, ela não é presumida neste projeto e deve ser confirmada antes de uma
redistribuição.

A aplicação usa a revisão `605a6e533ff59d296978653dd34c556e9c9b1671` do modelo,
registrada em [`sentiment.py`](./sentiment.py). A licença não está declarada na
ficha/API desse modelo; por isso, nenhuma licença é atribuída a ele.

Na execução local, o modelo roda no computador do desenvolvedor. Em uma
hospedagem pública, a inferência continua rodando no servidor da aplicação:
os comentários enviados pelo navegador chegam ao servidor para serem
processados. Este código não implementa banco de dados nem registro próprio de
comentários; ainda assim, a política de privacidade e os logs da plataforma
de hospedagem precisam ser verificados antes de uso público.

### Diagnóstico do modelo

Foi feita uma chamada direta ao modelo, sem prefixos nos textos. A configuração
observada confirmou `0=negative`, `1=neutral` e `2=positive`. As probabilidades
retornadas foram:

| Comentário | Negativo | Neutro | Positivo |
|---|---:|---:|---:|
| A comida chegou quente e deliciosa! | 4,82% | 5,17% | 90,02% |
| A entrega atrasou muito e a comida veio fria. | 63,15% | 28,64% | 8,20% |

O resultado negativo do segundo comentário é, portanto, do próprio modelo,
não de um mapeamento invertido ou de uma regra da aplicação.

Também foram comparados diretamente com o modelo dois comentários novos, fora
dos exemplos da interface:

- `O entregador esqueceu metade do pedido.` → **Negativo**, 64,00%
- `A embalagem veio rasgada e a comida derramou.` → **Negativo**, 67,95%

As mesmas previsões apareceram pela interface após o envio do conteúdo atual do
campo. Isso confirma o fluxo de integração, mas não transforma esses dois
casos em uma avaliação independente.

Como comparação exploratória, o modelo
`lxyuan/distilbert-base-multilingual-cased-sentiments-student` também foi
executado nos mesmos 30 exemplos e sem alterar os rótulos esperados. O model
card declara suporte a português e licença Apache-2.0. Ele obteve **11/30
(36,7%)**, contra **19/30 (63,3%)** do modelo atual. Por isso, a alternativa
não foi adotada. Essa comparação pequena não comprova desempenho geral nem
substitui uma avaliação de produção.

## Instalação

No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Execução

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

Insira um comentário por linha e clique em **Analisar feedbacks**. A
aplicação aceita até 20 comentários por análise e até 300 caracteres por
comentário. Linhas vazias são ignoradas; textos acima do limite são recusados
sem corte silencioso. Ao editar a entrada, o resultado anterior é removido
para deixar claro que é necessário analisar novamente.

O botão **Carregar exemplos** preenche quatro exemplos fictícios, sem incluir
rótulos esperados na entrada.

## Avaliação local

Antes da avaliação, foram definidos 30 exemplos fictícios em
[`evaluate.py`](./evaluate.py): 10 positivos, 10 neutros e 10 negativos,
incluindo negação e sentimentos mistos. O comando usado foi:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Resultado real executado em 27/10/2026:

- **19 acertos de 30**
- **11 erros de 30**
- **Acurácia neste conjunto: 63,3%**

Os resultados completos, com esperado, previsto, pontuação e indicação de
acerto, estão em [`resultados_avaliacao.json`](./resultados_avaliacao.json).
Os erros observados foram:

| Comentário | Esperado | Previsto | Pontuação |
|---|---|---|---:|
| Pedido perfeito, vou pedir novamente. | Positivo | Negativo | 0,7917 |
| A entrega foi rápida e tudo veio certo. | Positivo | Neutro | 0,4231 |
| A sobremesa estava maravilhosa. | Positivo | Negativo | 0,4609 |
| Gostei bastante do atendimento. | Positivo | Negativo | 0,5350 |
| Recebi exatamente o que pedi. | Neutro | Negativo | 0,8521 |
| O entregador deixou o pedido na portaria. | Neutro | Negativo | 0,7635 |
| A comida estava horrível e sem sabor. | Negativo | Neutro | 0,4999 |
| O entregador foi grosseiro. | Negativo | Neutro | 0,6534 |
| A embalagem chegou amassada e vazando. | Negativo | Neutro | 0,6032 |
| Esperei uma hora além do prazo. | Negativo | Neutro | 0,4901 |
| O preço foi alto e a porção pequena. | Negativo | Neutro | 0,7775 |

Essa avaliação é pequena, fictícia e orientada a exemplos escolhidos
manualmente. Ela não comprova desempenho em produção. A pontuação exibida é
uma confiança estimada pelo modelo, não uma garantia de acerto.

Os cinco comentários usados no diagnóstico inicial e os quatro exemplos
carregados pela interface são demonstrações/diagnóstico, não uma segunda
avaliação independente.

## Comparação com classificador tradicional

Foi implementada uma experiência separada em
[`train_compare.py`](./train_compare.py). Ela não altera o modelo da aplicação
e não é fine-tuning do Transformer: é o treinamento de um classificador
tradicional de aprendizado de máquina com `TfidfVectorizer` e
`LogisticRegression` dentro de um `Pipeline` do scikit-learn.

### Dados e preparação

Foi usada uma amostra estratificada de 6.000 avaliações do
[B2W-Reviews01](https://github.com/americanas-tech/b2w-reviews01), corpus de
avaliações de produtos em português brasileiro. O README da fonte original
informa a licença **CC BY-NC-SA 4.0**. Isso permite atribuição e adaptação para
uso não comercial, com compartilhamento pela mesma licença; não é uma
autorização para uso comercial. A atribuição usada é:

> Real, Livy; Oshiro, Marcio; Mafra, Alexandre. *B2W-reviews01: an open product
> reviews corpus*. STIL, 2019.

A fonte também orienta dar crédito à B2W Digital. Referência da licença:
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

As notas foram convertidas apenas para esta experiência:

- 1–2 estrelas: Negativo
- 3 estrelas: Neutro
- 4–5 estrelas: Positivo

Essa conversão é uma aproximação: avaliações de produtos pertencem a um
domínio diferente de feedbacks de delivery, e uma nota não representa
perfeitamente o sentimento textual. Antes da divisão, foram removidas linhas
sem texto e duplicatas após normalização de espaços e caixa. A amostra ficou
balanceada, com 2.000 textos por classe. A divisão estratificada foi feita
antes de ajustar o TF-IDF e o classificador:

- Treino: 4.800 textos
- Teste: 1.200 textos
- `test_size=0.2`
- `random_state=42`

O conjunto de teste é diferente dos 30 exemplos fictícios usados na avaliação
anterior. Os exemplos anteriores não foram usados como única evidência de
generalização.

### Resultados reais

Comando executado:

```powershell
.\.venv\Scripts\python.exe train_compare.py
```

Resultados no mesmo conjunto de teste B2W:

| Modelo | Acurácia | Macro-F1 |
|---|---:|---:|
| TF-IDF + regressão logística | 73,50% | 73,27% |
| Transformer atual | 55,42% | 53,66% |

As matrizes de confusão completas estão em
[`resultados_comparacao.json`](./resultados_comparacao.json), na ordem
`Negativo, Neutro, Positivo`:

- TF-IDF + regressão logística:
  `[[344, 51, 5], [67, 244, 89], [25, 81, 294]]`
- Transformer atual:
  `[[318, 75, 7], [189, 112, 99], [97, 68, 235]]`

Neste conjunto exploratório, o classificador tradicional obteve resultado
melhor, mas isso não prova que ele seja melhor para delivery: o domínio é
diferente, os rótulos foram aproximados a partir de estrelas e a amostra é
limitada. Por isso, o Transformer não foi substituído na aplicação. O modelo
tradicional também depende de dados rotulados para ser treinado, enquanto o
Transformer já está pronto para inferência.

## Comandos de desenvolvimento

Testes leves, sem baixar o modelo grande:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

Avaliação dos 30 exemplos fictícios do projeto:

```powershell
.\.venv\Scripts\python.exe evaluate.py
```

Treino e comparação no B2W (baixa o CSV da fonte e carrega o Transformer para
avaliar o mesmo teste):

```powershell
.\.venv\Scripts\python.exe train_compare.py
```

Os testes leves usam um classificador simulado para verificar validações e o
mapeamento da integração, sem inferência real. `evaluate.py`, a interface e
`train_compare.py` executam inferência real; o primeiro uso pode exigir
internet e o Transformer consome mais memória e tempo.

Foi configurado o workflow
[`testes.yml`](./.github/workflows/testes.yml) para executar os testes leves
em pushes e pull requests. Ele não baixa os pesos grandes nem afirma que a
inferência real passou.

## Limitações

- O modelo foi treinado para textos de redes sociais, não especificamente
  para atendimento de delivery.
- Negação, ironia, contexto e sentimentos mistos podem ser classificados
  incorretamente, como evidenciado pelos resultados acima.
- O modelo é executado em CPU e pode levar alguns segundos na primeira análise.
- O modelo é mantido em cache pelo Streamlit durante a execução da aplicação.
- Não há login, banco de dados, API paga, treinamento, regras de palavras-chave
  ou geração automática de respostas.
- O primeiro uso depende de internet para baixar o modelo.
- A versão local e a versão hospedada têm fluxos de dados diferentes: na
  hospedagem, os comentários são enviados ao servidor para inferência.

## Preparação para publicação

O GitHub deve armazenar o código e os arquivos de configuração, não o ambiente
virtual, o cache do Hugging Face ou os pesos do modelo. O arquivo
`.gitignore` exclui esses itens. O modelo será baixado pela aplicação no
primeiro uso do servidor, conforme as dependências instaladas.

Uma opção gratuita compatível a ser avaliada é o **Streamlit Community Cloud**:
ele conecta a um repositório GitHub e executa o `streamlit run app.py` no
servidor, portanto a inferência não deve ser desativada. A documentação oficial
informa que o serviço usa Debian, instala as dependências a partir do
`requirements.txt` e pode impor limites de recursos e atualizações. Também
informa que os aplicativos podem ficar indisponíveis durante manutenção ou
quando excedem limites; a página oficial de status deve ser consultada no dia
da publicação.

O modelo deste projeto ocupa aproximadamente **1,36 GB no cache local apenas
com seus arquivos baixados**, medição feita no computador do desenvolvedor.
Isso não é uma medição de RAM em produção. Como o Community Cloud tem limites
de recursos não garantidos para este projeto, é necessário publicar em um
ambiente de teste e observar o carregamento inicial antes de oferecer o link a
recrutadores. O aplicativo pode ter cold start e suspensão por inatividade
conforme as condições vigentes do serviço. Não foi contratado serviço pago e
nenhum custo foi gerado.

Referências consultadas:

- [Status e limitações do Streamlit Community Cloud](https://github.com/streamlit/docs/blob/main/content/deploy/community-cloud/status-and-limitations.md)
- [Documentação do Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud)
- [Página de status do Streamlit](https://www.streamlitstatus.com/)

O endereço público ainda não foi criado nem testado. Uma publicação só será
considerada concluída depois de confirmar o repositório de destino, fazer o
deploy, abrir o link em janela anônima e testar comentários novos e a largura
de celular.
