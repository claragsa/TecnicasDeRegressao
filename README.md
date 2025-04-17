# Técnicas de Regressão com Parâmetros Pouco Relacionais

Este projeto tem como objetivo entender como técnicas de regressão em Machine Learning se comportam ao tentar prever uma variável-alvo (neste caso, o preço de softwares) a partir de atributos que não apresentam correlações evidentes entre si. O dataset utilizado contém informações de softwares mais vendidos da Amazon, como ranking, número de estrelas e quantidade de avaliações.

## Fonte dos Dados

Os dados utilizados neste projeto foram obtidos do repositório público do Kaggle: [Amazon Best Seller Software](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-best-seller-software).

## Etapas do Processo

- **Carregamento dos dados**: Foi desenvolvida uma função para facilitar o processo de extração e carregamento do dataset.
- **Tratamento dos dados**:
  - A coluna de preços foi convertida para valores numéricos, removendo o símbolo `$`.
  - Valores nulos foram identificados e excluídos.
- **Análise exploratória**:
  - Investigação da origem dos produtos.
  - Análise da distribuição das avaliações.
  - Identificação e remoção de outliers.
  - Análise de correlação entre as variáveis (scatter plots e heatmap).
- **Modelagem preditiva**:
  - Separação dos dados em treino e teste.
  - Aplicação de três modelos de regressão: Linear Regression, Decision Tree Regressor e Random Forest Regressor.
- **Otimização**:
  - O modelo com melhor desempenho (Árvore de Decisão) foi otimizado com GridSearchCV e RandomizedSearchCV.
  - Os dois métodos de otimização foram comparados.
- **Exportação dos modelos**:
  - Todos os modelos foram salvos utilizando a biblioteca `joblib`.

## Resultados

Abaixo, os resultados de desempenho obtidos por cada modelo:

| Modelo                   | RMSE   | R²     |
|--------------------------|--------|--------|
| Linear Regression        | 973.31 | 4.42%  |
| Decision Tree Regressor | 505.76 | 74.25% |
| Random Forest Regressor | 546.08 | 69.97% |
| GridSearchCV (DT)        | 526.78 | 72.06% |
| RandomSearchCV (DT)      | 773.26 | 39.80% |

> A Árvore de Decisão simples (sem hiperparâmetros otimizados) apresentou o melhor desempenho geral.

## Conclusão

Mesmo com atributos que apresentam baixa correlação entre si, foi possível construir um modelo de regressão com desempenho satisfatório. A Árvore de Decisão superou os demais modelos em termos de RMSE e R².

Curiosamente, os modelos ajustados com GridSearchCV e RandomSearchCV não superaram o modelo inicial, o que mostra que a escolha e combinação dos hiperparâmetros nem sempre resultam em melhorias significativas. Além disso:

- **RandomizedSearchCV** se mostrou mais eficiente em tempo de execução por testar menos combinações.
- **GridSearchCV**, por testar todas as combinações possíveis, pode levar à sobreajuste ou resultados piores dependendo da faixa testada.

## Arquivos

- `tecnicas_regressao_script.py`: script com o código completo
- `modelos_salvos/`: diretório com os modelos treinados salvos.
- `graficos_tecnicasreg.py`: visualização dos resultados.
- `grafico_tecnicas_regressao.png`: imagem dos gráficos gerados 

---


