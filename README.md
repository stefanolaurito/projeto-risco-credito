# Projeto de Risco de Crédito — SCTEC

## 1. Apresentação

Este projeto foi desenvolvido como parte do projeto final do SCTEC Aperfeiçoar — Integração de Inteligência Artificial com Dados.

A solução utiliza dados históricos de operações de crédito para desenvolver um modelo de Machine Learning capaz de estimar a probabilidade de inadimplência de um cliente.

O projeto contempla as etapas de análise exploratória, tratamento e preparação dos dados, engenharia de atributos, treinamento e avaliação de modelos, seleção de threshold e disponibilização das previsões por meio de uma API REST.

## 2. Objetivo de negócio

Uma instituição financeira precisa avaliar o risco de inadimplência dos solicitantes de crédito, buscando reduzir perdas financeiras e apoiar o processo de análise das operações.

O objetivo deste projeto é desenvolver uma solução preditiva capaz de:

- Analisar características pessoais e financeiras dos solicitantes.
- Identificar padrões associados à inadimplência.
- Estimar a probabilidade de inadimplência.
- Classificar os clientes conforme o resultado do modelo.
- Disponibilizar as previsões por meio de uma API REST.

A solução considera dois tipos de erro:

- **Falso positivo:** um cliente adimplente é classificado como inadimplente, podendo resultar na perda de uma oportunidade de concessão de crédito.
- **Falso negativo:** um cliente inadimplente é classificado como adimplente, podendo resultar em perdas financeiras decorrentes de uma operação de crédito não paga.

A proposta é fornecer uma estimativa estatística que possa apoiar a análise de risco, sem substituir os critérios e as políticas de concessão de crédito da instituição.

## 3. Tecnologias utilizadas

- Python 3.12
- Pandas e NumPy
- Matplotlib e Seaborn
- Scikit-learn
- Imbalanced-learn
- FastAPI
- Pydantic
- Joblib
- Pytest
- Jupyter Notebook

## 4. Estrutura do projeto

```text
projeto-risco-credito/
│
├── app/
│   └── main.py
│
├── base/
│   └── credit_risk_dataset.csv
│
├── models/
│   └── modelo_risco_credito.joblib
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── features.py
│   ├── preprocessing.py
│   └── model_training.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 5. Análise exploratória dos dados (EDA)

A análise exploratória foi realizada sobre a base `credit_risk_dataset.csv`, contemplando dimensões, tipos de dados, estatísticas descritivas, distribuição das variáveis, valores ausentes, duplicidades e relações entre as características dos solicitantes e a inadimplência.

### 5.1 Dimensões e limpeza inicial

A base original possuía 32.581 registros e 12 variáveis.

Durante a análise foram identificados 165 registros duplicados, que foram removidos, resultando em uma base com 32.416 registros.

Também foram identificadas inconsistências em variáveis numéricas, como idades superiores a 100 anos e tempos de emprego superiores a 60 anos. Esses valores foram tratados como ausentes para posterior imputação.

### 5.2 Principais descobertas

**Distribuição da variável-alvo**

A variável `loan_status` apresenta aproximadamente:

- 78,18% de clientes adimplentes.
- 21,82% de clientes inadimplentes.

Essa distribuição evidencia um desbalanceamento entre as classes, justificando a avaliação de estratégias de balanceamento durante o treinamento.

**Valores ausentes**

Após o tratamento das inconsistências, foram identificados valores ausentes principalmente nas seguintes variáveis:

| Variável | Quantidade de ausentes | Percentual aproximado |
|---|---:|---:|
| `loan_int_rate` | 3.095 | 9,55% |
| `person_emp_length` | 889 | 2,74% |
| `person_age` | 5 | 0,02% |

**Finalidade do empréstimo**

A análise das taxas de inadimplência por finalidade do empréstimo identificou diferenças relevantes entre as categorias:

| Finalidade | Taxa de inadimplência aproximada |
|---|---:|
| Debt Consolidation | 28,4% |
| Medical | 26,6% |
| Home Improvement | 26,0% |
| Personal | 19,8% |
| Education | 17,2% |
| Venture | 14,7% |

Os resultados indicam que a finalidade do empréstimo apresenta relação com o comportamento de inadimplência e pode contribuir para a identificação de padrões de risco.

**Classificação de crédito**

Também foi observada uma elevação da taxa de inadimplência conforme a classificação de crédito (`loan_grade`):

| Classificação | Taxa de inadimplência aproximada |
|---|---:|
| A | 9,96% |
| B | 16,32% |
| C | 20,75% |
| D | 59,06% |
| E | 64,49% |
| F | 70,54% |
| G | 98,44% |

As categorias F e G possuem menor quantidade de observações, portanto suas taxas devem ser interpretadas com cautela.

A variável `loan_grade` foi excluída da modelagem por representar uma classificação de crédito potencialmente posterior ao processo de avaliação do solicitante, podendo introduzir vazamento de informação (*data leakage*).

### 5.3 Tratamento de valores ausentes e outliers

Para as variáveis numéricas, foi adotada a imputação pela mediana, estratégia adequada diante da assimetria observada em variáveis como renda e da presença de valores extremos.

Para as variáveis categóricas, foi utilizada a categoria mais frequente.

Valores inconsistentes de idade e tempo de emprego foram convertidos em ausentes antes da imputação.

Valores elevados de renda foram mantidos, pois valores extremos não representam necessariamente erros de registro. Sua remoção poderia eliminar perfis legítimos de solicitantes e reduzir a representatividade da base.

A utilização de modelos baseados em árvores, como Random Forest, também contribui para lidar com relações não lineares e diferentes escalas entre as variáveis.

## 6. Engenharia de atributos (Feature Engineering)

Foi criada a variável `comprometimento_renda`, que representa a relação percentual entre o valor solicitado do empréstimo e a renda anual do solicitante.

A fórmula utilizada foi:

\[
comprometimento\_renda =
\frac{loan\_amnt}{person\_income} \times 100
\]

Essa variável busca representar o comprometimento financeiro do solicitante em relação à sua renda.

A regra foi encapsulada em uma função reutilizável no arquivo `src/features.py`, com validação para evitar divisão por zero e preservar valores ausentes quando a renda não é válida.

A variável `loan_percent_income` foi excluída da modelagem por apresentar redundância com a nova variável criada.

## 7. Pré-processamento e prevenção de Data Leakage

O pré-processamento foi estruturado utilizando componentes do Scikit-learn.

Foram aplicadas as seguintes transformações:

**Variáveis numéricas:**
- Imputação de valores ausentes pela mediana.
- Padronização utilizando `StandardScaler`.

**Variáveis categóricas:**
- Imputação pela categoria mais frequente.
- Codificação utilizando `OneHotEncoder(handle_unknown="ignore")`.

A separação dos dados foi realizada na proporção de 80% para treinamento e 20% para teste, utilizando estratificação da variável-alvo e `random_state=42`.

O pré-processador foi ajustado utilizando somente os dados de treinamento, evitando que informações do conjunto de teste influenciassem as transformações.

O balanceamento das classes foi realizado com `RandomOverSampler`, integrado ao pipeline de treinamento. Dessa forma, a reamostragem ocorre somente nas partições de treinamento durante a validação cruzada, sem modificar o conjunto de teste.

## 8. Treinamento e seleção dos modelos

Foram avaliadas duas famílias de algoritmos:

- Regressão Logística.
- Random Forest.

Também foram comparadas estratégias com e sem balanceamento das classes.

### 8.1 Comparação inicial dos modelos

Os resultados iniciais obtidos foram:

| Modelo | Acurácia | Precisão | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest — baseline | 92,17% | 94,52% | 68,12% | 79,18% | 92,19% |
| Random Forest — balanceado | 91,89% | 90,40% | 70,38% | 79,14% | 92,25% |
| Regressão Logística — baseline | 85,10% | 75,11% | 47,67% | 58,33% | 85,44% |
| Regressão Logística — balanceada | 77,93% | 49,70% | 76,87% | 60,37% | 85,49% |

Os resultados indicaram melhor desempenho geral do Random Forest em relação à Regressão Logística, especialmente considerando F1-score e ROC-AUC.

O balanceamento aumentou o recall da Regressão Logística, mas reduziu sua precisão e acurácia.

### 8.2 Otimização de hiperparâmetros

A otimização foi realizada utilizando validação cruzada estratificada, com F1-score como métrica principal.

Na Regressão Logística, foram avaliados diferentes valores de regularização (`C`).

No Random Forest, foram testadas diferentes profundidades (`max_depth`), além de variações de `min_samples_leaf` e estratégias de balanceamento.

A análise identificou que modelos sem limite de profundidade apresentavam F1 elevado no treinamento, mas uma diferença expressiva em relação à validação cruzada, indicando sobreajuste.

Após o refinamento, foi selecionado o Random Forest com:

| Hiperparâmetro | Valor |
|---|---:|
| `n_estimators` | 100 |
| `max_depth` | 15 |
| `min_samples_leaf` | 5 |
| `min_samples_split` | 2 |
| `class_weight` | None |

A configuração foi escolhida considerando o equilíbrio entre desempenho preditivo e capacidade de generalização, em vez de selecionar exclusivamente a configuração com maior F1-score no treinamento.

## 9. Resultados e avaliação do modelo final

O modelo final foi avaliado no conjunto de teste utilizando threshold de 0,40.

### 9.1 Métricas de avaliação

| Métrica | Resultado |
|---|---:|
| Acurácia | 91,86% |
| Precisão | 91,67% |
| Recall | 69,04% |
| F1-score | 78,76% |
| ROC-AUC | 91,86% |

### 9.2 Matriz de confusão

Com threshold de 0,40, os resultados foram:

| | Previsto adimplente | Previsto inadimplente |
|---|---:|---:|
| Real adimplente | 4.977 | 89 |
| Real inadimplente | 439 | 979 |

A matriz evidencia que o modelo identificou corretamente 979 clientes inadimplentes, enquanto 439 inadimplentes foram classificados como adimplentes.

### 9.3 Análise do threshold

O threshold padrão de 0,50 foi comparado com o valor de 0,40, selecionado a partir das previsões *out-of-fold* realizadas no conjunto de treinamento.

| Métrica | Threshold 0,50 | Threshold 0,40 |
|---|---:|---:|
| Acurácia | 92,04% | 91,86% |
| Precisão | 95,37% | 91,67% |
| Recall | 66,85% | 69,04% |
| F1-score | 78,61% | 78,76% |
| ROC-AUC | 91,86% | 91,86% |

A redução do threshold de 0,50 para 0,40 aumentou a identificação de clientes inadimplentes, reduzindo os falsos negativos de 470 para 439.

Em contrapartida, os falsos positivos aumentaram de 46 para 89.

Essa alteração representa uma escolha de negócio: aceitar um aumento de classificações incorretas de bons pagadores como inadimplentes em troca de identificar mais potenciais inadimplentes.

O threshold de 0,40 foi escolhido por apresentar o melhor F1-score nas previsões *out-of-fold* do treinamento.

### 9.4 Veredito de negócio

O Random Forest foi selecionado para disponibilização na API por apresentar desempenho superior à Regressão Logística nas avaliações realizadas e por oferecer um equilíbrio adequado entre precisão, recall e F1-score após o refinamento dos hiperparâmetros.

A utilização do threshold de 0,40 prioriza uma identificação um pouco maior dos potenciais inadimplentes, considerando o impacto financeiro que a concessão de crédito a clientes que não honram seus compromissos pode representar.

Entretanto, o aumento dos falsos positivos também deve ser considerado, pois pode resultar na perda de oportunidades de negócio.

Os custos financeiros exatos de cada tipo de erro não foram estimados, pois a base utilizada não fornece informações suficientes para essa quantificação.

**Limitação metodológica:** durante o desenvolvimento, o conjunto de teste foi utilizado em comparações de desempenho. Portanto, embora as métricas apresentadas representem a avaliação final do modelo, o conjunto não deve ser considerado uma validação completamente independente de todas as decisões tomadas durante o projeto. Uma avaliação futura com uma base externa e não utilizada no desenvolvimento é recomendada.

## 10. Serialização do modelo

O modelo final foi serializado utilizando `joblib` e armazenado no diretório `models/`.

Arquivo:

```text
models/modelo_risco_credito.joblib
```

O artefato contém o modelo treinado, o threshold definido e metadados relacionados à configuração e aos resultados da avaliação.

A serialização permite carregar o modelo posteriormente sem a necessidade de executar novamente todo o processo de treinamento.

## 11. API de previsão

Foi desenvolvida uma API REST utilizando FastAPI para disponibilizar o modelo treinado.

### 11.1 Endpoint de saúde

`GET /health`

Verifica se a API está online e se o modelo foi carregado corretamente.

### 11.2 Endpoint de previsão

`POST /predict`

Recebe os dados de um cliente e retorna:

- Probabilidade de inadimplência.
- Percentual de risco.
- Faixa indicativa de risco.
- Threshold utilizado.
- Classificação prevista.
- Mensagem explicativa.

A API também realiza validações dos dados de entrada utilizando Pydantic.

### 11.3 Documentação interativa

Após iniciar a API, a documentação Swagger estará disponível em:

`http://127.0.0.1:8000/docs`

## 12. Instalação e execução

### Pré-requisitos

- Python 3.12 ou compatível.
- Git.

### Clonar o repositório

```bash
git clone https://github.com/stefanolaurito/projeto-risco-credito.git
cd projeto-risco-credito
```

### Criar ambiente virtual

```bash
python -m venv .venv
```

### Ativar ambiente virtual

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Executar a API

```bash
python -m uvicorn app.main:app --reload
```

A API ficará disponível em:

`http://127.0.0.1:8000`

A documentação interativa poderá ser acessada em:

`http://127.0.0.1:8000/docs`

## 13. Testes automatizados

O projeto utiliza Pytest para validar o funcionamento da API.

Os testes contemplam:

- Disponibilidade do endpoint de saúde.
- Estrutura da resposta de previsão.
- Coerência entre probabilidade e classificação.
- Rejeição de entradas inválidas.
- Cenário de baixo risco.
- Cenário de alto risco.

Para executar os testes:

```bash
python -m pytest -v
```

Na última execução registrada durante o desenvolvimento, os oito testes automatizados foram aprovados.

## 14. Considerações e limitações

O modelo fornece estimativas estatísticas de risco com base nos padrões identificados nos dados utilizados no treinamento.

As classificações e faixas de risco são indicativas e não devem ser interpretadas isoladamente como decisão definitiva de concessão de crédito.

Antes de uma utilização em ambiente produtivo, recomenda-se:

- Monitoramento periódico de desempenho.
- Avaliação de possíveis vieses.
- Análise da calibração das probabilidades.
- Validação com dados atualizados.
- Revisão periódica do threshold conforme a política de risco da instituição.

Também é importante considerar que os resultados obtidos dependem da qualidade, representatividade e atualidade da base utilizada.

## 15. Autor

**Stéfano Vicente Laurito**

Projeto desenvolvido como parte do SCTEC Aperfeiçoar — Integração de Inteligência Artificial com Dados.

Repositório: https://github.com/stefanolaurito/projeto-risco-credito
