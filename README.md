# Projeto de Risco de Crédito — SCTEC

## 1. Apresentação

Este projeto foi desenvolvido como parte do projeto final do SCTEC Aperfeiçoar, com o objetivo de aplicar técnicas de Ciência de Dados e Inteligência Artificial na análise de risco de crédito.

A solução utiliza dados históricos de operações de crédito para desenvolver um modelo de Machine Learning capaz de estimar a probabilidade de inadimplência de um cliente.

Além do treinamento e avaliação do modelo, foi desenvolvida uma API para disponibilizar as previsões, permitindo a integração da solução com outros sistemas.

## 2. Objetivo

Desenvolver uma solução de classificação de risco de crédito que permita:

- Analisar características pessoais e financeiras dos solicitantes.
- Identificar padrões associados à inadimplência.
- Estimar a probabilidade de inadimplência.
- Classificar os clientes conforme o resultado do modelo.
- Disponibilizar as previsões por meio de uma API REST.

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

## 5. Etapas do desenvolvimento

### 5.1 Análise exploratória dos dados (EDA)

Foi realizada a análise exploratória da base de crédito, contemplando:

- Análise das variáveis e seus tipos.
- Identificação e tratamento de registros duplicados.
- Avaliação de valores ausentes.
- Identificação de valores inconsistentes.
- Análise da distribuição da variável-alvo.
- Investigação das relações entre as características dos clientes e a inadimplência.

A base original possuía 32.581 registros. Após a remoção de duplicidades, foram mantidos 32.416 registros.

### 5.2 Engenharia de atributos

Foi criada a variável `comprometimento_renda`, calculada pela relação entre o valor solicitado do empréstimo e a renda do cliente.

Essa variável busca representar o comprometimento financeiro do solicitante em relação à sua capacidade de renda.

### 5.3 Pré-processamento

O pré-processamento foi estruturado utilizando pipelines do Scikit-learn.

Foram aplicadas as seguintes técnicas:

- Imputação de valores ausentes pela mediana nas variáveis numéricas.
- Imputação pela categoria mais frequente nas variáveis categóricas.
- Padronização das variáveis numéricas com StandardScaler.
- Codificação das variáveis categóricas com OneHotEncoder.

A divisão dos dados foi realizada em conjuntos de treinamento e teste, na proporção de 80% e 20%, respectivamente, utilizando estratificação da variável-alvo.

### 5.4 Treinamento e seleção do modelo

Foram avaliados modelos de Regressão Logística e Random Forest, considerando estratégias com e sem balanceamento das classes.

O balanceamento foi realizado com RandomOverSampler.

A seleção dos hiperparâmetros considerou validação cruzada estratificada, utilizando o F1-score como métrica principal.

Após as avaliações, foi selecionado um modelo Random Forest com profundidade máxima de 15 e mínimo de 5 amostras por folha.

A escolha considerou o equilíbrio entre desempenho preditivo e controle de sobreajuste.

## 6. Resultados do modelo

O modelo final apresentou os seguintes resultados no conjunto de teste, utilizando threshold de 0,40:

| Métrica | Resultado |
|---|---:|
| Acurácia | 91,86% |
| Precisão | 91,67% |
| Recall | 69,04% |
| F1-score | 78,76% |
| ROC-AUC | 91,86% |

O threshold de 0,40 foi selecionado com base na análise de F1-score em previsões out-of-fold realizadas no conjunto de treinamento.

A alteração do threshold em relação ao valor padrão de 0,50 buscou aumentar a identificação de potenciais casos de inadimplência.

## 7. API de previsão

Foi desenvolvida uma API REST utilizando FastAPI para disponibilizar o modelo treinado.

### Endpoint de saúde

`GET /health`

Verifica se a API está online e se o modelo foi carregado corretamente.

### Endpoint de previsão

`POST /predict`

Recebe os dados de um cliente e retorna:

- Probabilidade de inadimplência.
- Percentual de risco.
- Faixa indicativa de risco.
- Threshold utilizado.
- Classificação prevista.
- Mensagem explicativa.

A API também realiza validações dos dados de entrada utilizando Pydantic.

### Documentação interativa

Após iniciar a API, a documentação Swagger estará disponível em:

`http://127.0.0.1:8000/docs`

## 8. Instalação e execução

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

## 9. Testes automatizados

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

## 10. Considerações e limitações

O modelo fornece estimativas estatísticas de risco com base nos padrões identificados nos dados utilizados no treinamento.

As classificações e faixas de risco são indicativas e não devem ser interpretadas isoladamente como decisão definitiva de concessão de crédito.

Antes de uma utilização em ambiente produtivo, recomenda-se realizar monitoramento de desempenho, avaliação de possíveis vieses, calibração das probabilidades e validação periódica do modelo com dados atualizados.

Também é importante considerar que os resultados obtidos dependem da qualidade, representatividade e atualidade da base utilizada.

## 11. Autor

Stéfano Vicente Laurito

Projeto desenvolvido como parte do SCTEC Aperfeiçoar — Integração de Inteligência Artificial com Dados.

Repositório: https://github.com/stefanolaurito/projeto-risco-credito