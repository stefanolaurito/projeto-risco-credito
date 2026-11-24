
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# Dados de entrada válidos para teste
cliente_teste = {
    "person_age": 30,
    "person_income": 60000,
    "person_emp_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 10.5,
    "cb_person_cred_hist_length": 8,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "cb_person_default_on_file": "N"
}


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200

    dados = response.json()

    assert dados["status"] == "online"
    assert dados["model_loaded"] is True
    assert dados["threshold"] == 0.4


def test_predict_retorna_resposta_valida():
    response = client.post("/predict", json=cliente_teste)

    assert response.status_code == 200

    dados = response.json()

    campos_esperados = {
        "probabilidade_inadimplencia",
        "percentual_risco",
        "faixa_risco",
        "threshold",
        "classificacao",
        "predicao",
        "mensagem"
    }

    assert campos_esperados.issubset(dados.keys())


def test_predict_probabilidade_e_classificacao():
    response = client.post("/predict", json=cliente_teste)

    assert response.status_code == 200

    dados = response.json()

    probabilidade = dados["probabilidade_inadimplencia"]
    threshold = dados["threshold"]

    assert 0 <= probabilidade <= 1
    assert dados["percentual_risco"] == round(probabilidade * 100, 2)

    if probabilidade >= threshold:
        assert dados["predicao"] == 1
        assert dados["classificacao"] == "Inadimplente"
        assert dados["faixa_risco"] == "Elevado"
    else:
        assert dados["predicao"] == 0
        assert dados["classificacao"] == "Adimplente"


def test_predict_rejeita_idade_invalida():
    cliente_invalido = cliente_teste.copy()
    cliente_invalido["person_age"] = 15

    response = client.post("/predict", json=cliente_invalido)

    assert response.status_code == 422


def test_predict_rejeita_renda_invalida():
    cliente_invalido = cliente_teste.copy()
    cliente_invalido["person_income"] = 0

    response = client.post("/predict", json=cliente_invalido)

    assert response.status_code == 422


def test_predict_rejeita_valor_de_emprestimo_invalido():
    cliente_invalido = cliente_teste.copy()
    cliente_invalido["loan_amnt"] = -1000

    response = client.post("/predict", json=cliente_invalido)

    assert response.status_code == 422



def test_predict_cliente_baixo_risco():
    response = client.post("/predict", json=cliente_teste)

    assert response.status_code == 200

    dados = response.json()

    probabilidade = dados["probabilidade_inadimplencia"]

    assert 0 <= probabilidade < 0.20
    assert dados["threshold"] == 0.4
    assert dados["predicao"] == 0
    assert dados["classificacao"] == "Adimplente"
    assert dados["faixa_risco"] == "Baixo"



def test_predict_cliente_alto_risco():
    cliente_alto_risco = {
        "person_age": 22,
        "person_income": 24000,
        "person_emp_length": 1,
        "loan_amnt": 12000,
        "loan_int_rate": 18.5,
        "cb_person_cred_hist_length": 2,
        "person_home_ownership": "RENT",
        "loan_intent": "DEBTCONSOLIDATION",
        "cb_person_default_on_file": "Y"
    }

    response = client.post("/predict", json=cliente_alto_risco)

    assert response.status_code == 200

    dados = response.json()

    assert dados["probabilidade_inadimplencia"] == 0.9839
    assert dados["threshold"] == 0.4
    assert dados["predicao"] == 1
    assert dados["classificacao"] == "Inadimplente"
    assert dados["faixa_risco"] == "Elevado"
