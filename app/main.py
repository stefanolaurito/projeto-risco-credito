from pathlib import Path
from src.features import criar_features

import joblib
import pandas as pd

from fastapi import FastAPI
from app.schemas import Cliente

# =========================================================
# CONFIGURAÇÃO E CARREGAMENTO DO MODELO
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "modelo_risco_credito.joblib"

artefato = joblib.load(MODEL_PATH)

modelo = artefato["model"]
THRESHOLD = artefato["threshold"]


# =========================================================
# API
# =========================================================

app = FastAPI(
    title="API de Risco de Crédito",
    description="API de classificação de risco de inadimplência.",
    version="1.0.0"
)


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": modelo is not None,
        "threshold": THRESHOLD
    }


# =========================================================
# PREDIÇÃO
# =========================================================

@app.post("/predict")
def predict(cliente: Cliente):

    dados = cliente.model_dump()

    # Engenharia de atributos
    entrada = pd.DataFrame([dados])
    entrada = criar_features(entrada)

    # Probabilidade de inadimplência
    probabilidade = float(
        modelo.predict_proba(entrada)[0][1]
    )

    # Decisão do modelo
    previsao = int(probabilidade >= THRESHOLD)

    classificacao = (
        "Inadimplente"
        if previsao == 1
        else "Adimplente"
    )

    # Faixa indicativa de risco
    if probabilidade < 0.20:
        faixa_risco = "Baixo"
    elif probabilidade < THRESHOLD:
        faixa_risco = "Moderado"
    else:
        faixa_risco = "Elevado"

    acao_recomendada = (
        "Encaminhar para análise de crédito aprofundada"
        if previsao == 1
        else "Prosseguir para avaliação conforme política de crédito"
    )

    return {
        "probabilidade_inadimplencia": round(probabilidade, 4),
        "percentual_risco": round(probabilidade * 100, 2),
        "faixa_risco": faixa_risco,
        "threshold": THRESHOLD,
        "classificacao": classificacao,
        "predicao": previsao,
        "acao_recomendada": acao_recomendada,
        "mensagem": (
            f"Cliente classificado como {classificacao.lower()}, "
            f"com risco indicativo {faixa_risco.lower()}."
        )
    }