from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


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
# SCHEMA DE ENTRADA
# =========================================================

class Cliente(BaseModel):
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(gt=0)
    person_emp_length: float | None = Field(default=None, ge=0, le=60)

    loan_amnt: float = Field(gt=0)
    loan_int_rate: float | None = Field(default=None, ge=0)

    cb_person_cred_hist_length: float = Field(ge=0)

    person_home_ownership: str
    loan_intent: str
    cb_person_default_on_file: str


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

    # Engenharia de atributo
    dados["comprometimento_renda"] = (
        dados["loan_amnt"] / dados["person_income"]
    ) * 100

    entrada = pd.DataFrame([dados])

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

    return {
        "probabilidade_inadimplencia": round(probabilidade, 4),
        "percentual_risco": round(probabilidade * 100, 2),
        "faixa_risco": faixa_risco,
        "threshold": THRESHOLD,
        "classificacao": classificacao,
        "predicao": previsao,
        "mensagem": (
            f"Cliente classificado como {classificacao.lower()}, "
            f"com risco indicativo {faixa_risco.lower()}."
        )
    }