import pandas as pd
import numpy as np


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria atributos derivados para análise de risco de crédito.
    """
    df = df.copy()

    renda_valida = df["person_income"].notna() & (df["person_income"] > 0)

    df["comprometimento_renda"] = np.nan

    df.loc[renda_valida, "comprometimento_renda"] = (
        df.loc[renda_valida, "loan_amnt"]
        / df.loc[renda_valida, "person_income"]
    ) * 100

    return df