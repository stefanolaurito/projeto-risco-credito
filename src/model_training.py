from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from preprocessing import criar_preprocessador


def criar_pipeline_modelo(modelo, balancear=False):
    """
    Cria pipeline completo de treinamento:
    pré-processamento + balanceamento opcional + modelo.
    """

    etapas = [
        ("preprocessador", criar_preprocessador())
    ]

    if balancear:
        etapas.append(
            ("balanceador", RandomOverSampler(random_state=42))
        )

    etapas.append(("modelo", modelo))

    return ImbPipeline(etapas)


def criar_modelos():
    """
    Define os modelos que serão utilizados nos experimentos.
    """

    modelos = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
    }

    return modelos