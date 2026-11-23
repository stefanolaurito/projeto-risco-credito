from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Variáveis numéricas
VARIAVEIS_NUMERICAS = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
    "comprometimento_renda",
]

# Variáveis categóricas
VARIAVEIS_CATEGORICAS = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]


def criar_preprocessador():
    """
    Cria um pipeline de pré-processamento para variáveis
    numéricas e categóricas.
    """

    # Pipeline para variáveis numéricas
    pipeline_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("escalonador", StandardScaler()),
    ])

    # Pipeline para variáveis categóricas
    pipeline_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Unifica os dois tratamentos
    preprocessador = ColumnTransformer([
        ("numericas", pipeline_numerico, VARIAVEIS_NUMERICAS),
        ("categoricas", pipeline_categorico, VARIAVEIS_CATEGORICAS),
    ])

    return preprocessador