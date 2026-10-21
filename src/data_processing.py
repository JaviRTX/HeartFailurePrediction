"""
Módulo de adquisición, preprocesamiento y validación de datos clínicos.

Implementa la carga del dataset de insuficiencia cardíaca, la tipificación
de variables (cuantitativas y cualitativas), la corrección de
inconsistencias fisiológicas y la preparación de matrices estandarizadas.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src import config


def load_heart_data(file_path: str = config.DATA_PATH) -> pd.DataFrame:
    """
    Carga el archivo CSV con los registros de pacientes cardiológicos.

    Args:
        file_path (str): Ruta relativa o absoluta hacia el archivo heart.csv.

    Returns:
        pd.DataFrame: Copia independiente del DataFrame cargado.

    Raises:
        FileNotFoundError: Si la ruta especificada no existe en el sistema.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo de datos no encontrado en: {file_path}")

    # Carga y copia defensiva del dataset
    df_raw: pd.DataFrame = pd.read_csv(file_path)
    return df_raw.copy()


def classify_data_typology(data: pd.DataFrame) -> dict[str, list[str]]:
    """
    Clasifica las variables del dataset según su naturaleza estadística.

    Diferencia entre variables cuantitativas (continuas y discretas) y
    cualitativas (nominales, ordinales y binarias).

    Args:
        data (pd.DataFrame): Dataset con las 12 variables cardiológicas.

    Returns:
        dict[str, list[str]]: Columnas presentes en `data` agrupadas por tipo de dato.
    """
    # Clasificación por naturaleza de atributos (cuantitativos vs cualitativos)
    typology_ref: dict[str, list[str]] = {
        "cuantitativas_discretas": ["Age", "RestingBP", "Cholesterol", "MaxHR"],
        "cuantitativas_continuas": ["Oldpeak"],
        "cualitativas_nominales": ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"],
        "cualitativas_ordinales": ["ST_Slope"],
        "cualitativas_binarias_target": ["FastingBS", "HeartDisease"]
    }
    # Conservar solo las columnas que existen en el dataset recibido
    typology_dict: dict[str, list[str]] = {
        tipo: [col for col in columnas if col in data.columns]
        for tipo, columnas in typology_ref.items()
    }
    return typology_dict


def audit_clinical_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta valores fisiológicamente inverosímiles (ceros en presión y colesterol).

    Args:
        data (pd.DataFrame): Dataset a auditar.

    Returns:
        pd.DataFrame: Tabla resumen con recuento y porcentaje de ceros por columna.
    """
    # Recuento de ceros por columna numérica
    numeric_cols: list[str] = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    anomalies: list[dict[str, object]] = []

    for col in numeric_cols:
        zeros_count: int = int((data[col] == 0).sum())
        zeros_pct: float = round((zeros_count / len(data)) * 100, 2)
        anomalies.append({
            "Variable": col,
            "Ceros": zeros_count,
            "Porcentaje (%)": zeros_pct,
            "Anomalia_Fisiologica": "Si" if col in ["RestingBP", "Cholesterol"] and zeros_count > 0 else "No"
        })

    # Pasar a DataFrame con nombres de columna claros
    df_anomalies: pd.DataFrame = pd.DataFrame(anomalies)
    return df_anomalies


def clean_clinical_data(data: pd.DataFrame, impute_zeros: bool = True) -> pd.DataFrame:
    """
    Trata las anomalías clínicas de RestingBP y Cholesterol.

    En medicina, valores de presión arterial sistólica o colesterol en 0
    representan valores no reportados (datos faltantes en centros clínicos).

    Args:
        data (pd.DataFrame): Dataset original.
        impute_zeros (bool): Si es True, sustituye 0 por NaN e imputa con mediana.

    Returns:
        pd.DataFrame: Dataset limpio con tipos e imputación coherentes.
    """
    df_clean: pd.DataFrame = data.copy()

    if impute_zeros:
        # Reemplazar ceros no fisiológicos por NaN
        df_clean["RestingBP"] = df_clean["RestingBP"].replace(0, np.nan)
        df_clean["Cholesterol"] = df_clean["Cholesterol"].replace(0, np.nan)

        # Imputar por mediana de la cohorte para robustez ante outliers
        mediana_bp: float = float(df_clean["RestingBP"].median())
        mediana_chol: float = float(df_clean["Cholesterol"].median())

        df_clean["RestingBP"] = df_clean["RestingBP"].fillna(mediana_bp)
        df_clean["Cholesterol"] = df_clean["Cholesterol"].fillna(mediana_chol)

    return df_clean


def prepare_features(
    data: pd.DataFrame,
    target_col: str = "HeartDisease"
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convierte variables categoricas a numericas con one-hot encoding (pd.get_dummies).

    Args:
        data (pd.DataFrame): Dataset con variables clinicas limpias.
        target_col (str): Nombre de la variable objetivo.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Matriz de caracteristicas X y vector target y.
    """
    # Separar variable dependiente e independientes
    y: pd.Series = data[target_col].copy()
    features: pd.DataFrame = data.drop(columns=[target_col]).copy()

    # Codificar columnas categóricas con drop_first=True para evitar colinealidad
    X: pd.DataFrame = pd.get_dummies(features, drop_first=True)

    return X, y


def split_and_scale_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """
    Divide los datos en train/test estratificado y estandariza las escalas.

    El escalado con StandardScaler es imprescindible para algoritmos sensibles
    a distancias y magnitudes (Regresion Logistica, SVM, KNN).

    Args:
        X (pd.DataFrame): Matriz de predictores.
        y (pd.Series): Vector de etiquetas reales.
        test_size (float): Proporcion del conjunto de prueba.
        random_state (int): Semilla para garantizar reproducibilidad.

    Returns:
        tuple: (X_train_s, X_test_s, y_train, y_test, scaler, feature_names)
    """
    # Guardar nombres de caracteristicas
    feature_names: list[str] = list(X.columns)

    # Division train/test estratificada segun distribucion de la clase y
    X_train, X_test, y_train_split, y_test_split = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Conserva la proporción de clases en train y test
    )

    # Escalado homogeneo
    scaler: StandardScaler = StandardScaler()
    X_train_s: np.ndarray = scaler.fit_transform(X_train)
    X_test_s: np.ndarray = scaler.transform(X_test)

    # Convertir etiquetas a arrays numpy tipados
    y_train: np.ndarray = y_train_split.to_numpy()
    y_test: np.ndarray = y_test_split.to_numpy()

    return X_train_s, X_test_s, y_train, y_test, scaler, feature_names
