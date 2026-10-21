"""
Heart Failure Prediction: Benchmarking Comparativo & Optimización de Umbral Clínico

Pipeline integral de ingeniería de datos, modelado predictivo supervisado
y optimización de umbrales de decisión para detección temprana de insuficiencia cardíaca.
"""

import os
import sys
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)

# Configurar stdout defensivo para consolas Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Configuración estética
sns.set_theme(style="whitegrid", palette="muted")


# ==============================================================================
# 1. PARÁMETROS GLOBALES Y CONFIGURACIÓN
# ==============================================================================
DATA_PATH_DEFAULT: str = os.path.join("data", "heart.csv")
IMG_DIR_DEFAULT: str = "img"

# Semilla fijada: La respuesta del sentido de la vida, el universo y todo lo demás
SEED: int = 42
SEED_TREE: int = 3008

TEST_SIZE: float = 0.2
CLINICAL_MIN_RECALL: float = 0.95


# ==============================================================================
# 2. TIPOLOGÍA, AUDITORÍA Y PREPROCESAMIENTO DE DATOS
# ==============================================================================
def load_data(file_path: str = DATA_PATH_DEFAULT) -> pd.DataFrame:
    """
    Carga el dataset de predicción de fallo cardíaco.

    Args:
        file_path (str): Ruta hacia el archivo heart.csv.

    Returns:
        pd.DataFrame: Copia independiente del DataFrame.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Archivo no encontrado en: {file_path}")

    df: pd.DataFrame = pd.read_csv(file_path)
    return df.copy()


def classify_variables(data: pd.DataFrame) -> dict[str, list[str]]:
    """
    Clasifica las variables clínicas según su naturaleza (cuantitativas vs cualitativas).

    Args:
        data (pd.DataFrame): Dataset a tipificar.

    Returns:
        dict[str, list[str]]: Diccionario estructurado x naturaleza del dato.
    """
    # Clasificación x naturaleza de atributos (cuantitativos vs cualitativos)
    return {
        "cuantitativas_discretas": ["Age", "RestingBP", "Cholesterol", "MaxHR"],
        "cuantitativas_continuas": ["Oldpeak"],
        "cualitativas_nominales": ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"],
        "cualitativas_ordinales": ["ST_Slope"],
        "cualitativas_binarias_target": ["FastingBS", "HeartDisease"]
    }


def audit_anomalies(data: pd.DataFrame) -> pd.DataFrame:
    """
    Audita valores fisiológicamente anómalos (ceros en presión arterial y colesterol).

    Args:
        data (pd.DataFrame): Dataset original.

    Returns:
        pd.DataFrame: Resumen tabulado de anomalías detectadas.
    """
    columnas_analizar: list[str] = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    filas: list[dict[str, object]] = []

    for col in columnas_analizar:
        num_ceros: int = int((data[col] == 0).sum())
        porcentaje_ceros: float = round((num_ceros / len(data)) * 100, 2)
        filas.append({
            "Variable": col,
            "Ceros": num_ceros,
            "Porcentaje (%)": porcentaje_ceros,
            "Inconsistencia_Clinica": "Si" if col in ["RestingBP", "Cholesterol"] and num_ceros > 0 else "No"
        })

    return pd.DataFrame(filas)


def clean_data(data: pd.DataFrame, impute_zeros: bool = True) -> pd.DataFrame:
    """
    Sustituye e imputa ceros no fisiológicos en presión y colesterol.

    Args:
        data (pd.DataFrame): Dataset a limpiar.
        impute_zeros (bool): Activa la imputación mediante la mediana de cohorte.

    Returns:
        pd.DataFrame: DataFrame tratado y limpio.
    """
    df_clean: pd.DataFrame = data.copy()

    if impute_zeros:
        # Reemplazar ceros fisiológicamente imposibles x NaN
        df_clean["RestingBP"] = df_clean["RestingBP"].replace(0, np.nan)
        df_clean["Cholesterol"] = df_clean["Cholesterol"].replace(0, np.nan)

        # Imputar x mediana de la muestra
        med_bp: float = float(df_clean["RestingBP"].median())
        med_chol: float = float(df_clean["Cholesterol"].median())

        df_clean["RestingBP"] = df_clean["RestingBP"].fillna(med_bp)
        df_clean["Cholesterol"] = df_clean["Cholesterol"].fillna(med_chol)

    return df_clean


def prepare_matrices(
    data: pd.DataFrame,
    target_col: str = "HeartDisease",
    test_size: float = TEST_SIZE,
    random_state: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame, pd.Series]:
    """
    Aplica One-Hot Encoding (pd.get_dummies), estratificación y escalado con StandardScaler.

    Args:
        data (pd.DataFrame): Dataset limpio.
        target_col (str): Variable a predecir.
        test_size (float): Fracción de prueba.
        random_state (int): Semilla para split reproducible.

    Returns:
        tuple: (X_train_s, X_test_s, y_train, y_test, feature_names, X, y)
    """
    y: pd.Series = data[target_col].copy()
    X: pd.DataFrame = pd.get_dummies(data.drop(columns=[target_col]), drop_first=True)
    feature_names: list[str] = list(X.columns)

    # División train/test estratificada x distribución de la clase y
    X_train, X_test, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Escalado homogéneo
    scaler: StandardScaler = StandardScaler()
    X_train_s: np.ndarray = scaler.fit_transform(X_train)
    X_test_s: np.ndarray = scaler.transform(X_test)

    y_train: np.ndarray = y_tr.to_numpy()
    y_test: np.ndarray = y_te.to_numpy()

    return X_train_s, X_test_s, y_train, y_test, feature_names, X, y


# ==============================================================================
# 3. ENTRENAMIENTO, EVALUACIÓN Y VALIDACIÓN CRUZADA
# ==============================================================================
def get_models(
    random_state: int = SEED,
    class_weight: str | None = None
) -> list[tuple[str, object]]:
    """
    Retorna 5 modelos estándar de aprendizaje supervisado para clasificación.

    Args:
        random_state (int): Semilla reproducible.
        class_weight (str | None): Ponderación de pesos de clases.

    Returns:
        list[tuple[str, object]]: Modelos listos para entrenamiento.
    """
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1951, class_weight=class_weight, random_state=random_state)),
        ("Decision Tree", DecisionTreeClassifier(class_weight=class_weight, random_state=SEED_TREE)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=SEED_TREE)),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM (RBF)", SVC(probability=True, class_weight=class_weight, random_state=random_state))
    ]


def evaluate_models(
    modelos: list[tuple[str, object]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Entrena los modelos y genera métricas en test y gap de sobreajuste.

    Args:
        modelos (list[tuple[str, object]]): Estimadores a contrastar.
        X_train (np.ndarray): Matriz de entrenamiento escalada.
        X_test (np.ndarray): Matriz de prueba escalada.
        y_train (np.ndarray): Etiquetas de entrenamiento.
        y_test (np.ndarray): Etiquetas de prueba.

    Returns:
        tuple[pd.DataFrame, dict[str, object]]: Métricas y diccionario de modelos ajustados.
    """
    filas: list[dict[str, object]] = []
    trained_dict: dict[str, object] = {}

    for nombre, m in modelos:
        m.fit(X_train, y_train)
        trained_dict[nombre] = m

        y_train_pred: np.ndarray = m.predict(X_train)
        y_test_pred: np.ndarray = m.predict(X_test)

        if hasattr(m, "predict_proba"):
            y_test_proba: np.ndarray = m.predict_proba(X_test)[:, 1]
            auc_val: float = float(roc_auc_score(y_test, y_test_proba))
        else:
            auc_val = 0.0

        train_acc: float = float(accuracy_score(y_train, y_train_pred))
        test_acc: float = float(accuracy_score(y_test, y_test_pred))
        test_rec: float = float(recall_score(y_test, y_test_pred))
        test_prec: float = float(precision_score(y_test, y_test_pred))
        test_f1: float = float(f1_score(y_test, y_test_pred))
        gap: float = float(train_acc - test_acc)

        filas.append({
            "Modelo": nombre,
            "Train_Accuracy": train_acc,
            "Test_Accuracy": test_acc,
            "Recall": test_rec,
            "Precision": test_prec,
            "F1_Score": test_f1,
            "ROC_AUC": auc_val,
            "Overfitting_Gap": gap
        })

    # Ordenar x F1 desc
    df_metrics: pd.DataFrame = pd.DataFrame(filas).sort_values(
        by="F1_Score", ascending=False
    ).reset_index(drop=True)
    return df_metrics, trained_dict


def evaluate_cross_val(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = SEED
) -> pd.DataFrame:
    """
    Ejecuta 5-Fold Stratified Cross Validation con Pipeline para evitar fuga de datos.

    Args:
        X (pd.DataFrame): Datos sin escalar.
        y (pd.Series): Etiquetas verdaderas.
        n_splits (int): Número de pliegues.
        random_state (int): Semilla para barajar los pliegues.

    Returns:
        pd.DataFrame: Métricas promedio de validación cruzada.
    """
    cv: StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    modelos: list[tuple[str, object]] = get_models(random_state=random_state)
    scoring: list[str] = ["accuracy", "recall", "precision", "f1", "roc_auc"]

    cv_filas: list[dict[str, object]] = []

    for nombre, clf in modelos:
        pipeline: Pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

        cv_filas.append({
            "Modelo": nombre,
            "CV_Accuracy": float(scores["test_accuracy"].mean()),
            "CV_Recall": float(scores["test_recall"].mean()),
            "CV_Precision": float(scores["test_precision"].mean()),
            "CV_F1": float(scores["test_f1"].mean()),
            "CV_ROC_AUC": float(scores["test_roc_auc"].mean())
        })

    return pd.DataFrame(cv_filas).sort_values(by="CV_F1", ascending=False).reset_index(drop=True)


# ==============================================================================
# 4. ANÁLISIS DE UMBRALES CLÍNICOS Y MATRIZ DE CONFUSIÓN
# ==============================================================================
def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float
) -> tuple[int, int, int, int, float, float, float, float]:
    """
    Calcula matriz de confusión y métricas según umbral de decisión.

    Permite evaluar el compromiso entre Sensibilidad (Recall) y Precisión.

    Args:
        y_true (np.ndarray): Etiquetas reales.
        y_pred_proba (np.ndarray): Probabilidades predichas de clase positiva.
        threshold (float): Umbral de corte.

    Returns:
        tuple: (tp, tn, fp, fn, accuracy, precision, recall, f1)
    """
    # Predicción x umbral
    y_pred: np.ndarray = (y_pred_proba >= threshold).astype(int)

    # Matriz confusión
    tn_v, fp_v, fn_v, tp_v = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn: int = int(tn_v)
    fp: int = int(fp_v)
    fn: int = int(fn_v)
    tp: int = int(tp_v)

    # Cálculo métricas
    total: int = tp + tn + fp + fn
    accuracy: float = (tp + tn) / total if total > 0 else 0.0
    precision: float = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall: float = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1: float = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return tp, tn, fp, fn, accuracy, precision, recall, f1


def evaluate_threshold_range(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    steps: int = 101
) -> pd.DataFrame:
    """
    Evalúa 101 umbrales equidistantes entre 0.0 y 1.0.

    Args:
        y_true (np.ndarray): Valores verdaderos.
        y_pred_proba (np.ndarray): Probabilidades predichas.
        steps (int): Número de intervalos.

    Returns:
        pd.DataFrame: Métricas x cada umbral evaluado.
    """
    thresholds: np.ndarray = np.linspace(0.0, 1.0, steps)
    filas: list[dict[str, object]] = []

    for t in thresholds:
        tp, tn, fp, fn, acc, prec, rec, f1 = calculate_metrics_at_threshold(
            y_true, y_pred_proba, float(t)
        )
        filas.append({
            "Threshold": round(float(t), 4),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

    return pd.DataFrame(filas)


# ==============================================================================
# 5. VISUALIZACIÓN DESACOPLADA (REGLA 2 Y 7 DEL DECÁLOGO)
# ==============================================================================
def plot_target_distribution(data: pd.DataFrame, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Representa la distribución de la variable objetivo."""
    conteo: pd.Series = data["HeartDisease"].value_counts()
    porcentajes: pd.Series = (data["HeartDisease"].value_counts(normalize=True) * 100).round(2)

    plt.figure(figsize=(8, 5))
    barras = plt.bar(["Normal (0)", "Cardiopatía (1)"], conteo.values, color=["#4C72B0", "#C44E52"], width=0.5, edgecolor="black")

    for i, b in enumerate(barras):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, f"{int(b.get_height())} ({porcentajes.values[i]}%)", ha="center", fontweight="bold")

    plt.title("Distribución de HeartDisease (Comprobación de Balanceo)")
    plt.xlabel("Diagnóstico")
    plt.ylabel("Nº de Pacientes")
    plt.ylim(0, max(conteo.values) + 80)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "01_target_balance_distribution.png"), dpi=300)
    plt.close()


def plot_models_comparison(df_metrics: pd.DataFrame, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Grafica la comparativa de métricas entre modelos."""
    columnas: list[str] = ["Test_Accuracy", "Recall", "Precision", "F1_Score"]
    df_melt: pd.DataFrame = df_metrics.melt(id_vars=["Modelo"], value_vars=columnas, var_name="Metrica", value_name="Valor")

    map_nom: dict[str, str] = {
        "Test_Accuracy": "Exactitud (Accuracy)",
        "Recall": "Exhaustividad (Recall)",
        "Precision": "Precisión",
        "F1_Score": "F1-Score"
    }
    df_melt["Metrica"] = df_melt["Metrica"].map(map_nom)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melt, x="Modelo", y="Valor", hue="Metrica", palette="Blues_r", edgecolor="black")
    plt.title("Benchmarking Comparativo de Clasificadores de Heart Failure")
    plt.xlabel("Modelo Algorítmico")
    plt.ylabel("Puntuación (0.0 - 1.0)")
    plt.ylim(0.65, 1.02)
    plt.xticks(rotation=15)
    plt.legend(title="Métrica Clínica", loc="lower right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "02_models_benchmark_comparison.png"), dpi=300)
    plt.close()


def plot_overfitting(df_metrics: pd.DataFrame, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Visualiza la brecha entre Train y Test Accuracy (Overfitting)."""
    plt.figure(figsize=(10, 5))
    x: np.ndarray = np.arange(len(df_metrics))
    ancho: float = 0.35

    plt.bar(x - ancho/2, df_metrics["Train_Accuracy"], width=ancho, label="Train Accuracy", color="#4C72B0", edgecolor="black")
    plt.bar(x + ancho/2, df_metrics["Test_Accuracy"], width=ancho, label="Test Accuracy", color="#55A868", edgecolor="black")

    plt.title("Análisis de Sesgo vs Varianza: Train vs Test Accuracy")
    plt.xlabel("Modelo")
    plt.ylabel("Exactitud (Accuracy)")
    plt.xticks(x, df_metrics["Modelo"], rotation=15)
    plt.ylim(0.65, 1.05)
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "03_overfitting_gap_analysis.png"), dpi=300)
    plt.close()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, threshold: float = 0.5, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Grafica la matriz de confusión con etiquetas diagnósticas."""
    mat: np.ndarray = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = mat.ravel()

    etiquetas: np.ndarray = np.array([
        [f"TN: {tn}\n(Sano Detectado)", f"FP: {fp}\n(Falsa Alarma)"],
        [f"FN: {fn}\n(Enfermo Omitido)", f"TP: {tp}\n(Diagnóstico Correcto)"]
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, annot=etiquetas, fmt="", cmap="Blues", cbar=False, linewidths=1.0, linecolor="black",
                xticklabels=["Negativo (0)", "Positivo (1)"], yticklabels=["Negativo (0)", "Positivo (1)"])
    plt.title(f"Matriz de Confusión: {model_name} (Umbral = {threshold:.2f})")
    plt.xlabel("Predicción")
    plt.ylabel("Condición Real")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"04_confusion_matrix_{model_name.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def plot_thresholds(threshold_df: pd.DataFrame, model_name: str, opt_t: float, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Grafica curvas Precision, Recall y F1-Score en función del umbral."""
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df["Threshold"], threshold_df["Precision"], label="Precisión", color="#4C72B0", linewidth=2.5)
    plt.plot(threshold_df["Threshold"], threshold_df["Recall"], label="Recall (Sensibilidad)", color="#C44E52", linewidth=2.5)
    plt.plot(threshold_df["Threshold"], threshold_df["F1_Score"], label="F1-Score", color="#55A868", linewidth=2.5, linestyle="--")
    plt.axvline(x=opt_t, color="purple", linestyle=":", linewidth=2.0, label=f"Umbral Clínico Recomendado ({opt_t:.2f})")

    plt.title(f"Optimización de Umbral Clínico: {model_name}")
    plt.xlabel("Umbral de Decisión (Threshold)")
    plt.ylabel("Puntuación de la Métrica")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"05_threshold_curve_{model_name.replace(' ', '_').lower()}.png"), dpi=300)
    plt.close()


def plot_rocs(trained_models: dict[str, object], X_test: np.ndarray, y_test: np.ndarray, output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Grafica la curva ROC comparativa."""
    plt.figure(figsize=(9, 7))
    for nom, m in trained_models.items():
        if hasattr(m, "predict_proba"):
            y_proba = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, label=f"{nom} (AUC = {auc(fpr, tpr):.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Aleatorio (AUC = 0.500)")
    plt.title("Curvas ROC: Capacidad de Discriminación Clínica")
    plt.xlabel("Tasa Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa Verdaderos Positivos (Sensibilidad / Recall)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "06_roc_curves_comparison.png"), dpi=300)
    plt.close()


def plot_features(model: object, feature_names: list[str], output_dir: str = IMG_DIR_DEFAULT) -> None:
    """Grafica los factores de riesgo principales."""
    if hasattr(model, "feature_importances_"):
        val: np.ndarray = model.feature_importances_
    elif hasattr(model, "coef_"):
        val = np.abs(model.coef_[0])
    else:
        return

    df_imp = pd.DataFrame({"Variable": feature_names, "Importancia": val}).sort_values(by="Importancia", ascending=False).head(10)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df_imp, x="Importancia", y="Variable", hue="Variable", legend=False, palette="viridis", edgecolor="black")
    plt.title("Factores de Riesgo Más Determinantes (Importancia Relativa)")
    plt.xlabel("Importancia")
    plt.ylabel("Variable Clínica")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "07_top_risk_factors.png"), dpi=300)
    plt.close()


# ==============================================================================
# 6. ORQUESTADOR PRINCIPAL
# ==============================================================================
def main() -> None:
    """Punto de entrada principal para ejecución autónoma del benchmark."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Heart Failure Classifier Benchmark")
    parser.add_argument("--data", type=str, default=DATA_PATH_DEFAULT, help="Ruta al archivo CSV")
    parser.add_argument("--img-dir", type=str, default=IMG_DIR_DEFAULT, help="Directorio de imágenes")
    args = parser.parse_args()

    print("=" * 80)
    print(" [+] BENCHMARK CLINICO: PREDICCION DE INSUFICIENCIA CARDIACA")
    print("=" * 80)

    # Carga y auditoría
    df_raw: pd.DataFrame = load_data(args.data)
    print(f"\n[1] Dataset cargado: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

    # Auditoría fisiológica
    df_anomalias: pd.DataFrame = audit_anomalies(df_raw)
    for _, fila in df_anomalias.iterrows():
        if fila["Ceros"] > 0 and fila["Inconsistencia_Clinica"] == "Si":
            print(f"    ⚠️  {fila['Variable']}: {fila['Ceros']} registros en 0 ({fila['Porcentaje (%)']}%) - Imputando x mediana...")

    df_clean: pd.DataFrame = clean_data(df_raw, impute_zeros=True)

    # Comprobación balanceo
    cnt = df_clean["HeartDisease"].value_counts()
    pct = (df_clean["HeartDisease"].value_counts(normalize=True) * 100).round(2)
    print(f"\n[2] Comprobación de balanceo: Clase 0 = {cnt[0]} ({pct[0]}%) | Clase 1 = {cnt[1]} ({pct[1]}%)")
    print("    -> Datos naturalmente equilibrados (55.3% / 44.7%). Estratificación aplicada.")

    # Matrices
    X_train_s, X_test_s, y_train, y_test, feat_names, X, y = prepare_matrices(df_clean)

    # Modelos
    modelos = get_models()
    df_metrics, trained_dict = evaluate_models(modelos, X_train_s, X_test_s, y_train, y_test)

    print("\n[3] Resultados del Benchmark:")
    print("-" * 90)
    print(f"{'Modelo':22s} | {'Test Acc':9s} | {'Recall':8s} | {'Precision':10s} | {'F1-Score':9s} | {'ROC-AUC':8s} | {'Overfit':8s}")
    print("-" * 90)
    for _, r in df_metrics.iterrows():
        print(f"{r['Modelo']:22s} | {r['Test_Accuracy']:8.2%} | {r['Recall']:7.2%} | {r['Precision']:9.2%} | {r['F1_Score']:8.2%} | {r['ROC_AUC']:7.2%} | {r['Overfitting_Gap']:+7.2%}")
    print("-" * 90)

    # Mejor modelo
    idx_best = int(df_metrics["F1_Score"].idxmax())
    best_name = str(df_metrics.loc[idx_best, "Modelo"])
    best_model = trained_dict[best_name]
    print(f"\n🏆 Mejor clasificador: {best_name} (F1: {df_metrics.loc[idx_best, 'F1_Score']:.2%})")

    # Validación cruzada
    print("\n[4] 5-Fold Stratified Cross Validation:")
    df_cv = evaluate_cross_val(X, y)
    print(df_cv.to_string(index=False))

    # Análisis de umbral clínico
    print("\n[5] Análisis de Umbral Clínico (Minimización de Falsos Negativos):")
    y_proba = best_model.predict_proba(X_test_s)[:, 1] if hasattr(best_model, "predict_proba") else best_model.decision_function(X_test_s)
    df_th = evaluate_threshold_range(y_test, y_proba)

    # Sweet spot (Recall >= 95%)
    cand = df_th[df_th["Recall"] >= CLINICAL_MIN_RECALL]
    idx_opt = cand["F1_Score"].idxmax() if not cand.empty else df_th["Recall"].idxmax()
    row_opt = df_th.loc[idx_opt]
    opt_t = float(row_opt["Threshold"])

    _, _, fp_def, fn_def, _, prec_def, rec_def, f1_def = calculate_metrics_at_threshold(y_test, y_proba, 0.50)
    print(f"    • Umbral estándar (0.50): Recall: {rec_def:.2%} | Precision: {prec_def:.2%} | Falsos Negativos (Omitidos): {fn_def}")
    print(f"    • Umbral clínico ({opt_t:.2f}):  Recall: {row_opt['Recall']:.2%} | Precision: {row_opt['Precision']:.2%} | Falsos Negativos (Omitidos): {int(row_opt['FN'])}")
    print(f"    💡 Al ajustar a {opt_t:.2f}, los Falsos Negativos bajan de {fn_def} a {int(row_opt['FN'])}, priorizando la salud del paciente.")

    # Visualizaciones
    print(f"\n[6] Generando gráficos en '{args.img_dir}/'...")
    plot_target_distribution(df_clean, args.img_dir)
    plot_models_comparison(df_metrics, args.img_dir)
    plot_overfitting(df_metrics, args.img_dir)
    plot_confusion(y_test, best_model.predict(X_test_s), best_name, threshold=0.50, output_dir=args.img_dir)
    plot_thresholds(df_th, best_name, opt_t, args.img_dir)
    plot_rocs(trained_dict, X_test_s, y_test, args.img_dir)
    if "Random Forest" in trained_dict:
        plot_features(trained_dict["Random Forest"], feat_names, args.img_dir)
    print("    ✅ Gráficos generados correctamente.")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
