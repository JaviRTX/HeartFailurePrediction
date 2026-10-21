"""
Módulo de visualización gráfica desacoplada y exportación de figuras.

Sigue estrictamente el principio de separación de responsabilidades:
las funciones de este módulo reciben datos previamente calculados y
se encargan exclusivamente del renderizado estilizado, anotaciones y guardado en disco.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from src import config

# Configuración estética global
sns.set_theme(style="whitegrid", palette="muted")


def plot_target_distribution(
    data: pd.DataFrame,
    target_col: str = "HeartDisease",
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Representa la distribución de la variable objetivo y comprueba balanceo.

    Args:
        data (pd.DataFrame): Dataset con la columna objetivo.
        target_col (str): Nombre de la variable dependiente.
        output_dir (str): Directorio donde guardar la imagen.

    Returns:
        None: Guarda la gráfica en disco.
    """
    conteo: pd.Series = data[target_col].value_counts()
    porcentajes: pd.Series = (data[target_col].value_counts(normalize=True) * 100).round(2)

    labels: list[str] = ["Normal (0)", "Cardiopatía (1)"]
    colores: list[str] = ["#4C72B0", "#C44E52"]

    plt.figure(figsize=(8, 5))
    barras = plt.bar(labels, conteo.values, color=colores, width=0.5, edgecolor="black", alpha=0.85)

    # Anotaciones con recuento y porcentaje sobre cada barra
    for i, barra in enumerate(barras):
        altura: float = barra.get_height()
        pct: float = porcentajes.values[i]
        plt.text(
            barra.get_x() + barra.get_width() / 2.0,
            altura + 10,
            f"{int(altura)} ({pct}%)",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11
        )

    plt.title("Distribución de la Variable Objetivo: HeartDisease (Comprobación de Balanceo)")
    plt.xlabel("Diagnóstico Cardiológico")
    plt.ylabel("Número de Pacientes")
    plt.ylim(0, max(conteo.values) + 80)
    plt.tight_layout()

    # Guardar gráfica en directorio img
    os.makedirs(output_dir, exist_ok=True)
    ruta_salida: str = os.path.join(output_dir, "01_target_balance_distribution.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Compara las metricas clave (Accuracy, Recall, Precision, F1-Score) x modelo.

    Args:
        metrics_df (pd.DataFrame): Métricas calculadas x evaluate_models.
        output_dir (str): Directorio de guardado.

    Returns:
        None: Guarda la imagen comparativa.
    """
    # Transformar a formato largo para seaborn
    columnas_plot: list[str] = ["Test_Accuracy", "Recall", "Precision", "F1_Score"]
    df_melt: pd.DataFrame = metrics_df.melt(
        id_vars=["Modelo"],
        value_vars=columnas_plot,
        var_name="Metrica",
        value_name="Valor"
    )

    # Renombrar métricas para presentación limpia
    nombres_metricas: dict[str, str] = {
        "Test_Accuracy": "Exactitud (Accuracy)",
        "Recall": "Exhaustividad (Recall)",
        "Precision": "Precisión",
        "F1_Score": "F1-Score"
    }
    df_melt["Metrica"] = df_melt["Metrica"].map(nombres_metricas)

    plt.figure(figsize=(12, 6))
    grafico = sns.barplot(
        data=df_melt,
        x="Modelo",
        y="Valor",
        hue="Metrica",
        palette="Blues_r",
        edgecolor="black",
        alpha=0.9
    )

    plt.title("Benchmarking Comparativo de Clasificadores de Heart Failure")
    plt.xlabel("Modelo Algorítmico")
    plt.ylabel("Puntuación (0.0 - 1.0)")
    plt.ylim(0.65, 1.02)
    plt.xticks(rotation=15)
    plt.legend(title="Métrica Clínica", loc="lower right", frameon=True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ruta_salida: str = os.path.join(output_dir, "02_models_benchmark_comparison.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_overfitting_analysis(
    metrics_df: pd.DataFrame,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Analiza la brecha entre entrenamiento y prueba para detectar sobreajuste.

    Args:
        metrics_df (pd.DataFrame): Dataset con Train_Accuracy y Test_Accuracy.
        output_dir (str): Directorio de guardado.

    Returns:
        None: Guarda la figura.
    """
    plt.figure(figsize=(10, 5))
    x_indices: np.ndarray = np.arange(len(metrics_df))
    ancho_barra: float = 0.35

    plt.bar(
        x_indices - ancho_barra / 2,
        metrics_df["Train_Accuracy"],
        width=ancho_barra,
        label="Train Accuracy",
        color="#4C72B0",
        alpha=0.85,
        edgecolor="black"
    )
    plt.bar(
        x_indices + ancho_barra / 2,
        metrics_df["Test_Accuracy"],
        width=ancho_barra,
        label="Test Accuracy",
        color="#55A868",
        alpha=0.85,
        edgecolor="black"
    )

    plt.title("Análisis de Sesgo vs Varianza: Train vs Test Accuracy (Overfitting)")
    plt.xlabel("Modelo")
    plt.ylabel("Exactitud (Accuracy)")
    plt.xticks(x_indices, metrics_df["Modelo"], rotation=15)
    plt.ylim(0.65, 1.05)
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ruta_salida: str = os.path.join(output_dir, "03_overfitting_gap_analysis.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    threshold: float = config.DEFAULT_THRESHOLD,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Grafica la matriz de confusión detallando TP, TN, FP y FN.

    Args:
        y_true (np.ndarray): Etiquetas verdaderas.
        y_pred (np.ndarray): Predicciones binarias.
        model_name (str): Nombre del modelo para el título.
        threshold (float): Umbral aplicado.
        output_dir (str): Directorio de destino.

    Returns:
        None: Guarda la figura.
    """
    matriz: np.ndarray = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matriz.ravel()

    etiquetas_anotadas: np.ndarray = np.array([
        [f"TN: {tn}\n(Sano Detectado)", f"FP: {fp}\n(Falsa Alarma)"],
        [f"FN: {fn}\n(Enfermo Omitido)", f"TP: {tp}\n(Diagnóstico Correcto)"]
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matriz,
        annot=etiquetas_anotadas,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Negativo (0)", "Positivo (1)"],
        yticklabels=["Negativo (0)", "Positivo (1)"],
        linewidths=1.0,
        linecolor="black"
    )

    plt.title(f"Matriz de Confusión: {model_name} (Umbral = {threshold:.2f})")
    plt.xlabel("Predicción del Modelo")
    plt.ylabel("Condición Real del Paciente")
    plt.tight_layout()

    nombre_archivo: str = f"04_confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    ruta_salida: str = os.path.join(output_dir, nombre_archivo)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_threshold_curves(
    threshold_df: pd.DataFrame,
    model_name: str,
    optimal_threshold: float,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Representa las curvas continuas de Precision, Recall y F1-Score en funcion del umbral.

    Visualiza el compromiso entre métricas para selección informada del punto de corte.

    Args:
        threshold_df (pd.DataFrame): Métricas x rango de umbrales.
        model_name (str): Nombre del modelo evaluado.
        optimal_threshold (float): Umbral clínico recomendado.
        output_dir (str): Directorio de guardado.

    Returns:
        None: Genera y almacena el gráfico.
    """
    plt.figure(figsize=(10, 6))

    # Curvas de métricas diagnósticas
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Precision"],
        label="Precisión",
        color="#4C72B0",
        linewidth=2.5
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["Recall"],
        label="Recall (Sensibilidad)",
        color="#C44E52",
        linewidth=2.5
    )
    plt.plot(
        threshold_df["Threshold"],
        threshold_df["F1_Score"],
        label="F1-Score",
        color="#55A868",
        linewidth=2.5,
        linestyle="--"
    )

    # Línea vertical del umbral clínico óptimo
    plt.axvline(
        x=optimal_threshold,
        color="purple",
        linestyle=":",
        linewidth=2.0,
        label=f"Umbral Clínico Recomendado ({optimal_threshold:.2f})"
    )

    plt.title(f"Optimización de Umbral Clínico: {model_name}")
    plt.xlabel("Umbral de Decisión (Threshold)")
    plt.ylabel("Puntuación de la Métrica")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    nombre_archivo: str = f"05_threshold_curve_{model_name.replace(' ', '_').lower()}.png"
    ruta_salida: str = os.path.join(output_dir, nombre_archivo)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_roc_curves(
    trained_models: dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Grafica la curva ROC (Receiver Operating Characteristic) para todos los modelos.

    Args:
        trained_models (dict[str, object]): Diccionario con clasificadores entrenados.
        X_test (np.ndarray): Matriz de prueba escalada.
        y_test (np.ndarray): Vector de etiquetas reales.
        output_dir (str): Directorio de guardado.

    Returns:
        None: Guarda la curva ROC.
    """
    plt.figure(figsize=(9, 7))

    for nombre, modelo in trained_models.items():
        if hasattr(modelo, "predict_proba"):
            y_proba: np.ndarray = modelo.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            puntuacion_auc: float = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{nombre} (AUC = {puntuacion_auc:.3f})", linewidth=2)

    # Diagonal de referencia aleatoria
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Clasificador Aleatorio (AUC = 0.500)")

    plt.title("Curvas ROC: Capacidad de Discriminación Clínica")
    plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
    plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad / Recall)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ruta_salida: str = os.path.join(output_dir, "06_roc_curves_comparison.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()


def plot_feature_importance(
    model: object,
    feature_names: list[str],
    model_name: str,
    top_n: int = 10,
    output_dir: str = config.IMG_DIR
) -> None:
    """
    Grafica los predictores clínicos más determinantes según el modelo.

    Args:
        model (object): Modelo con coef_ o feature_importances_.
        feature_names (list[str]): Nombres de las características.
        model_name (str): Nombre del modelo.
        top_n (int): Número de predictores principales a visualizar.
        output_dir (str): Directorio de guardado.

    Returns:
        None: Guarda el gráfico.
    """
    if hasattr(model, "feature_importances_"):
        valores: np.ndarray = model.feature_importances_
        nombre_eje: str = "Importancia Relativa (Gini)"
    elif hasattr(model, "coef_"):
        valores = np.abs(model.coef_[0])
        nombre_eje = "Magnitud de Coeficiente Absoluto"
    else:
        return

    df_importancia: pd.DataFrame = pd.DataFrame({
        "Caracteristica": feature_names,
        "Importancia": valores
    }).sort_values(by="Importancia", ascending=False).head(top_n).reset_index(drop=True)

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=df_importancia,
        x="Importancia",
        y="Caracteristica",
        hue="Caracteristica",
        legend=False,
        palette="viridis",
        edgecolor="black"
    )

    plt.title(f"Factores de Riesgo Más Determinantes ({model_name})")
    plt.xlabel(nombre_eje)
    plt.ylabel("Variable Clínica")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    ruta_salida: str = os.path.join(output_dir, "07_top_risk_factors.png")
    plt.savefig(ruta_salida, dpi=300)
    plt.close()
