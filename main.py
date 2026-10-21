"""
Script principal del proyecto Heart Failure Prediction.

Orquesta el pipeline completo de ingeniería de datos, benchmarking de modelos
supervisados, análisis de sobreajuste, validación cruzada y optimización de umbrales
diagnósticos en cardiología.
"""

import argparse
import sys
import numpy as np
import pandas as pd

# Configurar stdout defensivo para consolas Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src import config
from src import data_processing
from src import models
from src import visualization


def parse_arguments() -> argparse.Namespace:
    """
    Parsea los argumentos de línea de comandos (CLI).

    Returns:
        argparse.Namespace: Objeto con los parámetros parseados.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Heart Failure Prediction: Benchmarking & Clinical Threshold Optimization"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.DATA_PATH,
        help="Ruta al archivo CSV con los datos de pacientes (por defecto: data/heart.csv)"
    )
    parser.add_argument(
        "--impute-zeros",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Imputar ceros no fisiológicos en RestingBP y Colesterol mediante la mediana "
             "(desactivar con --no-impute-zeros)"
    )
    parser.add_argument(
        "--export-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generar y guardar gráficos en el directorio img/ (desactivar con --no-export-plots)"
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=config.CLINICAL_MIN_RECALL,
        help="Recall clínico mínimo requerido para el análisis de umbrales (por defecto: 0.95)"
    )
    return parser.parse_args()


def run_pipeline(
    data_path: str = config.DATA_PATH,
    impute_zeros: bool = True,
    export_plots: bool = True,
    min_recall: float = config.CLINICAL_MIN_RECALL
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Ejecuta el flujo integral de análisis y modelado predictivo.

    Args:
        data_path (str): Ruta al archivo de datos CSV.
        impute_zeros (bool): Activar imputación de ceros en presión y colesterol.
        export_plots (bool): Si es True, renderiza y guarda las figuras en img/.
        min_recall (float): Sensibilidad mínima para el umbral clínico óptimo.

    Returns:
        tuple: (df_metrics, df_cv, sweet_spot)
    """
    print("=" * 80)
    print(" [+] HEART FAILURE PREDICTION: BENCHMARK CLINICO Y OPTIMIZACION DE UMBRAL")
    print("=" * 80)

    # 1. Carga de datos
    print(f"\n[1/7] Carga y verificación del dataset desde: {data_path}")
    df_raw: pd.DataFrame = data_processing.load_heart_data(data_path)
    print(f"      Dimensiones iniciales: {df_raw.shape[0]} registros x {df_raw.shape[1]} columnas")

    # 2. Tipologia de datos y clasificacion de atributos clinicos
    print("\n[2/7] Tipología y estructura de datos (Atributos Clínicos):")
    tipologia: dict[str, list[str]] = data_processing.classify_data_typology(df_raw)
    for categoria, columnas in tipologia.items():
        print(f"      • {categoria.replace('_', ' ').capitalize()}: {', '.join(columnas)}")

    # 3. Auditoria de anomalias clinicas
    print("\n[3/7] Auditoría de calidad de datos y valores fisiológicos:")
    df_anomalias: pd.DataFrame = data_processing.audit_clinical_anomalies(df_raw)
    for _, fila in df_anomalias.iterrows():
        if fila["Ceros"] > 0 and fila["Anomalia_Fisiologica"] == "Si":
            print(f"      ⚠️  {fila['Variable']}: {fila['Ceros']} registros en 0 ({fila['Porcentaje (%)']}%) - Imposible fisiológicamente.")

    df_clean: pd.DataFrame = data_processing.clean_clinical_data(df_raw, impute_zeros=impute_zeros)
    if impute_zeros:
        print("      ✅ Ceros en RestingBP y Colesterol sustituidos por NaN e imputados con la mediana de la cohorte.")

    # 4. Comprobacion de balanceo de clases
    print("\n[4/7] Comprobación de balanceo de la variable objetivo (HeartDisease):")
    conteo_target: pd.Series = df_clean["HeartDisease"].value_counts()
    pct_target: pd.Series = (df_clean["HeartDisease"].value_counts(normalize=True) * 100).round(2)
    print(f"      • Clase 0 (Normal):      {conteo_target[0]} pacientes ({pct_target[0]}%)")
    print(f"      • Clase 1 (Cardiopatía): {conteo_target[1]} pacientes ({pct_target[1]}%)")
    print(f"      ℹ️  Conclusión: La distribución ({pct_target[1]}% vs {pct_target[0]}%) está prácticamente equilibrada.")
    print("          No requiere sobremuestreo sintético (SMOTE); la estratificación (stratify=y) es óptima.")

    # 5. Preprocesamiento, One-Hot Encoding y Escalado
    print("\n[5/7] Preparación matricial, One-Hot Encoding y división estratificada:")
    X, y = data_processing.prepare_features(df_clean, target_col="HeartDisease")
    X_train_s, X_test_s, y_train, y_test, _, feature_names = data_processing.split_and_scale_data(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )
    print(f"      • Conjunto de entrenamiento: {X_train_s.shape[0]} muestras x {X_train_s.shape[1]} variables codificadas")
    print(f"      • Conjunto de prueba:        {X_test_s.shape[0]} muestras")
    print("      • Escalador: StandardScaler ajustado exclusivamente sobre Train")

    # 6. Benchmarking comparativo de los 5 modelos
    print("\n[6/7] Entrenamiento y evaluación comparativa de los 5 clasificadores:")
    modelos_lista: list[tuple[str, object]] = models.get_classification_models()
    df_metrics, trained_models = models.train_and_evaluate_models(
        modelos_lista, X_train_s, X_test_s, y_train, y_test
    )

    print("\n" + "-" * 95)
    print(f"{'Modelo':22s} | {'Test Acc':9s} | {'Recall':8s} | {'Precision':10s} | {'F1-Score':9s} | {'ROC-AUC':8s} | {'Gap (Overfit)':12s}")
    print("-" * 95)
    for _, fila in df_metrics.iterrows():
        print(
            f"{fila['Modelo']:22s} | "
            f"{fila['Test_Accuracy']:8.2%} | "
            f"{fila['Recall']:7.2%} | "
            f"{fila['Precision']:9.2%} | "
            f"{fila['F1_Score']:8.2%} | "
            f"{fila['ROC_AUC']:7.2%} | "
            f"{fila['Overfitting_Gap']:+11.2%}"
        )
    print("-" * 95)

    # Identificar mejor modelo segun F1_Score (seleccion robusta con idxmax)
    idx_mejor: int = int(df_metrics["F1_Score"].idxmax())
    mejor_modelo_nombre: str = str(df_metrics.loc[idx_mejor, "Modelo"])
    mejor_modelo_obj: object = trained_models[mejor_modelo_nombre]
    print(f"\n      🏆 Mejor clasificador global (mayor F1): {mejor_modelo_nombre} (F1: {df_metrics.loc[idx_mejor, 'F1_Score']:.2%})")

    # Validacion cruzada estratificada (5 pliegues)
    print("\n      🔄 Ejecutando 5-Fold Stratified Cross Validation para estabilidad estadística...")
    df_cv: pd.DataFrame = models.evaluate_cross_validation(X, y, n_splits=config.CV_FOLDS, random_state=config.SEED)
    print(f"      • {df_cv.loc[0, 'Modelo']} lidera CV con F1 medio de: {df_cv.loc[0, 'CV_F1_Mean']:.2%} (±{df_cv.loc[0, 'CV_F1_Std']:.2%})")

    # Comparativa de impacto de class_weight
    print("\n      ⚖️  Impacto de class_weight='balanced' sobre el F1 en test:")
    df_balance_impact: pd.DataFrame = models.analyze_class_balance_impact(
        X_train_s, X_test_s, y_train, y_test
    )
    for _, fila in df_balance_impact.iterrows():
        print(
            f"      • {fila['Modelo']:20s} F1 normal: {fila['F1_Normal']:.2%} | "
            f"F1 balanced: {fila['F1_Balanced']:.2%}"
        )

    # 7. Analisis de umbrales clinicos (Threshold Tuning)
    print("\n[7/7] Análisis de Umbral Clínico (Minimización de Falsos Negativos):")
    # Extraer probabilidades de predicción del mejor modelo (todos exponen predict_proba)
    y_test_proba: np.ndarray = mejor_modelo_obj.predict_proba(X_test_s)[:, 1]

    df_thresholds: pd.DataFrame = models.evaluate_threshold_range(
        y_test, y_test_proba, steps=config.THRESHOLDS_STEPS
    )

    # Métricas al umbral por defecto (0.50)
    _, _, _, fn_def, _, prec_def, rec_def, _ = models.calculate_metrics_at_threshold(
        y_test, y_test_proba, threshold=config.DEFAULT_THRESHOLD
    )

    # Punto dulce clinico
    sweet_spot: dict[str, object] = models.find_clinical_sweet_spot(
        df_thresholds, min_recall=min_recall
    )
    t_opt: float = float(sweet_spot["Threshold"])
    rec_opt: float = float(sweet_spot["Recall"])
    prec_opt: float = float(sweet_spot["Precision"])
    fn_opt: int = int(sweet_spot["FN"])

    print(f"      • Umbral estándar (0.50): Recall: {rec_def:.2%} | Precision: {prec_def:.2%} | Falsos Negativos (Omitidos): {fn_def}")
    print(f"      • Umbral clínico ({t_opt:.2f}):  Recall: {rec_opt:.2%} | Precision: {prec_opt:.2%} | Falsos Negativos (Omitidos): {fn_opt}")
    print(f"      💡 Al ajustar el umbral a {t_opt:.2f}, los Falsos Negativos pasan de {fn_def} a {fn_opt},")
    print("         a cambio de más falsas alarmas. Nota: el umbral se elige sobre el test set (estimación optimista).")

    # Generación y exportación de gráficos
    if export_plots:
        print(f"\n📊 Exportando visualizaciones a: {config.IMG_DIR}/")
        visualization.plot_target_distribution(df_clean, output_dir=config.IMG_DIR)
        visualization.plot_model_comparison(df_metrics, output_dir=config.IMG_DIR)
        visualization.plot_overfitting_analysis(df_metrics, output_dir=config.IMG_DIR)

        # Matriz de confusion con prediccion estandar
        y_test_pred_best: np.ndarray = mejor_modelo_obj.predict(X_test_s)
        visualization.plot_confusion_matrix(
            y_test, y_test_pred_best, model_name=mejor_modelo_nombre, threshold=config.DEFAULT_THRESHOLD,
            output_dir=config.IMG_DIR
        )

        # Curva de umbrales clinicos
        visualization.plot_threshold_curves(
            df_thresholds, model_name=mejor_modelo_nombre, optimal_threshold=t_opt, output_dir=config.IMG_DIR
        )

        # Curvas ROC
        visualization.plot_roc_curves(
            trained_models, X_test_s, y_test, output_dir=config.IMG_DIR
        )

        # Factores de riesgo determinantes (Random Forest o Regresion Logistica)
        if "Random Forest" in trained_models:
            visualization.plot_feature_importance(
                trained_models["Random Forest"], feature_names, "Random Forest", output_dir=config.IMG_DIR
            )
        elif "Logistic Regression" in trained_models:
            visualization.plot_feature_importance(
                trained_models["Logistic Regression"], feature_names, "Logistic Regression", output_dir=config.IMG_DIR
            )

        print("      ✅ Figuras guardadas a 300 DPI.")

    print("\n" + "=" * 80)
    print(" Pipeline ejecutado con éxito.")
    print("=" * 80)

    return df_metrics, df_cv, sweet_spot


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    run_pipeline(
        data_path=args.data,
        impute_zeros=args.impute_zeros,
        export_plots=args.export_plots,
        min_recall=args.min_recall
    )
