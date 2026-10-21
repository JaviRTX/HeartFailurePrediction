"""
Módulo de modelado, benchmarking comparativo y análisis de umbrales clínicos.

Entrena 5 clasificadores estándar (Regresión Logística, Árbol de Decisión,
Random Forest, KNN y Máquinas de Vectores de Soporte), evalúa métricas
de diagnóstico clínico y explora la variación de umbrales para minimizar
Falsos Negativos en triaje cardiológico.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from src import config


def get_classification_models(
    balanced_weights: bool = False
) -> list[tuple[str, object]]:
    """
    Instancia los 5 clasificadores estándar con hiperparámetros controlados.

    Las semillas de cada estimador se toman de config.MODEL_SEEDS.

    Args:
        balanced_weights (bool): Si es True, activa ponderación inversa de clases.

    Returns:
        list[tuple[str, object]]: Lista de pares (nombre_modelo, estimador).
    """
    weights: str | None = "balanced" if balanced_weights else None

    # Colección de clasificadores supervisados evaluados
    modelos: list[tuple[str, object]] = [
        ("Logistic Regression", LogisticRegression(
            max_iter=1000,
            class_weight=weights,
            random_state=config.MODEL_SEEDS["logistic_regression"]
        )),
        ("Decision Tree", DecisionTreeClassifier(
            class_weight=weights,
            random_state=config.MODEL_SEEDS["decision_tree"]
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=100,
            class_weight=weights,
            random_state=config.MODEL_SEEDS["random_forest"]
        )),
        ("KNN", KNeighborsClassifier(n_neighbors=5)),
        ("SVM (RBF)", SVC(
            probability=True,
            class_weight=weights,
            random_state=config.MODEL_SEEDS["svm"]
        ))
    ]
    return modelos


def train_and_evaluate_models(
    modelos: list[tuple[str, object]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Entrena cada modelo y extrae metricas en conjuntos de entrenamiento y prueba.

    Args:
        modelos (list[tuple[str, object]]): Estimadores a entrenar.
        X_train (np.ndarray): Matriz predictora de entrenamiento escalada.
        X_test (np.ndarray): Matriz predictora de prueba escalada.
        y_train (np.ndarray): Vector objetivo de entrenamiento.
        y_test (np.ndarray): Vector objetivo de prueba.

    Returns:
        tuple[pd.DataFrame, dict[str, object]]: DataFrame con metricas y dict de modelos entrenados.
    """
    results: list[dict[str, object]] = []
    trained_dict: dict[str, object] = {}

    for nombre, modelo in modelos:
        # Ajuste del estimador
        modelo.fit(X_train, y_train)
        trained_dict[nombre] = modelo

        # Predicciones de clase y probabilidades
        y_train_pred: np.ndarray = modelo.predict(X_train)
        y_test_pred: np.ndarray = modelo.predict(X_test)

        if hasattr(modelo, "predict_proba"):
            y_test_proba: np.ndarray = modelo.predict_proba(X_test)[:, 1]
            auc: float = float(roc_auc_score(y_test, y_test_proba))
        else:
            # Sin probabilidades no se puede calcular el AUC
            auc = float("nan")

        # Calculo de metricas clasicas
        train_acc: float = float(accuracy_score(y_train, y_train_pred))
        test_acc: float = float(accuracy_score(y_test, y_test_pred))
        test_rec: float = float(recall_score(y_test, y_test_pred))
        test_prec: float = float(precision_score(y_test, y_test_pred))
        test_f1: float = float(f1_score(y_test, y_test_pred))
        gap_overfitting: float = float(train_acc - test_acc)

        results.append({
            "Modelo": nombre,
            "Train_Accuracy": train_acc,
            "Test_Accuracy": test_acc,
            "Recall": test_rec,
            "Precision": test_prec,
            "F1_Score": test_f1,
            "ROC_AUC": auc,
            "Overfitting_Gap": gap_overfitting
        })

    # DataFrame comparativo ordenado por F1_Score desc
    df_metrics: pd.DataFrame = pd.DataFrame(results).sort_values(
        by="F1_Score", ascending=False
    ).reset_index(drop=True)

    return df_metrics, trained_dict


def evaluate_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = config.CV_FOLDS,
    random_state: int = config.SEED
) -> pd.DataFrame:
    """
    Valida la robustez de los modelos mediante 5-Fold Stratified Cross Validation.

    Aplica Pipeline con StandardScaler en cada pliegue para evitar fuga de datos.

    Args:
        X (pd.DataFrame): Predictores sin escalar.
        y (pd.Series): Etiquetas reales.
        n_splits (int): Numero de particiones estratificadas.
        random_state (int): Semilla para barajar los pliegues (los modelos usan config.MODEL_SEEDS).

    Returns:
        pd.DataFrame: Resumen estadistico de metricas medias y desviacion estandar.
    """
    cv: StratifiedKFold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    modelos: list[tuple[str, object]] = get_classification_models()
    scoring: list[str] = ["accuracy", "recall", "precision", "f1", "roc_auc"]

    cv_results: list[dict[str, object]] = []

    for nombre, clf in modelos:
        pipeline: Pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        scores: dict[str, np.ndarray] = cross_validate(
            pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True
        )

        cv_results.append({
            "Modelo": nombre,
            "CV_Accuracy_Mean": float(scores["test_accuracy"].mean()),
            "CV_Accuracy_Std": float(scores["test_accuracy"].std()),
            "CV_Recall_Mean": float(scores["test_recall"].mean()),
            "CV_Recall_Std": float(scores["test_recall"].std()),
            "CV_Precision_Mean": float(scores["test_precision"].mean()),
            "CV_Precision_Std": float(scores["test_precision"].std()),
            "CV_F1_Mean": float(scores["test_f1"].mean()),
            "CV_F1_Std": float(scores["test_f1"].std()),
            "CV_ROC_AUC_Mean": float(scores["test_roc_auc"].mean()),
            "CV_ROC_AUC_Std": float(scores["test_roc_auc"].std()),
            "CV_Train_Acc_Mean": float(scores["train_accuracy"].mean())
        })

    # Ordenar por F1 medio desc
    df_cv: pd.DataFrame = pd.DataFrame(cv_results).sort_values(
        by="CV_F1_Mean", ascending=False
    ).reset_index(drop=True)
    return df_cv


def analyze_class_balance_impact(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> pd.DataFrame:
    """
    Evalua si el balanceo de pesos (class_weight='balanced') aporta beneficio real.

    En este dataset la proporción es 55.3% enfermos / 44.7% sanos, una distribución
    prácticamente equilibrada donde la estratificación suele ser suficiente.

    Args:
        X_train (np.ndarray): Matriz de entrenamiento escalada.
        X_test (np.ndarray): Matriz de prueba escalada.
        y_train (np.ndarray): Vector objetivo train.
        y_test (np.ndarray): Vector objetivo test.

    Returns:
        pd.DataFrame: Comparativa directa entre modelos sin y con ponderacion de clases.
    """
    modelos_std: list[tuple[str, object]] = get_classification_models(balanced_weights=False)
    modelos_bal: list[tuple[str, object]] = get_classification_models(balanced_weights=True)

    comparativa: list[dict[str, object]] = []

    for (nom_std, m_std), (_, m_bal) in zip(modelos_std, modelos_bal):
        # Ajuste estandar
        m_std.fit(X_train, y_train)
        pred_std: np.ndarray = m_std.predict(X_test)

        # Ajuste con balanceo de pesos
        m_bal.fit(X_train, y_train)
        pred_bal: np.ndarray = m_bal.predict(X_test)

        comparativa.append({
            "Modelo": nom_std,
            "Test_Acc_Normal": float(accuracy_score(y_test, pred_std)),
            "Test_Acc_Balanced": float(accuracy_score(y_test, pred_bal)),
            "Recall_Normal": float(recall_score(y_test, pred_std)),
            "Recall_Balanced": float(recall_score(y_test, pred_bal)),
            "F1_Normal": float(f1_score(y_test, pred_std)),
            "F1_Balanced": float(f1_score(y_test, pred_bal))
        })

    return pd.DataFrame(comparativa)


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float
) -> tuple[int, int, int, int, float, float, float, float]:
    """
    Calcula matriz de confusion y metricas diagnosticas para un umbral especifico.

    Permite evaluar el compromiso clínico entre Sensibilidad (Recall) y Precisión.

    Args:
        y_true (np.ndarray): Etiquetas reales del conjunto de prueba.
        y_pred_proba (np.ndarray): Probabilidades asignadas a la clase positiva.
        threshold (float): Punto de corte para clasificar como positivo (>= threshold).

    Returns:
        tuple: (tp, tn, fp, fn, accuracy, precision, recall, f1)
    """
    # Prediccion segun umbral
    y_pred: np.ndarray = (y_pred_proba >= threshold).astype(int)

    # Matriz confusion
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn: int = int(tn_val)
    fp: int = int(fp_val)
    fn: int = int(fn_val)
    tp: int = int(tp_val)

    # Calculo metricas
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
    steps: int = config.THRESHOLDS_STEPS
) -> pd.DataFrame:
    """
    Itera sobre un rango continuo de umbrales [0.0, 1.0] calculando el vector de metricas.

    Args:
        y_true (np.ndarray): Vector de etiquetas verdaderas.
        y_pred_proba (np.ndarray): Vector de probabilidades predichas de clase positiva.
        steps (int): Numero de intervalos equidistantes evaluados.

    Returns:
        pd.DataFrame: Tabla con TP, TN, FP, FN, Precision, Recall, F1 y Accuracy por umbral.
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

    df_thresholds: pd.DataFrame = pd.DataFrame(filas)
    return df_thresholds


def find_clinical_sweet_spot(
    threshold_df: pd.DataFrame,
    min_recall: float = config.CLINICAL_MIN_RECALL
) -> dict[str, object]:
    """
    Determina el umbral optimo de cribado clinico para minimizar Falsos Negativos.

    En cardiologia, un Falso Negativo (enfermo no detectado) es potencialmente letal.
    Esta funcion localiza el umbral que maximiza el F1-Score garantizando una
    sensibilidad minima (Recall >= min_recall).

    Args:
        threshold_df (pd.DataFrame): DataFrame generado por evaluate_threshold_range.
        min_recall (float): Sensibilidad minima aceptable en entorno hospitalario.

    Returns:
        dict[str, object]: Fila y metricas del umbral clinico optimo seleccionado.
    """
    # Filtrar candidatos que respeten el recall clinico minimo
    candidatos: pd.DataFrame = threshold_df[threshold_df["Recall"] >= min_recall].copy()

    if candidatos.empty:
        # Fallback al registro con mayor recall si ninguno alcanza el umbral estricto
        idx_optimo = threshold_df["Recall"].idxmax()
    else:
        # Seleccion de maximo F1 entre los que cumplen el requisito clinico
        idx_optimo = candidatos["F1_Score"].idxmax()

    fila_optima: pd.Series = threshold_df.loc[idx_optimo]
    return fila_optima.to_dict()
