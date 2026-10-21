"""
Módulo de configuración y variables globales del proyecto.

Define rutas, hiperparámetros por defecto, semillas de reproducibilidad
y configuración del pipeline de clasificación para Heart Failure.
"""

import os
from datetime import datetime

# Rutas del proyecto
DATA_PATH: str = os.path.join("data", "heart.csv")
IMG_DIR: str = "img"

# Marca temporal para versionado de artefactos gráficos
TIMESTAMP: str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Semillas de reproducibilidad
# 42: La respuesta del sentido de la vida, el universo y todo lo demás
SEED: int = 42
MODEL_SEEDS: dict[str, int] = {
    "logistic_regression": 42,
    "decision_tree": 3008,
    "random_forest": 3008,
    "svm": 42
}

# Parámetros del particionado
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# Configuración del análisis de umbrales clínicos
THRESHOLDS_STEPS: int = 101
DEFAULT_THRESHOLD: float = 0.50
CLINICAL_MIN_RECALL: float = 0.95  # Prioridad médica: minimizar falsos negativos

# Nombres descriptivos de modelos analizados
MODEL_NAMES: list[str] = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "KNN",
    "SVM (RBF)"
]
