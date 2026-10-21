"""
Batería de pruebas unitarias para el pipeline de predicción de Heart Failure.

Verifica la integridad de la carga, la consistencia de la imputación clínica,
el One-Hot Encoding, el escalado, el entrenamiento de modelos y el cálculo
preciso de matrices de confusión y métricas por umbral.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src import config
from src import data_processing
from src import models


class TestHeartFailurePipeline(unittest.TestCase):
    """Comprueba las funciones críticas del flujo de machine learning cardiológico."""

    def setUp(self) -> None:
        """Carga y prepara el dataset de prueba para cada test."""
        self.data_path: str = config.DATA_PATH
        self.assertTrue(os.path.isfile(self.data_path), f"No existe el archivo {self.data_path}")
        self.df_raw: pd.DataFrame = data_processing.load_heart_data(self.data_path)

    def test_data_loading_dimensions(self) -> None:
        """Verifica que el dataset se cargue con las dimensiones esperadas (918x12)."""
        self.assertEqual(self.df_raw.shape[0], 918)
        self.assertEqual(self.df_raw.shape[1], 12)
        self.assertIn("HeartDisease", self.df_raw.columns)
        self.assertIn("Cholesterol", self.df_raw.columns)

    def test_typology_classification(self) -> None:
        """Verifica que la clasificación de variables cubra todas las columnas sin duplicados."""
        tipologia = data_processing.classify_data_typology(self.df_raw)
        todas_las_variables: list[str] = []
        for cols in tipologia.values():
            todas_las_variables.extend(cols)

        # 12 variables únicas en total
        self.assertEqual(len(todas_las_variables), 12)
        self.assertEqual(len(set(todas_las_variables)), 12)

    def test_clean_clinical_anomalies(self) -> None:
        """Comprueba que la imputación elimine los ceros fisiológicamente imposibles."""
        # Comprobar existencia previa de ceros
        ceros_bp_antes: int = int((self.df_raw["RestingBP"] == 0).sum())
        ceros_chol_antes: int = int((self.df_raw["Cholesterol"] == 0).sum())
        self.assertGreater(ceros_bp_antes, 0)
        self.assertGreater(ceros_chol_antes, 0)

        # Limpiar
        df_clean = data_processing.clean_clinical_data(self.df_raw, impute_zeros=True)
        ceros_bp_despues: int = int((df_clean["RestingBP"] == 0).sum())
        ceros_chol_despues: int = int((df_clean["Cholesterol"] == 0).sum())

        self.assertEqual(ceros_bp_despues, 0)
        self.assertEqual(ceros_chol_despues, 0)
        self.assertEqual(len(df_clean), len(self.df_raw))

    def test_prepare_features_and_scaling(self) -> None:
        """Verifica One-Hot Encoding y que el escalador estandarice con media 0 y varianza 1."""
        df_clean = data_processing.clean_clinical_data(self.df_raw, impute_zeros=True)
        X, y = data_processing.prepare_features(df_clean, target_col="HeartDisease")

        self.assertEqual(len(X), len(y))
        # Codificación debe expandir categorías cualitativas
        self.assertGreater(X.shape[1], df_clean.shape[1] - 1)

        X_train_s, X_test_s, y_train, y_test, scaler, _ = data_processing.split_and_scale_data(
            X, y, test_size=0.2, random_state=42
        )

        # Comprobar estratificación aproximada en particiones
        ratio_train = float(np.mean(y_train))
        ratio_test = float(np.mean(y_test))
        self.assertAlmostEqual(ratio_train, ratio_test, places=2)

        # Comprobar propiedades de StandardScaler en Train (media aprox 0)
        medias_train = np.mean(X_train_s, axis=0)
        self.assertTrue(np.all(np.abs(medias_train) < 1e-6))

    def test_model_training_and_metrics(self) -> None:
        """Verifica que los modelos se entrenen y reporten métricas válidas (entre 0 y 1)."""
        df_clean = data_processing.clean_clinical_data(self.df_raw, impute_zeros=True)
        X, y = data_processing.prepare_features(df_clean, target_col="HeartDisease")
        X_train_s, X_test_s, y_train, y_test, _, _ = data_processing.split_and_scale_data(
            X, y, test_size=0.2, random_state=42
        )

        modelos_lista = models.get_classification_models()
        df_metrics, trained_dict = models.train_and_evaluate_models(
            modelos_lista, X_train_s, X_test_s, y_train, y_test
        )

        self.assertEqual(len(df_metrics), len(modelos_lista))
        self.assertIn("Logistic Regression", trained_dict)
        self.assertIn("SVM (RBF)", trained_dict)

        # Todas las métricas en rango [0, 1]
        for col in ["Test_Accuracy", "Recall", "Precision", "F1_Score", "ROC_AUC"]:
            self.assertTrue(all(df_metrics[col] >= 0.0))
            self.assertTrue(all(df_metrics[col] <= 1.0))

    def test_threshold_metrics_calculation(self) -> None:
        """Verifica la coherencia aritmética de calculate_metrics_at_threshold."""
        y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.85, 0.05])

        # A umbral 0.5:
        # Predicciones: [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
        # Reales:       [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
        # Positivos predichos: idx 0(1), 1(1), 2(1), 3(0), 8(1) -> TP=4, FP=1
        # Negativos predichos: idx 4(0), 5(0), 6(1), 7(0), 9(0) -> TN=4, FN=1
        tp, tn, fp, fn, acc, prec, rec, f1 = models.calculate_metrics_at_threshold(y_true, y_proba, 0.50)

        self.assertEqual(tp, 4)
        self.assertEqual(tn, 4)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 1)
        self.assertAlmostEqual(acc, 0.80)
        self.assertAlmostEqual(prec, 4 / 5)
        self.assertAlmostEqual(rec, 4 / 5)
        self.assertAlmostEqual(f1, 0.80)

    def test_clinical_sweet_spot(self) -> None:
        """Verifica que el punto dulce clínico garantice el Recall mínimo exigido."""
        df_clean = data_processing.clean_clinical_data(self.df_raw, impute_zeros=True)
        X, y = data_processing.prepare_features(df_clean, target_col="HeartDisease")
        X_train_s, X_test_s, y_train, y_test, _, _ = data_processing.split_and_scale_data(
            X, y, test_size=0.2, random_state=42
        )

        clf = models.get_classification_models()[0][1]
        clf.fit(X_train_s, y_train)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        df_th = models.evaluate_threshold_range(y_test, y_proba, steps=51)
        sweet_spot = models.find_clinical_sweet_spot(df_th, min_recall=0.95)

        self.assertGreaterEqual(sweet_spot["Recall"], 0.95)


if __name__ == "__main__":
    unittest.main()
