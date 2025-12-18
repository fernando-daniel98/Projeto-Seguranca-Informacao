from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Treina SVM (LBP) e avalia. Suporta validação cruzada por sujeito (GroupKFold)."
        )
    )
    parser.add_argument(
        "--subject-cv",
        action="store_true",
        help=(
            "Ativa GroupKFold por sujeito usando subject_train.npy/subject_val.npy. "
            "Nesse modo, a seleção de hiperparâmetros é feita por AUC (roc_auc)."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de folds para GroupKFold (quando --subject-cv estiver ativo).",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Carrega treino e validação separadamente
    X_train = np.load(data_dir / "X_train_lbp.npy")
    y_train = np.load(data_dir / "y_train_lbp.npy")
    X_val = np.load(data_dir / "X_val_lbp.npy")
    y_val = np.load(data_dir / "y_val_lbp.npy")

    # Metadados (para CV por sujeito)
    subject_train_path = data_dir / "subject_train.npy"
    subject_val_path = data_dir / "subject_val.npy"
    subject_train = np.load(subject_train_path) if subject_train_path.exists() else None
    subject_val = np.load(subject_val_path) if subject_val_path.exists() else None

    print("Shapes:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Val:  ", X_val.shape, y_val.shape)

    # Grade de hiperparâmetros para C e gamma
    param_grid = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }

    if args.subject_cv:
        if subject_train is None or subject_val is None:
            raise SystemExit(
                "Metadados de sujeito não encontrados. Rode extract_lbp.py (split-mode != filesystem) "
                "para gerar data/subject_train.npy e data/subject_val.npy."
            )

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        groups = np.concatenate([subject_train, subject_val])

        cv = GroupKFold(n_splits=args.k)

        base_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                    ),
                ),
            ]
        )

        print(f"=== GridSearchCV com GroupKFold por sujeito (k={args.k}) ===")
        grid = GridSearchCV(
            base_pipe,
            param_grid=param_grid,
            scoring={"auc": "roc_auc", "acc": "accuracy"},
            refit="auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
        grid.fit(X_all, y_all, groups=groups)

        print(f"\nMelhores hiperparâmetros (refit=AUC): {grid.best_params_}")
        print(f"Melhor AUC média (CV): {grid.best_score_:.4f}")

        # Treina modelo final com os melhores hiperparâmetros em TRAIN+VAL
        print("\n=== Treinando modelo final em TRAIN+VAL (best_params) ===")
        final_pipeline = grid.best_estimator_
        final_pipeline.fit(X_all, y_all)

        # Métrica rápida em VAL (apenas informativa; não é CV)
        y_val_proba = final_pipeline.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_proba)
        print(f"Acurácia em VAL (informativo): {val_acc:.4f} | AUC em VAL: {val_auc:.4f}")
    else:
        print("=== GridSearch manual: treina em TRAIN, valida em VAL ===")

        best_score = -1.0
        best_params = None
        best_pipeline = None

        for C in param_grid["svm__C"]:
            for gamma in param_grid["svm__gamma"]:
                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "svm",
                            SVC(
                                kernel="rbf",
                                C=C,
                                gamma=gamma,
                                class_weight="balanced",
                                probability=True,
                            ),
                        ),
                    ]
                )

                pipe.fit(X_train, y_train)
                val_proba = pipe.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_proba)
                print(f"C={C}, gamma={gamma}: val_auc={val_auc:.4f}")

                if val_auc > best_score:
                    best_score = val_auc
                    best_params = {"svm__C": C, "svm__gamma": gamma}
                    best_pipeline = pipe

        print(f"\nMelhores hiperparâmetros: {best_params}")
        print(f"Melhor AUC na validação: {best_score:.4f}")

        y_val_pred = best_pipeline.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"\nAcurácia em VAL: {val_acc:.4f}")
        print("Relatório em validação:")
        print(classification_report(y_val, y_val_pred, target_names=["fake (0)", "real (1)"]))

        print("\n=== Treinando modelo final em TRAIN+VAL ===")
        final_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        **{k.replace("svm__", ""): v for k, v in best_params.items()},
                        class_weight="balanced",
                        probability=True,
                    ),
                ),
            ]
        )

        X_all = np.vstack([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        final_pipeline.fit(X_all, y_all)

    # === EVALUATE ON TEST SET ===
    X_test = np.load(data_dir / "X_test_lbp.npy")
    y_test = np.load(data_dir / "y_test_lbp.npy")
    
    print(f"\n{'='*50}")
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print(f"{'='*50}")
    
    y_test_pred = final_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print(f"Test Set AUC (ROC): {test_auc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["fake (0)", "real (1)"]))

    model_path = models_dir / "metodo1_lbp_svm.pkl"
    joblib.dump(final_pipeline, model_path)
    print(f"\nModelo final (treinado em train+val) salvo em {model_path}")


if __name__ == "__main__":
    main()
