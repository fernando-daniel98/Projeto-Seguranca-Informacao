from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int  # fake=0, real=1
    subject: int
    session: int


def _parse_nuaa_filename(file_path: Path) -> Tuple[int, int]:
    """Extrai (subject, session) do padrão de nome NUAA.

    Observação: no dataset NUAA via Kaggle (aleksandrpikul222/nuaaaa), os nomes
    seguem um padrão com campos separados por "_". Na prática, o primeiro campo
    é o sujeito (0001..0015) e o 4º campo é uma sessão (01/02/03).

    Ex.: 0011_01_07_03_202.jpg -> subject=11, session=3
    Quando organizado por database.py, o formato é: 0011_0011_01_07_03_202.jpg
    onde o sujeito aparece duplicado no início.
    """

    stem = file_path.stem
    parts = stem.split("_")
    
    # Suporta ambos formatos: original (0011_01_07_03_202.jpg) 
    # e organizado por database.py (0011_0011_01_07_03_202.jpg)
    if len(parts) >= 5 and parts[0] == parts[1]:
        # Formato database.py: subject_subject_...
        subject = int(parts[0])
        session = int(parts[4])
    elif len(parts) >= 4:
        # Formato original: subject_...
        subject = int(parts[0])
        session = int(parts[3])
    else:
        raise ValueError(f"Nome de arquivo NUAA inesperado: {file_path.name}")

    return subject, session


def _index_all_images(dataset_dir: Path) -> List[ImageRecord]:
    """Indexa todas as imagens sob dataset_dir, inferindo label pelo diretório pai.

    Aceita a estrutura já existente (dataset/train|val|test/{real,fake}/...), mas
    ignora o split atual e trata tudo como um pool único.
    """

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")

    records: List[ImageRecord] = []
    for file_path in dataset_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in IMG_EXTS:
            continue

        parent = file_path.parent.name.lower()
        if parent not in {"real", "fake"}:
            continue

        label = 1 if parent == "real" else 0
        subject, session = _parse_nuaa_filename(file_path)
        records.append(
            ImageRecord(path=file_path, label=label, subject=subject, session=session)
        )

    if not records:
        raise RuntimeError(
            f"Nenhuma imagem encontrada em {dataset_dir} (extensões: {IMG_EXTS})"
        )

    return records


def _split_by_group(
    records: Sequence[ImageRecord],
    group_by: str,
    seed: int,
    split: Tuple[float, float, float],
) -> Dict[str, List[ImageRecord]]:
    """Particiona por grupo (subject ou session), sem vazamento entre splits."""

    if group_by not in {"subject", "session"}:
        raise ValueError("group_by deve ser 'subject' ou 'session'")

    rng = np.random.default_rng(seed)
    groups = sorted({getattr(r, group_by) for r in records})
    rng.shuffle(groups)

    n = len(groups)
    if n < 3:
        raise ValueError(
            f"Poucos grupos ({n}) para criar train/val/test por {group_by}."
        )

    n_train = int(round(n * split[0]))
    n_val = int(round(n * split[1]))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train : n_train + n_val])
    test_groups = set(groups[n_train + n_val :])

    out: Dict[str, List[ImageRecord]] = {"train": [], "val": [], "test": []}
    for r in records:
        g = getattr(r, group_by)
        if g in train_groups:
            out["train"].append(r)
        elif g in val_groups:
            out["val"].append(r)
        elif g in test_groups:
            out["test"].append(r)
        else:
            raise AssertionError("Grupo não alocado")

    return out


def _split_official_like(
    records: Sequence[ImageRecord],
    seed: int,
    val_subject_fraction: float,
) -> Dict[str, List[ImageRecord]]:
    """Cria uma divisão *compatível em tamanho* com a literatura NUAA.

    Objetivo (valores frequentemente reportados):
    - treino: 3.491 real / 5.761 fake
    - teste:  1.614 real / 1.748 fake

    Implementação (determinística e reprodutível):
    - fake: session 01/02 -> test (1.748); session 03 -> train (5.761)
    - real: escolhe 1.614 amostras de session 01/02 para o test; o restante real -> train
    - val: separa uma fração de sujeitos do treino para validação (sem vazamento por sujeito entre train/val)

    Observação: esta divisão NÃO garante sujeitos inéditos em test (não é um "subject-disjoint holdout").
    Para isso, use split-mode=group-subject e/ou o GroupKFold no treino.
    """

    rng = np.random.default_rng(seed)

    reals = [r for r in records if r.label == 1]
    fakes = [r for r in records if r.label == 0]

    fake_test = [r for r in fakes if r.session in {1, 2}]
    fake_train = [r for r in fakes if r.session == 3]
    if len(fake_test) != 1748 or len(fake_train) != 5761:
        raise ValueError(
            "Divisão official-like não bate com o esperado para FAKE. "
            f"Obtido: test={len(fake_test)}, train={len(fake_train)}"
        )

    real_candidates_test = [r for r in reals if r.session in {1, 2}]
    real_train = [r for r in reals if r.session == 3]

    if len(real_candidates_test) + len(real_train) != 5105:
        # sanidade: total de reals do NUAA (para este dump) costuma ser 5105
        pass

    # Seleção determinística (embaralha mas fixa seed)
    rng.shuffle(real_candidates_test)
    real_test = real_candidates_test[:1614]
    real_train += real_candidates_test[1614:]

    if len(real_test) != 1614 or len(real_train) != 3491:
        raise ValueError(
            "Divisão official-like não bate com o esperado para REAL. "
            f"Obtido: test={len(real_test)}, train={len(real_train)}"
        )

    train = fake_train + real_train
    test = fake_test + real_test

    # --- Val por sujeito, retirada do treino ---
    train_subjects = sorted({r.subject for r in train})
    if not train_subjects:
        raise ValueError("Treino vazio")

    n_val_subjects = max(1, int(round(len(train_subjects) * val_subject_fraction)))
    rng.shuffle(train_subjects)
    val_subjects = set(train_subjects[:n_val_subjects])

    train2: List[ImageRecord] = []
    val: List[ImageRecord] = []
    for r in train:
        if r.subject in val_subjects:
            val.append(r)
        else:
            train2.append(r)

    return {"train": train2, "val": val, "test": test}

def extract_lbp_features(
    image_path,
    radius=2,
    n_points=None,
    method="uniform",
    resize_to=(224, 224),
):
    """
    Lê uma imagem, converte para cinza, redimensiona e extrai
    histograma LBP normalizado.
    """
    if n_points is None:
        n_points = 8 * radius

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if resize_to is not None:
        gray = cv2.resize(gray, resize_to)

    lbp = local_binary_pattern(gray, n_points, radius, method)

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,  # histograma normalizado
    )

    return hist.astype("float32")


def _records_to_arrays(records: Sequence[ImageRecord]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    subjects: List[int] = []
    sessions: List[int] = []

    for r in records:
        try:
            feat = extract_lbp_features(r.path)
            features.append(feat)
            labels.append(r.label)
            subjects.append(r.subject)
            sessions.append(r.session)
        except Exception as e:
            print(f"[ERRO] {r.path}: {e}")

    if not features:
        raise RuntimeError("Nenhuma feature extraída (todas as leituras falharam)")

    X = np.stack(features).astype("float32")
    y = np.array(labels, dtype="int64")
    subj = np.array(subjects, dtype="int64")
    sess = np.array(sessions, dtype="int64")
    return X, y, subj, sess

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extrai LBP e salva .npy. Pode refazer splits sem vazamento por sujeito/sessão."
    )
    parser.add_argument(
        "--split-mode",
        choices=["filesystem", "group-subject", "group-session", "official-like"],
        default="group-subject",
        help=(
            "Como gerar train/val/test. "
            "filesystem=usa dataset/train|val|test como está; "
            "group-subject=holdout por sujeito (sem vazamento); "
            "group-session=holdout por sessão (sem vazamento); "
            "official-like=treino/test com tamanhos da literatura + val por sujeito dentro do treino."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade (embaralhamento de grupos/seleções).",
    )
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=(0.7, 0.2, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Proporção (apenas para group-*) em termos de grupos (sujeitos/sessões).",
    )
    parser.add_argument(
        "--val-subject-fraction",
        type=float,
        default=0.15,
        help="Apenas para official-like: fração de sujeitos do treino enviada para validação.",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    print(f"Base directory: {base_dir}")
    # Usa dataset da raiz do projeto (gerado por database.py)
    dataset_path = base_dir.parent / "dataset"
    print(f"Dataset path: {dataset_path}")
    output_path = base_dir / "data"
    print(f"Output path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    if args.split_mode == "filesystem":
        # Modo antigo (mantido para compatibilidade)
        def load_split_features(dataset_dir: Path, split: str):
            split_dir = dataset_dir / split
            features = []
            labels = []
            subjects = []
            sessions = []

            label_map = {"real": 1, "fake": 0}
            for label_name, label_value in label_map.items():
                class_dir = split_dir / label_name
                if not class_dir.exists():
                    print(f"[AVISO] Diretório não encontrado: {class_dir}")
                    continue

                for root, _, files in os.walk(class_dir):
                    for fname in files:
                        if not fname.lower().endswith(IMG_EXTS):
                            continue
                        img_path = Path(root) / fname
                        try:
                            feat = extract_lbp_features(img_path)
                            subject, session = _parse_nuaa_filename(img_path)
                            features.append(feat)
                            labels.append(label_value)
                            subjects.append(subject)
                            sessions.append(session)
                        except Exception as e:
                            print(f"[ERRO] {img_path}: {e}")

            if not features:
                raise RuntimeError(f"Nenhuma imagem encontrada em {split_dir}")

            X = np.stack(features).astype("float32")
            y = np.array(labels, dtype="int64")
            subj = np.array(subjects, dtype="int64")
            sess = np.array(sessions, dtype="int64")
            return X, y, subj, sess

        print("=== Extraindo LBP (filesystem) TREINO/VALIDAÇÃO/TESTE ===")
        X_train, y_train, subject_train, session_train = load_split_features(dataset_path, "train")
        X_val, y_val, subject_val, session_val = load_split_features(dataset_path, "val")
        X_test, y_test, subject_test, session_test = load_split_features(dataset_path, "test")
    else:
        print("Indexando toda a base (ignorando splits atuais)...")
        all_records = _index_all_images(dataset_path)
        print(f"Total indexado: {len(all_records)} imagens")

        if args.split_mode == "group-subject":
            print("Gerando splits por SUJEITO (sem vazamento entre train/val/test)...")
            splits = _split_by_group(all_records, "subject", seed=args.seed, split=tuple(args.split))
        elif args.split_mode == "group-session":
            print("Gerando splits por SESSÃO (sem vazamento entre train/val/test)...")
            splits = _split_by_group(all_records, "session", seed=args.seed, split=tuple(args.split))
        elif args.split_mode == "official-like":
            print("Gerando splits 'official-like' (tamanhos da literatura + val por sujeito)...")
            splits = _split_official_like(
                all_records, seed=args.seed, val_subject_fraction=args.val_subject_fraction
            )
        else:
            raise ValueError(f"split-mode inválido: {args.split_mode}")

        # Materializa arrays
        X_train, y_train, subject_train, session_train = _records_to_arrays(splits["train"])
        X_val, y_val, subject_val, session_val = _records_to_arrays(splits["val"])
        X_test, y_test, subject_test, session_test = _records_to_arrays(splits["test"])

    def _print_split_stats(name: str, y: np.ndarray, subj: np.ndarray, sess: np.ndarray) -> None:
        counts = np.bincount(y, minlength=2)
        print(
            f"{name}: X={len(y)} | fake={counts[0]} real={counts[1]} | "
            f"subjects={len(set(subj.tolist()))} sessions={len(set(sess.tolist()))}"
        )

    print("\n=== Resumo dos splits ===")
    _print_split_stats("Train", y_train, subject_train, session_train)
    _print_split_stats("Val  ", y_val, subject_val, session_val)
    _print_split_stats("Test ", y_test, subject_test, session_test)

    np.save(output_path / "X_train_lbp.npy", X_train)
    np.save(output_path / "y_train_lbp.npy", y_train)
    np.save(output_path / "X_val_lbp.npy", X_val)
    np.save(output_path / "y_val_lbp.npy", y_val)
    np.save(output_path / "X_test_lbp.npy", X_test) 
    np.save(output_path / "y_test_lbp.npy", y_test)

    # Metadados para validação cruzada por sujeito e auditoria de vazamento
    np.save(output_path / "subject_train.npy", subject_train)
    np.save(output_path / "subject_val.npy", subject_val)
    np.save(output_path / "subject_test.npy", subject_test)
    np.save(output_path / "session_train.npy", session_train)
    np.save(output_path / "session_val.npy", session_val)
    np.save(output_path / "session_test.npy", session_test)

    print(f"Features LBP salvas em {output_path}")

if __name__ == "__main__":
    main()