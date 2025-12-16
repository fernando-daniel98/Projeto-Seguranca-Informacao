"""Exploração do dataset (Método 1).

Este script inspeciona a pasta `dataset/` (train/val/test com classes real/fake)
para exibir informações úteis de EDA (exploratory data analysis):
- contagem de imagens por split e por classe
- distribuição de extensões
- verificação de arquivos ausentes/quebrados comparando com `dataset.csv` (se existir)
- estatísticas de tamanho (HxW, canais) e intensidade (média/desvio) em uma amostra

Saídas:
- imprime um resumo no console
- salva um relatório em `results/exploratory_dataset_summary.txt`
- opcionalmente salva CSVs de problemas e estatísticas de amostra em `results/`

Uso (a partir da pasta Metodo_1):
  python exploratory_analises.py

Opções:
  python exploratory_analises.py --sample-size 2000
  python exploratory_analises.py --no-image-read   # apenas contagens/paths
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None  # type: ignore


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"} 


@dataclass(frozen=True)
class DatasetRecord:
    abs_path: Path
    rel_path: str
    split: str
    label: str


class DualWriter:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.file_path, "w", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def write(self, msg: str = "") -> None:
        print(msg)
        self._fp.write(msg + "\n")


def _collect_from_filesystem(dataset_dir: Path) -> List[DatasetRecord]:
    records: List[DatasetRecord] = []
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Pasta de dataset não encontrada: {dataset_dir}")

    for split_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()]):
        split = split_dir.name
        for label_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            label = label_dir.name
            for file_path in label_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in IMG_EXTS:
                    continue
                rel_path = file_path.relative_to(dataset_dir.parent).as_posix()
                records.append(
                    DatasetRecord(
                        abs_path=file_path,
                        rel_path=rel_path,
                        split=split,
                        label=label,
                    )
                )

    return records


def _collect_from_csv(base_dir: Path, csv_path: Path) -> List[DatasetRecord]:
    required_cols = {"filepath", "split", "label"}

    # Prefer pandas when available (mais rápido e robusto), mas não dependa disso.
    if pd is not None:
        df = pd.read_csv(csv_path)
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"CSV {csv_path} deve conter colunas {sorted(required_cols)}; tem {list(df.columns)}"
            )

        records: List[DatasetRecord] = []
        for _, row in df.iterrows():
            rel = str(row["filepath"]).replace("\\\\", "/")
            abs_path = (base_dir / rel).resolve()
            records.append(
                DatasetRecord(
                    abs_path=abs_path,
                    rel_path=rel,
                    split=str(row["split"]),
                    label=str(row["label"]),
                )
            )
        return records

    # Fallback: módulo csv
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not required_cols.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"CSV {csv_path} deve conter colunas {sorted(required_cols)}; tem {reader.fieldnames}"
            )

        records = []
        for row in reader:
            rel = str(row["filepath"]).replace("\\\\", "/")
            abs_path = (base_dir / rel).resolve()
            records.append(
                DatasetRecord(
                    abs_path=abs_path,
                    rel_path=rel,
                    split=str(row["split"]),
                    label=str(row["label"]),
                )
            )
        return records


def _counts_table_str(records: Iterable[DatasetRecord]) -> str:
    counts: Dict[str, Counter] = {}
    labels_set = set()

    for r in records:
        counts.setdefault(r.split, Counter())
        counts[r.split][r.label] += 1
        labels_set.add(r.label)

    if not counts:
        return "(vazio)"

    labels = sorted(labels_set)
    splits = sorted(counts.keys())

    # Build table
    header = ["split"] + labels + ["total"]
    rows: List[List[str]] = []
    totals = Counter()

    for split in splits:
        row_counts = counts[split]
        total = sum(row_counts.values())
        totals.update(row_counts)
        rows.append([split] + [str(row_counts.get(lbl, 0)) for lbl in labels] + [str(total)])

    grand_total = sum(totals.values())
    rows.append(["TOTAL"] + [str(totals.get(lbl, 0)) for lbl in labels] + [str(grand_total)])

    # column widths
    col_widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_row(cols: List[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))

    out_lines = [fmt_row(header), fmt_row(["-" * w for w in col_widths])]
    out_lines += [fmt_row(r) for r in rows]
    return "\n".join(out_lines)


def _ext_distribution(records: Iterable[DatasetRecord]) -> Counter:
    c = Counter()
    for r in records:
        c[r.abs_path.suffix.lower()] += 1
    return c


def _compare_sources(
    fs_records: List[DatasetRecord], csv_records: List[DatasetRecord]
) -> Tuple[List[DatasetRecord], List[DatasetRecord], List[DatasetRecord]]:
    """Retorna (presentes_no_csv_mas_ausentes_no_fs, presentes_no_fs_mas_ausentes_no_csv, ilegiveis_no_fs)."""

    fs_set = {r.rel_path for r in fs_records}
    csv_set = {r.rel_path for r in csv_records}

    csv_missing_in_fs = [r for r in csv_records if r.rel_path not in fs_set]
    fs_missing_in_csv = [r for r in fs_records if r.rel_path not in csv_set]

    unreadable: List[DatasetRecord] = []
    for r in fs_records:
        # Checagem leve (existência), leitura completa fica na etapa de amostra
        if not r.abs_path.exists():
            unreadable.append(r)

    return csv_missing_in_fs, fs_missing_in_csv, unreadable


def _sample_image_stats(
    records: List[DatasetRecord],
    sample_size: int,
    seed: int,
) -> Tuple["object", "object"]:
    """Lê uma amostra de imagens e retorna (stats_df, issues_df).

    Retorna DataFrames (pandas) quando disponível; caso contrário, retorna listas de dict.
    Se dependências (opencv/numpy) não estiverem instaladas, retorna vazio.
    """

    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        return (pd.DataFrame() if pd is not None else []), (pd.DataFrame() if pd is not None else [])

    if not records:
        return pd.DataFrame(), pd.DataFrame()

    rng = random.Random(seed)
    n = min(sample_size, len(records))
    sampled = rng.sample(records, n)

    stats_rows: List[Dict[str, object]] = []
    issues_rows: List[Dict[str, object]] = []

    for r in sampled:
        img = cv2.imread(str(r.abs_path))
        if img is None:
            issues_rows.append(
                {
                    "filepath": r.rel_path,
                    "split": r.split,
                    "label": r.label,
                    "issue": "cv2.imread returned None",
                }
            )
            continue

        h, w = img.shape[:2]
        ch = 1 if img.ndim == 2 else (img.shape[2] if img.ndim == 3 else -1)

        # intensidade em escala de cinza para padronizar
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            gray = gray.astype(np.float32)
            mean_intensity = float(gray.mean())
            std_intensity = float(gray.std())
        except Exception as e:
            issues_rows.append(
                {
                    "filepath": r.rel_path,
                    "split": r.split,
                    "label": r.label,
                    "issue": f"failed intensity stats: {e}",
                }
            )
            mean_intensity = float("nan")
            std_intensity = float("nan")

        stats_rows.append(
            {
                "filepath": r.rel_path,
                "split": r.split,
                "label": r.label,
                "height": int(h),
                "width": int(w),
                "channels": int(ch),
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "ext": r.abs_path.suffix.lower(),
            }
        )

    if pd is not None:
        return pd.DataFrame(stats_rows), pd.DataFrame(issues_rows)
    return stats_rows, issues_rows


def _resolution_summary_str(stats: object) -> str:
    if pd is not None and hasattr(stats, "empty"):
        stats_df = stats  # type: ignore
        if stats_df is None or stats_df.empty:
            return "(vazio)"

        tmp = stats_df.copy()
        tmp["resolution"] = tmp["height"].astype(str) + "x" + tmp["width"].astype(str)
        by_split_label = (
            tmp.groupby(["split", "label", "resolution"]).size().reset_index(name="count")
        )

        out_rows = []
        for (split, label), group in by_split_label.groupby(["split", "label"]):
            top = group.sort_values("count", ascending=False).head(10)
            for _, row in top.iterrows():
                out_rows.append(
                    {
                        "split": split,
                        "label": label,
                        "resolution": row["resolution"],
                        "count": int(row["count"]),
                    }
                )
        return pd.DataFrame(out_rows).to_string(index=False) if out_rows else "(vazio)"

    # Fallback sem pandas
    if not stats:
        return "(vazio)"
    counts = Counter()
    for row in stats:  # type: ignore
        key = (row.get("split"), row.get("label"), f"{row.get('height')}x{row.get('width')}")
        counts[key] += 1
    if not counts:
        return "(vazio)"

    out_lines = ["split  label  resolution  count", "-----  -----  ----------  -----"]
    grouped: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}
    for (split, label, res), cnt in counts.items():
        grouped.setdefault((split, label), []).append((res, cnt))
    for (split, label), items in sorted(grouped.items()):
        for res, cnt in sorted(items, key=lambda x: x[1], reverse=True)[:10]:
            out_lines.append(f"{split}  {label}  {res}  {cnt}")
    return "\n".join(out_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis do dataset em Metodo_1/dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Caminho para a pasta dataset (default: ./dataset)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Caminho para dataset.csv (default: ./dataset.csv se existir)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1500,
        help="Quantidade de imagens para amostrar e ler (default: 1500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para amostragem (default: 42)",
    )
    parser.add_argument(
        "--no-image-read",
        action="store_true",
        help="Não lê imagens; faz apenas contagens por pasta/csv",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Salva CSVs de stats/issues em results/",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    dataset_dir = Path(args.dataset_dir).resolve() if args.dataset_dir else (base_dir / "dataset")

    results_dir = base_dir / "results"
    summary_path = results_dir / "exploratory_dataset_summary.txt"

    writer = DualWriter(summary_path)
    try:
        writer.write("=== Exploratory Data Analysis (Metodo_1) ===")
        writer.write(f"Base dir:   {base_dir}")
        writer.write(f"Dataset dir:{dataset_dir}")
        writer.write("")

        fs_records = _collect_from_filesystem(dataset_dir)
        writer.write(f"Total de imagens (filesystem): {len(fs_records)}")
        writer.write("")

        writer.write("--- Contagem por split/label (filesystem) ---")
        writer.write(_counts_table_str(fs_records))
        writer.write("")

        ext_dist = _ext_distribution(fs_records)
        writer.write("--- Distribuição de extensões (filesystem) ---")
        writer.write(str(dict(ext_dist)))
        writer.write("")

        # CSV (opcional)
        csv_records: List[DatasetRecord] = []
        csv_path: Optional[Path] = None

        if args.csv:
            csv_path = Path(args.csv).resolve()
        else:
            candidate = base_dir / "dataset.csv"
            if candidate.exists():
                csv_path = candidate

        if csv_path is not None and csv_path.exists():
            writer.write(f"--- Verificação com CSV ---")
            writer.write(f"CSV path: {csv_path}")
            csv_records = _collect_from_csv(base_dir, csv_path)
            writer.write(f"Total de linhas no CSV: {len(csv_records)}")

            writer.write("\n--- Contagem por split/label (CSV) ---")
            writer.write(_counts_table_str(csv_records))

            csv_missing_in_fs, fs_missing_in_csv, unreadable = _compare_sources(
                fs_records, csv_records
            )

            writer.write("")
            writer.write(
                f"CSV->FS: arquivos listados no CSV mas NÃO encontrados no filesystem: {len(csv_missing_in_fs)}"
            )
            writer.write(
                f"FS->CSV: arquivos no filesystem mas NÃO presentes no CSV: {len(fs_missing_in_csv)}"
            )
            if unreadable:
                writer.write(
                    f"FS: caminhos que não existem (deveria ser 0): {len(unreadable)}"
                )

            if args.save_csv:
                results_dir.mkdir(parents=True, exist_ok=True)
                if csv_missing_in_fs:
                    out_path = results_dir / "eda_missing_in_fs.csv"
                    with open(out_path, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["filepath", "split", "label", "abs_path"])
                        w.writeheader()
                        for r in csv_missing_in_fs:
                            w.writerow(
                                {
                                    "filepath": r.rel_path,
                                    "split": r.split,
                                    "label": r.label,
                                    "abs_path": str(r.abs_path),
                                }
                            )

                if fs_missing_in_csv:
                    out_path = results_dir / "eda_missing_in_csv.csv"
                    with open(out_path, "w", encoding="utf-8", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=["filepath", "split", "label", "abs_path"])
                        w.writeheader()
                        for r in fs_missing_in_csv:
                            w.writerow(
                                {
                                    "filepath": r.rel_path,
                                    "split": r.split,
                                    "label": r.label,
                                    "abs_path": str(r.abs_path),
                                }
                            )

        else:
            writer.write("--- CSV ---")
            writer.write("dataset.csv não encontrado (ok; análise via filesystem apenas).")
            writer.write("")

        # Leitura das imagens (amostra)
        if args.no_image_read:
            writer.write("--- Amostra de imagens ---")
            writer.write("Leitura de imagens desativada (--no-image-read).")
            writer.write("")
            return 0

        if args.sample_size <= 0:
            writer.write("--- Amostra de imagens ---")
            writer.write("sample-size <= 0; pulando leitura de imagens.")
            writer.write("")
            return 0

        writer.write("--- Amostra de imagens (leitura + stats) ---")
        writer.write(f"Amostra: {min(args.sample_size, len(fs_records))} de {len(fs_records)}")

        stats_df, issues_df = _sample_image_stats(
            fs_records, sample_size=args.sample_size, seed=args.seed
        )

        if pd is None:
            if not stats_df:
                writer.write("\n[INFO] Pacotes 'opencv-python' e 'numpy' não parecem instalados; pulei leitura de imagens.")
                writer.write("      Para habilitar stats de imagem, instale as dependências: pip install -r ../requirements.txt")
                writer.write("")
                return 0
        else:
            if getattr(stats_df, "empty", True):
                writer.write("\n[INFO] Não foi possível coletar stats de imagem (amostra vazia).")
                writer.write("")
                return 0

        if pd is not None and not stats_df.empty:
            writer.write("")
            writer.write("Resumo de tamanhos (amostra):")
            writer.write(
                stats_df[["height", "width", "channels"]]
                .describe(percentiles=[0.1, 0.5, 0.9])
                .to_string()
            )

            writer.write("")
            writer.write("Resumo de intensidade (amostra, escala de cinza):")
            writer.write(
                stats_df[["mean_intensity", "std_intensity"]]
                .describe(percentiles=[0.1, 0.5, 0.9])
                .to_string()
            )

            writer.write("")
            writer.write("Top resoluções por split/label (amostra, top-10):")
            writer.write(_resolution_summary_str(stats_df))

        if pd is not None and issues_df is not None and not issues_df.empty:
            writer.write("")
            writer.write(f"Arquivos com problemas na amostra: {len(issues_df)}")
            writer.write(issues_df.head(30).to_string(index=False))

        if args.save_csv:
            results_dir.mkdir(parents=True, exist_ok=True)
            if pd is not None:
                if not stats_df.empty:
                    stats_df.to_csv(results_dir / "eda_sample_image_stats.csv", index=False)
                if issues_df is not None and not issues_df.empty:
                    issues_df.to_csv(results_dir / "eda_sample_image_issues.csv", index=False)

        writer.write("")
        writer.write(f"Relatório salvo em: {summary_path}")
        if args.save_csv:
            writer.write(f"CSVs salvos em: {results_dir}")

        return 0
    finally:
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
