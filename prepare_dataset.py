"""Yorumları eğitim/test kümelerine ayırır ve TF-IDF özellikleri üretir."""

import argparse
import csv
import logging
from collections import Counter
from pathlib import Path

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit


LOGGER = logging.getLogger(__name__)
OUTPUT_FIELDS = ("yorum", "ozellik_metni", "etiket")


def read_rows(input_path: Path, text_column: str) -> list[dict[str, str]]:
    """Ön işlenmiş CSV'den metin ve duygu etiketi bulunan satırları okur."""
    rows: list[dict[str, str]] = []

    with input_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, strict=True)
        required_columns = {"yorum", "etiket", text_column}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            missing_columns = required_columns - set(reader.fieldnames or ())
            raise ValueError(
                "CSV dosyasında gerekli sütunlar bulunmalı: "
                + ", ".join(sorted(missing_columns))
            )

        for row_number, row in enumerate(reader, start=2):
            comment = row.get("yorum")
            feature_text = row.get(text_column)
            label = row.get("etiket")
            if (
                None in row
                or not isinstance(comment, str)
                or not isinstance(feature_text, str)
                or not isinstance(label, str)
                or not comment.strip()
                or not feature_text.strip()
                or not label.strip()
            ):
                raise ValueError(
                    f"CSV satırı {row_number} içinde yorum, özellik metni "
                    "ve etiket boş ya da eksik."
                )

            rows.append(
                {
                    "yorum": comment.strip(),
                    "ozellik_metni": feature_text.strip(),
                    "etiket": label.strip(),
                }
            )

    if not rows:
        raise ValueError("CSV dosyasında kullanılabilir veri bulunamadı.")

    label_counts = Counter(row["etiket"] for row in rows)
    if len(label_counts) < 2 or min(label_counts.values()) < 2:
        raise ValueError(
            "Katmanlı eğitim/test ayırımı için en az iki sınıf ve "
            "her sınıfta en az iki kayıt bulunmalı."
        )

    return rows


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Bir veri bölümünü yorum, TF-IDF girdisi ve etiket sütunlarıyla yazar."""
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def grouped_train_test_indices(
    rows: list[dict[str, str]],
    test_size: float,
    random_state: int,
) -> tuple[list[int], list[int]]:
    """Aynı özellik metnini kümeler arasında bölmeden dengeli bir ayırım seçer."""
    group_labels: dict[str, str] = {}
    for row in rows:
        feature_text = row["ozellik_metni"]
        label = row["etiket"]
        previous_label = group_labels.setdefault(feature_text, label)
        if previous_label != label:
            raise ValueError(
                "Aynı özellik metni birden fazla etikete bağlı; "
                "çakışan kayıtlar düzeltilmeden güvenilir bölme yapılamaz."
            )

    labels = [row["etiket"] for row in rows]
    groups = [row["ozellik_metni"] for row in rows]
    total_label_counts = Counter(labels)
    target_test_size = len(rows) * test_size
    splitter = GroupShuffleSplit(
        n_splits=100,
        test_size=test_size,
        random_state=random_state,
    )

    best_split: tuple[list[int], list[int]] | None = None
    best_score = float("inf")
    for train_indices, test_indices in splitter.split(rows, labels, groups):
        test_labels = Counter(labels[index] for index in test_indices)
        if set(test_labels) != set(total_label_counts):
            continue

        size_error = abs(len(test_indices) - target_test_size) / len(rows)
        distribution_error = sum(
            abs(
                test_labels[label] / len(test_indices)
                - total_label_counts[label] / len(rows)
            )
            for label in total_label_counts
        )
        score = size_error + distribution_error
        if score < best_score:
            best_score = score
            best_split = (train_indices.tolist(), test_indices.tolist())

    if best_split is None:
        raise ValueError(
            "Aynı metinleri birlikte tutarak iki sınıfı eğitim ve test "
            "kümelerine dağıtan bir bölme bulunamadı."
        )

    return best_split


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    text_column: str = "lemmatizasyon",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[int, int, int]:
    """Veriyi ayırır, TF-IDF matrislerini ve eğitilmiş vektörleştiriciyi kaydeder."""
    rows = read_rows(input_path, text_column)
    x = [row["ozellik_metni"] for row in rows]
    train_indices, test_indices = grouped_train_test_indices(
        rows,
        test_size=test_size,
        random_state=random_state,
    )
    train_rows = [rows[index] for index in train_indices]
    test_rows = [rows[index] for index in test_indices]
    x_train_texts = [x[index] for index in train_indices]
    x_test_texts = [x[index] for index in test_indices]

    vectorizer = TfidfVectorizer()
    # Sözlüğü sadece eğitim metinlerinden öğrenmek test verisi sızıntısını önler.
    x_train = vectorizer.fit_transform(x_train_texts)
    x_test = vectorizer.transform(x_test_texts)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(output_dir / "egitim_verisi.csv", train_rows)
    write_split_csv(output_dir / "test_verisi.csv", test_rows)
    save_npz(output_dir / "X_train_tfidf.npz", x_train)
    save_npz(output_dir / "X_test_tfidf.npz", x_test)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")

    LOGGER.info(
        "%s eğitim ve %s test kaydı hazırlandı; TF-IDF boyutu %s özellik.",
        len(train_rows),
        len(test_rows),
        x_train.shape[1],
    )
    LOGGER.info("Çıktı klasörü: %s", output_dir)
    return len(train_rows), len(test_rows), x_train.shape[1]


def main() -> None:
    """Komut satırı seçeneklerini alır ve veri hazırlama akışını çalıştırır."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Ön işlenmiş yorumları eğitim/test kümelerine ayırır ve "
            "TF-IDF vektörleri üretir."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=project_dir / "sentetik_yorumlar_onislenmis.csv",
        help="Ön işlenmiş CSV dosyası (varsayılan: sentetik_yorumlar_onislenmis.csv)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=project_dir / "veri_seti",
        help="Çıktı klasörü (varsayılan: veri_seti)",
    )
    parser.add_argument(
        "--text-column",
        default="lemmatizasyon",
        help="TF-IDF için metin sütunu (varsayılan: lemmatizasyon)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test kümesinin oranı (varsayılan: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Tekrarlanabilir bölme için rastgelelik tohumu (varsayılan: 42)",
    )
    args = parser.parse_args()

    prepare_dataset(
        args.input,
        args.output_dir,
        text_column=args.text_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
