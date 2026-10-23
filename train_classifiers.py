"""Eğitir ve karşılaştırır dört duygu sınıflandırma modelini."""

import argparse
import csv
import logging
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from scipy.sparse import load_npz


LOGGER = logging.getLogger(__name__)
MODEL_FACTORIES = {
    "naive_bayes": MultinomialNB,
    "lojistik_regresyon": lambda: LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "svm": lambda: LinearSVC(random_state=42),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1
    ),
}


def positive_class_scores(model: object, x_test: object, positive_label: str):
    """Olumlu sınıfa ait olasılık/karar skorlarını ROC-AUC için döndürür."""
    classes = model.classes_
    if positive_label not in classes:
        raise ValueError(
            f"Pozitif sınıf {positive_label!r} eğitim etiketlerinde bulunamadı."
        )

    positive_index = list(classes).index(positive_label)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, positive_index]

    decision_scores = model.decision_function(x_test)
    return decision_scores if positive_index == 1 else -decision_scores


def read_labels_and_comments(
    csv_path: Path,
) -> tuple[list[str], list[str]]:
    """Bölünmüş CSV'den yorumları ve gerçek etiketleri okur."""
    comments: list[str] = []
    labels: list[str] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file, strict=True)
        required_columns = {"yorum", "etiket"}
        if reader.fieldnames is None or not required_columns.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{csv_path} dosyasında 'yorum' ve 'etiket' sütunları bulunmalı."
            )

        for row_number, row in enumerate(reader, start=2):
            comment = row.get("yorum")
            label = row.get("etiket")
            if (
                None in row
                or not isinstance(comment, str)
                or not isinstance(label, str)
                or not comment.strip()
                or not label.strip()
            ):
                raise ValueError(
                    f"{csv_path} dosyasının {row_number}. satırında "
                    "yorum veya etiket boş ya da eksik."
                )
            comments.append(comment.strip())
            labels.append(label.strip())

    if not labels:
        raise ValueError(f"{csv_path} dosyasında kayıt bulunamadı.")
    return comments, labels


def train_and_evaluate(
    dataset_dir: Path,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    """Dört sınıflandırıcıyı eğitir, test tahminlerini ve ölçümleri kaydeder."""
    x_train = load_npz(dataset_dir / "X_train_tfidf.npz")
    x_test = load_npz(dataset_dir / "X_test_tfidf.npz")
    _, y_train = read_labels_and_comments(dataset_dir / "egitim_verisi.csv")
    test_comments, y_test = read_labels_and_comments(
        dataset_dir / "test_verisi.csv"
    )
    if x_train.shape[0] != len(y_train):
        raise ValueError("Eğitim TF-IDF satır sayısı ile etiket sayısı eşleşmiyor.")
    if x_test.shape[0] != len(y_test):
        raise ValueError("Test TF-IDF satır sayısı ile etiket sayısı eşleşmiyor.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Eğitim ve test TF-IDF özellik sayıları eşleşmiyor.")
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise ValueError("ROC-AUC hesaplamak için eğitim ve testte iki sınıf olmalı.")
    positive_label = "olumlu"
    if positive_label not in y_train or positive_label not in y_test:
        raise ValueError(
            f"ROC-AUC hesaplamak için pozitif sınıf {positive_label!r} "
            "eğitim ve test etiketlerinde bulunmalı."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, list[str]] = {}
    metrics: dict[str, dict[str, float]] = {}
    for model_name, create_model in MODEL_FACTORIES.items():
        model = create_model()
        model.fit(x_train, y_train)
        y_train_pred = model.predict(x_train).tolist()
        y_pred = model.predict(x_test).tolist()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0,
        )
        positive_scores = positive_class_scores(model, x_test, positive_label)
        predictions[model_name] = y_pred
        metrics[model_name] = {
            "training_accuracy": float(accuracy_score(y_train, y_train_pred)),
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "roc_auc": float(
                roc_auc_score(
                    [label == positive_label for label in y_test],
                    positive_scores,
                )
            ),
        }
        joblib.dump(model, output_dir / f"{model_name}.joblib")
        LOGGER.info(
            "%s: eğitim doğruluğu=%.4f, test doğruluğu=%.4f, macro F1=%.4f",
            model_name,
            metrics[model_name]["training_accuracy"],
            metrics[model_name]["test_accuracy"],
            metrics[model_name]["f1_macro"],
        )

    prediction_fields = ["yorum", "gercek_etiket", *MODEL_FACTORIES]
    with (output_dir / "test_tahminleri.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=prediction_fields)
        writer.writeheader()
        for index, (comment, actual_label) in enumerate(
            zip(test_comments, y_test, strict=True)
        ):
            row = {"yorum": comment, "gercek_etiket": actual_label}
            row.update(
                {
                    model_name: model_predictions[index]
                    for model_name, model_predictions in predictions.items()
                }
            )
            writer.writerow(row)

    metric_fields = [
        "model",
        "training_accuracy",
        "test_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
    ]
    with (output_dir / "model_metrikleri.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=metric_fields)
        writer.writeheader()
        ranked_metrics = sorted(
            metrics.items(),
            key=lambda item: (
                item[1]["test_accuracy"],
                item[1]["f1_macro"],
                item[1]["roc_auc"],
                item[1]["precision_macro"],
            ),
            reverse=True,
        )
        for model_name, model_metrics in ranked_metrics:
            writer.writerow({"model": model_name, **model_metrics})

    LOGGER.info("Tahminler ve ölçümler kaydedildi: %s", output_dir)
    return metrics


def main() -> None:
    """Komut satırı seçeneklerini alır ve modelleri eğitip değerlendirir."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Naive Bayes, lojistik regresyon, doğrusal SVM ve Random Forest "
            "modellerini TF-IDF verisiyle eğitir ve test eder."
        )
    )
    parser.add_argument(
        "dataset_dir",
        nargs="?",
        type=Path,
        default=project_dir / "veri_seti",
        help="Hazırlanmış veri klasörü (varsayılan: veri_seti)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=project_dir / "veri_seti" / "siniflandirma",
        help="Model ve sonuç klasörü (varsayılan: veri_seti/siniflandirma)",
    )
    args = parser.parse_args()
    train_and_evaluate(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
