"""Türkçe yorum duygu analizi çalışmaları için yerel Streamlit paneli."""

import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "veri_seti"
MODEL_DIR = DATA_DIR / "siniflandirma"
POSITIVE_LABEL = "olumlu"
MODEL_LABELS = {
    "naive_bayes": "Naive Bayes",
    "lojistik_regresyon": "Lojistik Regresyon",
    "svm": "SVM (LinearSVC)",
    "random_forest": "Random Forest",
}
PAGES = [
    "Genel Bakış",
    "Metin Ön İşleme",
    "Veri Setinin Hazırlanması",
    "Makine Öğrenmesi ve Duygu Analizi",
    "Eğitim ve Model Başarımı",
    "Değerlendirme Metrikleri",
    "Sonuç ve Karşılaştırma",
]


st.set_page_config(
    page_title="Türkçe Yorum Duygu Analizi",
    page_icon=":material/insights:",
    layout="wide",
)


@st.cache_data(ttl=60, max_entries=16)
def load_csv(path_string: str) -> pd.DataFrame:
    """CSV dosyasını kısa süreli önbellekle yükler."""
    return pd.read_csv(path_string, encoding="utf-8-sig", keep_default_na=False)


@st.cache_data(ttl=300, max_entries=2)
def load_readme() -> str:
    """README açıklamalarını panel içeriği olarak okur."""
    return (PROJECT_DIR / "README.md").read_text(encoding="utf-8")


@st.cache_resource(max_entries=4)
def load_model(path_string: str):
    """Yerel eğitim çıktılarından seçilen modeli bir kez yükler."""
    return joblib.load(path_string)


def readme_section(title: str) -> str:
    """README'den belirtilen başlığı ve varsa alt başlıklarını döndürür."""
    lines = load_readme().splitlines()
    target = title.casefold()
    start = None
    section_level = 0

    for index, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match and match.group(2).casefold() == target:
            start = index
            section_level = len(match.group(1))
            break

    if start is None:
        return ""

    for index in range(start + 1, len(lines)):
        match = re.match(r"^(#{1,6})\s+", lines[index])
        if match and len(match.group(1)) <= section_level:
            return "\n".join(lines[start:index]).strip()
    return "\n".join(lines[start:]).strip()


def show_readme_section(title: str) -> None:
    """README'deki seçilmiş bölümün açıklamasını gösterir."""
    section = readme_section(title)
    if section:
        st.markdown(section)
    else:
        st.info(f"README içinde “{title}” bölümü bulunamadı.")


def load_optional_csv(filename: str) -> pd.DataFrame | None:
    """İsteğe bağlı bir CSV varsa yükler, yoksa boş durumu bildirir."""
    path = PROJECT_DIR / filename
    if not path.is_file():
        return None
    return load_csv(str(path))


def metric_frame() -> pd.DataFrame | None:
    path = MODEL_DIR / "model_metrikleri.csv"
    if not path.is_file():
        return None
    return load_csv(str(path))


def show_pipeline_actions() -> None:
    st.subheader("Çalıştırma işlemleri")
    left, right = st.columns(2)
    with left:
        if st.button(
            "Veri setini yeniden hazırla",
            key="prepare_dataset",
            width="stretch",
        ):
            from prepare_dataset import prepare_dataset

            with st.spinner("Eğitim/test verileri ve TF-IDF matrisleri hazırlanıyor..."):
                prepare_dataset(
                    PROJECT_DIR / "sentetik_yorumlar_onislenmis.csv",
                    DATA_DIR,
                )
            st.cache_data.clear()
            st.success("Veri seti hazırlandı.")
            st.rerun()

    with right:
        if st.button(
            "Modelleri yeniden eğit",
            key="train_models",
            width="stretch",
        ):
            from train_classifiers import train_and_evaluate

            with st.spinner("Dört sınıflandırıcı eğitiliyor ve değerlendiriliyor..."):
                train_and_evaluate(DATA_DIR, MODEL_DIR)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Modeller eğitildi; tahminler ve metrikler güncellendi.")
            st.rerun()


def render_overview() -> None:
    st.title("Türkçe yorum duygu analizi")
    st.write(
        "Metin ön işlemeden model değerlendirmesine kadar proje çalışmalarını "
        "ve güncel sonuçları tek panelde görüntüleyin."
    )

    raw = load_optional_csv("sentetik_yorumlar.csv")
    metrics = metric_frame()
    if raw is not None and "etiket" in raw:
        counts = raw["etiket"].value_counts()
        total = len(raw)
        positive_count = int(counts.get("olumlu", 0))
        negative_count = int(counts.get("olumsuz", 0))
    else:
        counts = pd.Series(dtype="int64")
        total = 0
        positive_count = 0
        negative_count = 0

    with st.container(horizontal=True):
        st.metric("Yorum sayısı", f"{total:,}", border=True)
        st.metric("Olumlu", f"{positive_count:,}", border=True)
        st.metric("Olumsuz", f"{negative_count:,}", border=True)
        if metrics is not None and "test_accuracy" in metrics:
            best_accuracy = metrics["test_accuracy"].astype(float).max()
            st.metric("En yüksek test doğruluğu", f"{best_accuracy:.4f}", border=True)
        else:
            st.metric("Model sonuçları", "Henüz yok", border=True)

    if not counts.empty:
        st.subheader("Veri kümesindeki duygu dağılımı")
        chart_data = counts.rename_axis("duygu").reset_index(name="yorum_sayisi")
        st.bar_chart(
            chart_data,
            x="duygu",
            y="yorum_sayisi",
            alt="Olumlu ve olumsuz yorum sayıları",
        )

    st.subheader("Çalışma adımları")
    st.markdown(
        """
        1. **Metin ön işleme:** Türkçe yorumları temizleme, belirteçlere ayırma,
           durak kelimeleri çıkarma, kök bulma ve lemmatizasyon.
        2. **Veri seti hazırlama:** Özellik (`X`) ve etiket (`y`) oluşturma,
           katmanlı eğitim/test bölmesi ve TF-IDF dönüşümü.
        3. **Sınıflandırma:** Naive Bayes, Lojistik Regresyon, SVM ve Random Forest
           modellerini eğitme ve yorumları sınıflandırma.
        4. **Değerlendirme:** Eğitim/test doğruluğu, Precision, Recall, F1 ve
           ROC-AUC sonuçlarını karşılaştırma.
        """
    )
    show_pipeline_actions()


def render_preprocessing() -> None:
    st.title("Metin ön işleme")
    show_readme_section("Metin ön işleme")

    processed = load_optional_csv("sentetik_yorumlar_onislenmis.csv")
    if processed is None:
        st.warning(
            "Ön işlenmiş CSV bulunamadı. Önce aşağıdaki düğmeyle çalıştırın."
        )
        if st.button("Ön işlemeyi çalıştır", width="stretch"):
            from preprocess import preprocess_csv

            with st.spinner("Yorumlar Türkçe metin araçlarıyla işleniyor..."):
                processed_count, skipped_count = preprocess_csv(
                    PROJECT_DIR / "sentetik_yorumlar.csv",
                    PROJECT_DIR / "sentetik_yorumlar_onislenmis.csv",
                )
            st.cache_data.clear()
            st.success(
                f"{processed_count:,} satır işlendi; "
                f"{skipped_count:,} geçersiz satır atlandı."
            )
            st.rerun()
        return

    st.subheader("Ön işleme çıktısından örnekler")
    st.caption(f"Toplam {len(processed):,} işlenmiş yorum.")
    columns = [
        column
        for column in (
            "yorum",
            "etiket",
            "yorum_temiz",
            "durak_kelimeler_temiz",
            "kokler",
            "lemmatizasyon",
        )
        if column in processed.columns
    ]
    row_count = st.slider(
        "Gösterilecek örnek sayısı",
        min_value=5,
        max_value=min(100, max(5, len(processed))),
        value=min(10, max(5, len(processed))),
    )
    st.dataframe(
        processed[columns].head(row_count),
        hide_index=True,
        height=400,
        alt="Ön işlenmiş yorum örnekleri ve aşama çıktıları",
    )


def render_dataset_preparation() -> None:
    st.title("Veri setinin hazırlanması")
    show_readme_section("Veri setinin hazırlanması")

    train_path = DATA_DIR / "egitim_verisi.csv"
    test_path = DATA_DIR / "test_verisi.csv"
    if not train_path.is_file() or not test_path.is_file():
        st.warning("Eğitim/test CSV dosyaları henüz oluşturulmamış.")
        if st.button("Veri setini hazırla", width="stretch"):
            from prepare_dataset import prepare_dataset

            with st.spinner("Veri bölünüyor ve TF-IDF özellikleri üretiliyor..."):
                prepare_dataset(
                    PROJECT_DIR / "sentetik_yorumlar_onislenmis.csv",
                    DATA_DIR,
                )
            st.cache_data.clear()
            st.success("Eğitim ve test verileri oluşturuldu.")
            st.rerun()
        return

    train = load_csv(str(train_path))
    test = load_csv(str(test_path))
    with st.container(horizontal=True):
        st.metric("Eğitim kaydı", f"{len(train):,}", border=True)
        st.metric("Test kaydı", f"{len(test):,}", border=True)

    distribution = pd.DataFrame(
        [
            {"bölüm": "Eğitim", **train["etiket"].value_counts().to_dict()},
            {"bölüm": "Test", **test["etiket"].value_counts().to_dict()},
        ]
    ).fillna(0)
    label_columns = [label for label in ("olumlu", "olumsuz") if label in distribution]
    st.subheader("Bölümlere göre sınıf dağılımı")
    if label_columns:
        st.bar_chart(
            distribution,
            x="bölüm",
            y=label_columns,
            alt="Eğitim ve test verilerindeki olumlu ve olumsuz kayıt sayıları",
        )

    matrix_path = DATA_DIR / "X_train_tfidf.npz"
    if matrix_path.is_file():
        from scipy.sparse import load_npz

        matrix = load_npz(matrix_path)
        st.info(
            f"Eğitim TF-IDF matrisi: {matrix.shape[0]:,} yorum × "
            f"{matrix.shape[1]:,} özellik; seyrek matris biçiminde."
        )

    with st.expander("Eğitim verisinden örnekler"):
        st.dataframe(
            train.head(10),
            hide_index=True,
            alt="Eğitim kümesinden örnek yorum, özellik metni ve etiketler",
        )
    with st.expander("Test verisinden örnekler"):
        st.dataframe(
            test.head(10),
            hide_index=True,
            alt="Test kümesinden örnek yorum, özellik metni ve etiketler",
        )
    if st.button("Veri setini yeniden hazırla", width="stretch"):
        from prepare_dataset import prepare_dataset

        with st.spinner("Veri bölünüyor ve TF-IDF özellikleri üretiliyor..."):
            prepare_dataset(
                PROJECT_DIR / "sentetik_yorumlar_onislenmis.csv",
                DATA_DIR,
            )
        st.cache_data.clear()
        st.success("Eğitim ve test verileri güncellendi.")
        st.rerun()


def render_classification() -> None:
    st.title("Makine öğrenmesi ve duygu analizi")
    show_readme_section("Makine öğrenmesi ile sınıflandırma ve duygu analizi")

    st.subheader("Yeni bir yorumu sınıflandır")
    st.caption(
        "Tahmin için eğitimdekiyle aynı Türkçe ön işleme ve TF-IDF dönüşümü uygulanır."
    )
    with st.form("sentiment_prediction"):
        comment = st.text_area(
            "Türkçe yorum",
            placeholder="Örnek: Telefonun kamerası ve pil performansı çok iyi.",
            height=120,
        )
        model_key = st.selectbox(
            "Sınıflandırma modeli",
            options=list(MODEL_LABELS),
            format_func=lambda value: MODEL_LABELS[value],
        )
        submitted = st.form_submit_button("Duyguyu tahmin et", width="stretch")

    if submitted:
        if not comment.strip():
            st.warning("Tahmin için bir yorum girin.")
        elif not (DATA_DIR / "tfidf_vectorizer.joblib").is_file():
            st.error("TF-IDF vektörleştiricisi bulunamadı. Önce veri setini hazırlayın.")
        elif not (MODEL_DIR / f"{model_key}.joblib").is_file():
            st.error("Seçilen model bulunamadı. Önce modelleri eğitin.")
        else:
            from preprocess import TurkishTextPreprocessor

            with st.spinner("Yorum ön işleniyor ve sınıflandırılıyor..."):
                processed = TurkishTextPreprocessor().process(comment, "")
                feature_text = processed["lemmatizasyon"]
                if not feature_text:
                    st.error("Ön işleme sonrasında yorumdan özellik üretilemedi.")
                else:
                    vectorizer = load_model(
                        str(DATA_DIR / "tfidf_vectorizer.joblib")
                    )
                    model = load_model(str(MODEL_DIR / f"{model_key}.joblib"))
                    vector = vectorizer.transform([feature_text])
                    prediction = model.predict(vector)[0]
                    if prediction == POSITIVE_LABEL:
                        st.success("Tahmin edilen duygu: Olumlu")
                    else:
                        st.info("Tahmin edilen duygu: Olumsuz")
                    with st.expander("Ön işlenmiş metni görüntüle"):
                        st.write(feature_text)

    show_test_predictions()


def show_test_predictions() -> None:
    prediction_path = MODEL_DIR / "test_tahminleri.csv"
    if not prediction_path.is_file():
        st.info("Test tahminleri henüz yok. Modelleri eğiterek oluşturabilirsiniz.")
        return

    predictions = load_csv(str(prediction_path))
    model_keys = [key for key in MODEL_LABELS if key in predictions.columns]
    if not model_keys:
        st.warning("Tahmin dosyasında model sütunları bulunamadı.")
        return

    st.subheader("Test yorumları ve model tahminleri")
    selected_model = st.selectbox(
        "Tahminleri göster",
        options=model_keys,
        format_func=lambda value: MODEL_LABELS[value],
        key="prediction_model",
    )
    correct = (
        predictions[selected_model] == predictions["gercek_etiket"]
    ).mean()
    st.metric("Seçilen modelin test doğruluğu", f"{correct:.4f}")
    filter_value = st.selectbox(
        "Sonuç filtresi",
        options=["Tüm yorumlar", "Doğru tahminler", "Yanlış tahminler"],
    )
    visible = predictions
    is_correct = predictions[selected_model] == predictions["gercek_etiket"]
    if filter_value == "Doğru tahminler":
        visible = predictions[is_correct]
    elif filter_value == "Yanlış tahminler":
        visible = predictions[~is_correct]

    st.dataframe(
        visible[["yorum", "gercek_etiket", selected_model]].head(100),
        hide_index=True,
        height=420,
        alt="Test yorumlarının gerçek etiketleri ve seçili model tahminleri",
    )


def render_training_results() -> None:
    st.title("Eğitim ve model başarım sonuçları")
    show_readme_section("Eğitim süreci ve model başarımının karşılaştırılması")

    metrics = metric_frame()
    if metrics is not None and not metrics.empty:
        display_metrics(metrics)
    else:
        st.info("Başarım sonuçları bulunamadı. Modelleri eğiterek oluşturun.")
    show_pipeline_actions()


def display_metrics(metrics: pd.DataFrame) -> None:
    display = metrics.copy()
    if "model" in display:
        display["algoritma"] = display["model"].map(
            lambda value: MODEL_LABELS.get(value, value)
        )
    numeric_columns = [
        column
        for column in (
            "training_accuracy",
            "test_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
        )
        if column in display.columns
    ]
    for column in numeric_columns:
        display[column] = pd.to_numeric(display[column], errors="coerce")

    metric_columns = [
        column
        for column in (
            "training_accuracy",
            "test_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
        )
        if column in display.columns
    ]
    if metric_columns:
        metric_labels = {
            "training_accuracy": "Eğitim doğruluğu",
            "test_accuracy": "Test doğruluğu",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "f1_macro": "Macro F1",
            "roc_auc": "ROC-AUC (olumlu)",
        }
        column_config = {
            column: st.column_config.NumberColumn(
                label=metric_labels[column],
                format="%.6f",
            )
            for column in metric_columns
        }
        st.dataframe(
            display[["algoritma", *metric_columns]],
            hide_index=True,
            column_config=column_config,
            alt="Sınıflandırıcıların eğitim ve test başarı metrikleri",
        )


def render_metrics() -> None:
    st.title("Değerlendirme metrikleri")
    show_readme_section("Makine öğrenmesi ile sınıflandırma ve duygu analizi")

    metrics = metric_frame()
    if metrics is None or metrics.empty:
        st.warning("Metrik dosyası bulunamadı. Önce modelleri eğitin.")
        show_pipeline_actions()
        return

    display_metrics(metrics)
    chart = metrics.copy()
    chart["algoritma"] = chart["model"].map(
        lambda value: MODEL_LABELS.get(value, value)
    )
    metric_columns = [
        column
        for column in (
            "test_accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "roc_auc",
        )
        if column in chart.columns
    ]
    selected_metric = st.selectbox(
        "Grafikte karşılaştırılacak ölçüm",
        options=metric_columns,
        format_func=lambda value: {
            "test_accuracy": "Test doğruluğu",
            "precision_macro": "Macro Precision",
            "recall_macro": "Macro Recall",
            "f1_macro": "Macro F1",
            "roc_auc": "ROC-AUC (olumlu)",
        }.get(value, value),
    )
    chart[selected_metric] = pd.to_numeric(chart[selected_metric], errors="coerce")
    st.bar_chart(
        chart,
        x="algoritma",
        y=selected_metric,
        alt="Seçilen başarı metriğinin algoritmalara göre karşılaştırması",
    )

    prediction_path = MODEL_DIR / "test_tahminleri.csv"
    if prediction_path.is_file():
        predictions = load_csv(str(prediction_path))
        available_models = [
            model for model in MODEL_LABELS if model in predictions.columns
        ]
        selected_model = st.selectbox(
            "Karışıklık matrisi modeli",
            options=available_models,
            format_func=lambda value: MODEL_LABELS[value],
            key="confusion_model",
        )
        matrix = confusion_matrix(
            predictions["gercek_etiket"],
            predictions[selected_model],
            labels=["olumlu", "olumsuz"],
        )
        matrix_frame = pd.DataFrame(
            matrix,
            index=["Gerçek olumlu", "Gerçek olumsuz"],
            columns=["Tahmin olumlu", "Tahmin olumsuz"],
        )
        st.subheader("Karışıklık matrisi")
        st.dataframe(
            matrix_frame,
            alt="Gerçek sınıflar ile tahmin edilen sınıfların sayısal tablosu",
        )


def render_comparison() -> None:
    st.title("Sonuç ve karşılaştırma")
    show_readme_section("Sonuç ve karşılaştırma")

    metrics = metric_frame()
    if metrics is not None and not metrics.empty:
        st.subheader("Ölçümlere göre model karşılaştırması")
        chart = metrics.copy()
        chart["algoritma"] = chart["model"].map(
            lambda value: MODEL_LABELS.get(value, value)
        )
        chart["test_accuracy"] = pd.to_numeric(
            chart["test_accuracy"], errors="coerce"
        )
        st.bar_chart(
            chart,
            x="algoritma",
            y="test_accuracy",
            alt="Algoritmaların test doğruluğu karşılaştırması",
        )


def main() -> None:
    with st.sidebar:
        st.title("Yorum analizi")
        selected_page = st.selectbox("Panel bölümü", PAGES)
        st.divider()
        st.caption("Kaynak: README ve proje çıktıları")

    renderers = {
        "Genel Bakış": render_overview,
        "Metin Ön İşleme": render_preprocessing,
        "Veri Setinin Hazırlanması": render_dataset_preparation,
        "Makine Öğrenmesi ve Duygu Analizi": render_classification,
        "Eğitim ve Model Başarımı": render_training_results,
        "Değerlendirme Metrikleri": render_metrics,
        "Sonuç ve Karşılaştırma": render_comparison,
    }
    renderers[selected_page]()


main()
