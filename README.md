# yorum_siniflandirma
yorum_siniflandirma

## Sentetik veri kümesi

`sentetik_yorumlar.csv` dosyası, duygu sınıflandırması için hazırlanmış 10.000
sentetik Türkçe akıllı telefon yorumu içerir: 5.789 `olumlu` ve 4.211
`olumsuz` kayıt. Yorumlar telefonun pil, ekran, kamera, performans, ses ve
günlük kullanım deneyimine odaklanır. Yorumlarda sabit giriş ve kapanış
kalıpları yerine iki farklı kullanım özelliği birlikte anlatılır. CSV sütunları
`yorum` ve `etiket` şeklindedir.

## Metin ön işleme

Yorumlar `preprocess.py` ile aşağıdaki sırada ön işlenir:

1. **Küçük harfe dönüştürme:** Türkçe `I/İ` harfleri doğru eşlenerek metin
   küçük harfe çevrilir; Unicode karakterler standartlaştırılır.

   ```python
   TURKISH_CASE_MAP = str.maketrans({"I": "ı", "İ": "i"})
   lowered = unicodedata.normalize(
       "NFKC", text.translate(TURKISH_CASE_MAP).lower()
   )
   ```

2. **Noktalama ve özel karakter temizliği:** Türkçe harfler ve sayılar
   korunur; noktalama ve desteklenmeyen karakterler kaldırılır.

   ```python
   TOKEN_PATTERN = re.compile(r"[a-z0-9çğıöşü]+")
   cleaned = " ".join(TOKEN_PATTERN.findall(lowered))
   ```

3. **Tokenization:** NLTK'nin `wordpunct_tokenize` yöntemiyle metin
   belirteçlere ayrılır.

   ```python
   tokens = [
       token for token in wordpunct_tokenize(cleaned)
       if TOKEN_PATTERN.fullmatch(token)
   ]
   ```

4. **Stop word temizliği:** Belirlenen durak kelimeler çıkarılır. Duygu
   sınıflandırması açısından anlamlı olan `değil`, `yok`, `çok`, `daha`, `en`
   ve `ama` gibi ifadeler korunur.

   ```python
   filtered_tokens = [token for token in tokens if token not in STOP_WORDS]
   ```

5. **Stemming (kök bulma):** TurkishStemmer ile sözcüklerin kök biçimleri
   çıkarılır.

   ```python
   stemmer = TurkishStemmer()
   stems = [stemmer.stem(token) for token in filtered_tokens]
   ```

6. **Lemmatization:** zemberek-python ile sözcükler çözümlenir ve lemma
   biçimleri elde edilir. Tekrarlanan sözcüklerin çözümlemeleri önbelleğe alınır.

   ```python
   analysis = morphology.analyze(token)
   candidates = analysis.analysis_results
   lemma = token if not candidates or candidates[0].is_unknown() \
       else candidates[0].item.lemma
   ```

7. **Boşluk ve hatalı veri temizliği:** Gereksiz boşluklar normalleştirilir.
   Yorumu veya etiketi boş/eksik olan ya da temizleme sonrasında metni
   kalmayan satırlar atlanır; kalan yorum ve etiket değerleri korunur.

   ```python
   comment = row.get("yorum")
   label = row.get("etiket")
   if (not isinstance(comment, str) or not isinstance(label, str)
           or not comment.strip() or not label.strip()
           or not clean_text(comment)):
       continue
   comment, label = comment.strip(), label.strip()
   ```

Ön işleme betiğini çalıştırmak için:

```bash
python -m pip install -r requirements.txt
python preprocess.py
```

Varsayılan olarak `sentetik_yorumlar_onislenmis.csv` oluşturulur. Girdi ve çıktı
yolları komut satırından verilebilir:

```bash
python preprocess.py girdi.csv cikti.csv
```

Çıktıda özgün yorum ve etiketle birlikte `yorum_kucuk_harf`, `yorum_temiz`,
JSON biçimindeki `tokenlar` (NLTK), `durak_kelimeler_temiz`, `kokler`
(TurkishStemmer) ve `lemmatizasyon` (zemberek-python) sütunları bulunur.

## Veri setinin hazırlanması

Bu aşamanın amacı, metin ve etiket sütunlarını makine öğrenmesi algoritmalarının
kullanabileceği eğitim/test verilerine ve sayısal özelliklere dönüştürmektir.
İşlem `prepare_dataset.py` içinde yapılır.

### Temel kavramlar

- **Özellikler (`X`):** Modelin tahmin yapmak için kullandığı girdilerdir. Bu
  çalışmada yorumların ön işlenmiş metinleri özellik olarak kullanılır.
- **Etiketler (`y`):** Modelin tahmin etmeyi öğreneceği doğru sınıflardır. Bu
  veri kümesinde etiketler `olumlu` ve `olumsuz` değerleridir.
- **Eğitim kümesi:** Modeli eğitmek ve TF-IDF sözlüğünü öğrenmek için kullanılan
  veridir.
- **Test kümesi:** Eğitim sırasında kullanılmadan ayrılan, modelin görülmemiş
  verideki başarımını değerlendirmek için saklanan veridir.
- **Gruplu eğitim/test bölmesi:** Aynı ön işlenmiş özellik metnine sahip
  yorumları tek grup kabul eder ve grubun eğitim ile test arasında
  bölünmesini önler. Aday bölmeler arasından hem istenen test oranına hem de
  sınıf dağılımına en yakın olanı seçilir.
- **TF-IDF:** Sözcükleri sayısal değerlere dönüştürür. Bir sözcüğe, ilgili
  yorumda ne kadar belirgin olduğuna ve veri kümesinde ne kadar ayırt edici
  olduğuna göre ağırlık verir. Sonuç, satırları yorumları ve sütunları
  sözcük/özellikleri temsil eden bir matristir.
- **Seyrek matris:** Metin verisinde çoğu yorumda sözcük dağarcığının yalnızca
  küçük bir bölümü geçtiği için, sıfır değerleri saklamayıp belleği verimli
  kullanan matris biçimidir.
- **Veri sızıntısı:** Test verisinden eğitim sırasında bilgi öğrenilmesidir.
  TF-IDF yalnızca eğitim metinlerinde `fit_transform` ile öğrenilir; test
  metinlerine `transform` uygulanır. Böylece test kümesi sözlüğün öğrenilmesine
  katılmaz.

### Kullanılan Python kütüphaneleri ve önemli kodlar

| Kütüphane | Kullanıldığı yer ve amacı |
| --- | --- |
| `scikit-learn` (`sklearn`) | `GroupShuffleSplit`, aynı özellik metinlerini gruplar arasında bölmeden eğitim/test ayırımı yapar. `TfidfVectorizer`, metni TF-IDF özelliklerine dönüştürür. |
| `scipy.sparse` | TF-IDF sonuçlarını bellekte verimli seyrek matris olarak `.npz` biçiminde kaydeder. |
| `joblib` | Eğitilmiş `TfidfVectorizer` nesnesini saklar; yeni veride aynı sözcük sözlüğünü kullanmayı sağlar. |
| `csv` (Python standart kütüphanesi) | CSV dosyalarını sütun adlarıyla okuyup eğitim/test tabloları olarak yazar. |
| `argparse` (Python standart kütüphanesi) | Girdi dosyası, çıktı klasörü, özellik metni sütunu ve test oranı gibi komut satırı seçeneklerini tanımlar. |
| `pathlib` (Python standart kütüphanesi) | Dosya ve klasör yollarını yönetir. |
| `logging` ve `collections.Counter` (Python standart kütüphanesi) | İşlem özetini bildirir ve etiket dağılımını kontrol eder. |
| `NLTK` | Ön işleme aşamasında `wordpunct_tokenize` ile metni belirteçlere ayırır. |
| `TurkishStemmer` | Sözcüklerin eklerini ayırarak kök biçimlerini (`kokler`) üretir. |
| `zemberek-python` | Türkçe morfolojik çözümleme yapıp lemma biçimlerini (`lemmatizasyon`) üretir. |

`scikit-learn`, `scipy` ve `joblib` veri seti hazırlama betiğinde; `NLTK`,
`TurkishStemmer` ve `zemberek-python` ise önceki metin ön işleme aşamasında
kullanılır. Üçüncü taraf paketlerin sürüm aralıkları ve sabitlemeleri
`requirements.txt` dosyasında belirtilmiştir.

Betikteki temel veri akışı şöyledir:

```python
# Ön işlenmiş CSV'den metin özelliklerini (X) ve sınıf etiketlerini (y) al.
x = [row["ozellik_metni"] for row in rows]
y = [row["etiket"] for row in rows]

# Aynı özellik metinlerini kümeler arasında bölmeyen dengeli ayırım.
train_indices, test_indices = grouped_train_test_indices(
    rows,
    test_size=0.2,
    random_state=42,
)

# TF-IDF sözlüğünü yalnızca eğitim verisinde öğren; testte aynı sözlüğü uygula.
x_train = vectorizer.fit_transform(x_train_texts)
x_test = vectorizer.transform(x_test_texts)
```

Varsayılan olarak `lemmatizasyon` sütunu TF-IDF için metin girdisi,
`etiket` sütunu ise hedef sınıf (`y`) olur. Varsayılan hedef oranı %80 eğitim
ve %20 testtir; `random_state=42` bölmenin tekrarlanabilir olmasını sağlar.
Aynı lemmatize özellik metnine sahip yorumlar iki kümeye birden düşmez.

Ön işleme tamamlandıktan sonra aşağıdaki komutla veri seti hazırlanır:

```bash
python prepare_dataset.py
```

Varsayılan olarak `veri_seti/` klasöründe aşağıdaki dosyalar oluşturulur:

- `egitim_verisi.csv` ve `test_verisi.csv`: Bölünmüş yorumlar, TF-IDF'ye
  gönderilen `ozellik_metni` ve karşılık gelen `etiket` değerleri.
- `X_train_tfidf.npz` ve `X_test_tfidf.npz`: Seyrek TF-IDF özellik matrisleri.
  Her satır, aynı bölümdeki CSV dosyasının aynı sıradaki kaydıyla eşleşir.
- `tfidf_vectorizer.joblib`: Testte ve yeni yorumlarda aynı özellik sözlüğünü
  kullanmak için eğitilmiş TF-IDF vektörleştiricisi.

Girdi, çıktı klasörü, özellik metni sütunu ve test oranı değiştirilebilir:

```bash
python prepare_dataset.py girdi.csv cikti_klasoru --text-column kokler --test-size 0.25
```

## Makine öğrenmesi ile sınıflandırma ve duygu analizi

Bu aşamada dört sınıflandırma modeli, aynı eğitim TF-IDF matrisi (`X_train`) ve
etiketleri (`y_train`) ile eğitilir. Daha önce eğitimde kullanılmamış test
yorumlarının TF-IDF matrisleri (`X_test`) her modelden geçirilir; modeller
yorumları `olumlu` veya `olumsuz` olarak tahmin eder. Böylece algoritmalar aynı
veri bölmesi üzerinde karşılaştırılabilir.

### Kullanılan sınıflandırma algoritmaları

- **Naive Bayes (`MultinomialNB`):** Sözcük özelliklerinin sınıfa katkılarını
  olasılıklarla hesaplar. Metin sınıflandırmasında sık kullanılan ve TF-IDF
  gibi negatif olmayan özelliklerle çalışabilen bir yöntemdir.
- **Lojistik Regresyon (`LogisticRegression`):** Özellikleri sınıf olasılıkları
  ile ilişkilendirerek sınıflandırma yapar. Adında regresyon geçse de burada
  olumlu/olumsuz sınıf tahmini için kullanılır.
- **Destek Vektör Makineleri (`LinearSVC`):** Sınıfları ayıran karar sınırını
  bulur. TF-IDF gibi çok sayıda ve seyrek metin özelliğinde verimli bir doğrusal
  SVM uygulamasıdır.
- **Random Forest (`RandomForestClassifier`):** Özellikler üzerinden farklı
  karar ağaçları kurar ve ağaçların tahminlerini birleştirerek sınıfı belirler.

### Kullanılan kütüphaneler ve önemli kodlar

Bu aşama `train_classifiers.py` içinde uygulanır:

| Kütüphane/kod | Amacı |
| --- | --- |
| `sklearn.naive_bayes.MultinomialNB` | Naive Bayes sınıflandırıcısını oluşturur. |
| `sklearn.linear_model.LogisticRegression` | Lojistik regresyon modelini oluşturur. |
| `sklearn.svm.LinearSVC` | Doğrusal SVM sınıflandırıcısını oluşturur. |
| `sklearn.ensemble.RandomForestClassifier` | Random Forest sınıflandırıcısını oluşturur. |
| `model.fit(X_train, y_train)` | Modeli eğitim özellikleri ve doğru etiketlerle eğitir. |
| `model.predict(X_test)` | Test yorumlarının sınıf etiketlerini tahmin eder. |
| `sklearn.metrics` | Eğitim/test doğruluğu, macro precision, macro recall, macro F1 ve ROC-AUC ölçümlerini hesaplar. |
| `predict_proba` / `decision_function` | Sınıf etiketi yerine olumlu sınıfa ait olasılık veya karar skorlarını ROC-AUC için üretir. |
| `scipy.sparse.load_npz` | Önceki aşamada kaydedilen seyrek TF-IDF matrislerini yükler. |
| `joblib` | Eğitilmiş modelleri tekrar kullanılabilmeleri için dosyaya kaydeder. |
| `csv` (Python standart kütüphanesi) | Her test yorumu için gerçek etiketi ve dört modelin tahminini, ayrıca metrikleri CSV olarak yazar. |

Temel eğitim, tahmin ve ölçüm kodu:

```python
model.fit(x_train, y_train)
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
if hasattr(model, "predict_proba"):
    positive_index = list(model.classes_).index("olumlu")
    positive_scores = model.predict_proba(x_test)[:, positive_index]
else:
    decision_scores = model.decision_function(x_test)
    positive_scores = (
        decision_scores
        if model.classes_[1] == "olumlu"
        else -decision_scores
    )

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0,
)
training_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
roc_auc = roc_auc_score(
    [label == "olumlu" for label in y_test],
    positive_scores,
)
```

- **Eğitim doğruluğu (`training_accuracy`):** Eğitim verisinde doğru tahmin
  edilen yorumların oranıdır. Modelin gördüğü verideki uyumunu gösterir;
  tek başına yeni verideki başarımın ölçüsü değildir.
- **Test doğruluğu (`test_accuracy`):** Eğitim sırasında kullanılmamış test
  yorumlarında doğru tahmin edilenlerin oranıdır. Modelleri karşılaştırırken
  temel alınan ölçü budur.
- **Precision (kesinlik):** Bir sınıf için o sınıf olarak tahmin edilen
  yorumların ne kadarının gerçekten o sınıfa ait olduğunu ölçer.
- **Recall (duyarlılık):** Gerçekte bir sınıfa ait yorumların ne kadarının
  doğru bulunduğunu ölçer.
- **F1 skoru:** Precision ve recall değerlerini birlikte özetler. `macro`
  ortalama, her sınıfın ölçümünü eşit ağırlıkla değerlendirir.
- **ROC (Receiver Operating Characteristic):** Farklı karar eşiklerinde
  doğru pozitif oranı (recall) ile yanlış pozitif oranı arasındaki değişimi
  gösterir.
- **AUC (Area Under the Curve):** ROC eğrisinin altında kalan alandır.
  `1.0` kusursuz sıralamayı, `0.5` ise iki sınıfı ayırmada rastgele seviyeyi
  ifade eder. Bu çalışmada `olumlu` pozitif sınıf kabul edilir. AUC, yalnızca
  sınıf tahminlerini değil pozitif sınıfa ait olasılık/karar skorlarını
  kullandığı için `MultinomialNB`, Lojistik Regresyon ve Random Forest'ta
  `predict_proba`; `LinearSVC` modelinde sınıfa göre yönü düzenlenmiş
  `decision_function` skorları kullanılır.

### Eğitim süreci ve model başarımının karşılaştırılması

`train_classifiers.py`, eğitim ve test TF-IDF matrislerini yükler; satır
sayılarının etiketlerle eşleştiğini doğrular. Ardından her sınıflandırıcı için
`fit` çağrısıyla yalnızca eğitim kümesinden öğrenir. Eğitim ve test
doğruluklarını ayrı ayrı hesaplar; test kümesinde ayrıca macro precision,
macro recall, macro F1 ve `olumlu` sınıfı için ROC-AUC değerlerini ölçer.
Eğitim ve testte iki sınıfın da bulunması kontrol edilir. Test tahminleri ve
eğitilmiş modeller de kaydedilir.

`model_metrikleri.csv` modelleri test doğruluğuna, eşitlik durumunda macro F1
ROC-AUC ve macro precision değerlerine göre en iyiden başlayarak sıralar.
Eğitim ve test doğrulukları arasındaki fark, eğitim verisine aşırı uyum
olabileceğini incelemek için kullanılabilir. Model seçimini test ölçümlerine
göre yaptıktan sonra aynı test kümesinde tekrar tekrar ayar yapmak, test
kümesini de dolaylı olarak eğitime dahil eder; nihai tarafsız değerlendirme
için ayrı bir test kümesi veya çapraz doğrulama gerekir.

Mevcut 10.000 yorumlu sentetik veri kümesinde, aynı özellik metinlerini
gruplandıran yeni bölme 8.004 eğitim ve 1.996 test kaydı oluşturdu. Güncel
sonuçlar:

| Algoritma | Eğitim doğruluğu | Test doğruluğu | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (`olumlu`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lojistik Regresyon | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 |
| SVM (`LinearSVC`) | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 |
| Random Forest | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 | 1,000000 |
| Naive Bayes | 0,999125 | 0,998998 | 0,999136 | 0,998811 | 0,998972 | 0,999990 |

Bu ölçümler mevcut sentetik veri ve sabit eğitim/test bölmesi için geçerlidir.
Üç modelin aynı puanı alması, bu veri üzerindeki ölçümlere göre aralarında
başarım farkı gözlenmediği anlamına gelir; daha geniş veya gerçek kullanıcı
yorumlarında aynı sonuçların alınacağını garanti etmez.

### Beklenmedik derecede yüksek puanların kontrolü

İlk rastgele bölmede 2.000 test yorumunun 261'inin ön işlenmiş
`lemmatizasyon` metni eğitim kümesinde de bulunuyordu. Ham yorumların kendileri
yinelenmiyordu; aynı özellik metinleri, farklı yorumların normalizasyon ve
lemmatizasyon sonrasında aynı biçime gelmesinden kaynaklanıyordu. Etiketler
çatışmıyordu, ancak böyle grupların iki kümeye dağılması test başarımını
olduğundan yüksek gösterebilirdi.

Bu nedenle veri hazırlama adımı `GroupShuffleSplit` ile güncellendi. Aynı
özellik metnini taşıyan kayıtlar artık eğitim veya test kümesinden yalnızca
birinde yer alır. Eski bölme üzerindeki ek kontrol olarak, her katmanda TF-IDF
yalnızca o katın eğitim metinlerinde öğrenilecek şekilde beş katlı
`StratifiedGroupKFold` doğrulaması da çalıştırıldı; hiçbir özellik metni
grubu eğitim ve doğrulama arasında paylaşılmadı. Bu kontrolde Lojistik
Regresyon, SVM ve Random Forest her katta 1,000000 doğruluk verdi. Naive
Bayes'in kat doğrulukları 0,9990; 1,0000; 0,9995; 0,9975 ve 0,9995 oldu
(ortalama 0,9991, standart sapma yaklaşık 0,0009).

Dolayısıyla önceki bölmede aynı işlenmiş metinlerin sızması bir değerlendirme
kusuruydu ve düzeltildi; ancak gruplar ayrıldığında da sonuçlar değişmedi.
Mevcut sentetik veri, sınıflandırılabilir olumlu/olumsuz dil kalıplarını
belirgin biçimde içerdiğinden yüksek puanların ana açıklaması veri kümesinin
kolaylığıdır. Bu, gerçek kullanıcı yorumlarında benzer sonuç alınacağını
kanıtlamaz. Ayrıca önceki tabloda ROC-AUC dört ondalığa yuvarlandığı için
Naive Bayes'in 1'den küçük AUC değeri 1,0000 görünüyordu; tabloda artık altı
ondalık kullanılıyor.

## Sonuç ve karşılaştırma

### En başarılı algoritma

Bu deneyde tek bir algoritmayı diğerlerinden daha başarılı ilan etmek mümkün
değildir: **Lojistik Regresyon, SVM (`LinearSVC`) ve Random Forest** test
kümesindeki tabloda gösterilen tüm ölçümlerde 1,000000 alarak eşit sonuç
vermiştir. **Naive Bayes** test doğruluğunda 0,998998, macro F1'de 0,998972
ve olumlu sınıf ROC-AUC değerinde 0,999990 elde etmiştir. Naive Bayes'in
sonuçları da bu veri kümesinde çok yüksektir; görülen küçük fark, gerçek
veride aynı sıralamanın korunacağını göstermez.

Bu nedenle araştırma amacı olan ön işlenmiş Türkçe yorumlardan olumlu/olumsuz
duyguyu tahmin etme bakımından dört yöntem de mevcut sentetik test kümesinde
başarılıdır. Ancak sonuçlar yalnızca bu veri ve sabit bölme için kanıttır;
gerçek kullanıcı yorumlarına genellenebilirlik ayrıca sınanmalıdır.

### Algoritmaların güçlü ve zayıf yönleri

| Algoritma | Güçlü yönleri | Dikkat edilmesi gereken yönleri |
| --- | --- | --- |
| Naive Bayes | Hızlı eğitilir, metin sınıflandırmasında güçlü bir başlangıç/baseline yöntemidir; olasılık skorları ROC-AUC hesabında kullanılabilir. | Özelliklerin sınıfa göre bağımsız olduğu varsayımı gerçek dilde çoğu zaman tam karşılanmaz; olasılık tahminleri kalibrasyon açısından ayrıca incelenmelidir. |
| Lojistik Regresyon | TF-IDF gibi seyrek metin özelliklerinde verimli ve anlaşılabilir bir doğrusal modeldir; sınıf olasılıkları üretir. | Doğrusal karar sınırı karmaşık ilişkileri tek başına yakalayamayabilir; düzenlileştirme ve diğer hiperparametreler ayarlanmadı. |
| SVM (`LinearSVC`) | Yüksek boyutlu ve seyrek metin özelliklerinde etkili bir doğrusal ayırıcıdır; bu deneyde en iyi gruptaki sonuçlardan birini vermiştir. | Doğrudan olasılık üretmez; bu çalışmada ROC-AUC için karar skorları kullanılır. Sonuçları kullanıcıya olasılık gibi sunmak için kalibrasyon gerekir. |
| Random Forest | Birden çok karar ağacını birleştirir ve doğrusal olmayan ilişkileri yakalayabilir. | Bu veri setinde daha karmaşık olmasına rağmen diğer en iyi modelleri geçmemiştir; büyük özellik uzaylarında kaynak tüketimi artabilir ve kararları doğrusal modellere göre daha zor açıklanabilir. |

Bu güçlü/zayıf yönler algoritmaların genel özellikleridir; bu deneydeki puanlar
hangi modelin gerçek dünyada mutlaka üstün olacağını kanıtlamaz. Eşit puanlar
arasında seçim yapılacaksa hız, model boyutu, açıklanabilirlik, tahmin
gereksinimleri ve gerçek verideki doğrulama sonuçları da dikkate alınmalıdır.

### Araştırma amacı açısından yorum

Çalışma, Türkçe akıllı telefon yorumlarının ön işleme ve TF-IDF ile sayısal
özelliklere dönüştürülmesinin ardından, olumlu/olumsuz duygu sınıflandırmasının
klasik makine öğrenmesi algoritmalarıyla yapılabildiğini göstermektedir. Testte
4 modelin de ROC-AUC değerinin 1,0000'a çok yakın veya eşit olması, bu sabit
test kümesindeki olumlu ve olumsuz örneklerin model skorlarıyla çok iyi
ayrıldığını gösterir. Üç modelin kusursuz test ölçümleri ve Naive Bayes'in
bunlara çok yakın sonucu, sentetik veri oluşturma kalıplarının veya belirgin
kelime ipuçlarının sınıflandırmayı kolaylaştırmış olabileceği hipotezini
gündeme getirir; bu sonuç tek başına gerçek metinlerde de benzer başarım
olacağının kanıtı değildir.

### Çalışmanın kısıtları

- Veri kümesi sentetiktir; doğal kullanıcı dili, yazım hataları, ironi, argo,
  bağlam ve farklı ürün deneyimleri yeterince temsil edilmeyebilir.
- Ana karşılaştırma sabit bir yaklaşık %80/%20 gruplu bölme ve
  `random_state=42` ile raporlanmıştır. Beş katlı gruplu çapraz doğrulama,
  tekrarları önleme etkisini kontrol etmek için ek olarak çalıştırılmış; model
  ayarında kullanılmamıştır.
- Hiperparametre araması ve bağımsız harici test kümesi kullanılmamıştır.
- Mevcut karşılaştırma `lemmatizasyon` metni ve tek bir varsayılan TF-IDF
  ayarıyla yapılmıştır; kök bulma, farklı n-gram aralıkları ve sözcük
  dağarcığı ayarları karşılaştırılmamıştır.
- Çok yüksek sentetik test puanları, olası veri üretim kalıpları veya
  yinelenen/benzer yorumlar bakımından bağımsız bir sızıntı denetiminin yerine
  geçmez.

### Gelecekte yapılabilecek çalışmalar

1. Gerçek kullanıcı yorumlarından oluşan, insan tarafından gözden geçirilmiş
   ve sınıf tanımları tutarlı bir veri kümesi hazırlamak.
2. Yinelenen ve çok benzer yorumları daha geniş ölçekte kontrol edip, aynı
   kaynaktan gelen yorumların eğitim ve teste dağılmasını önleyen kaynak/grup
   temelli bölme uygulamak.
3. Stratified group k-fold çapraz doğrulama ve bağımsız bir harici test kümesiyle
   sonuçların kararlılığını değerlendirmek.
4. Hiperparametre ayarı yaparken test kümesini son değerlendirmeye saklamak;
   ayarları yalnızca eğitim kümesi içindeki çapraz doğrulamayla seçmek.
5. Lemma/kök seçeneklerini, unigram-bigram TF-IDF ayarlarını ve sınıf
   ağırlıklarını aynı doğrulama düzeninde karşılaştırmak.
6. Karışıklık matrisi, sınıf bazlı precision/recall, ROC eğrisi ve eşik
   seçimini incelemek; özellikle yanlış olumlu ve yanlış olumsuz tahminleri
   örnekler üzerinden değerlendirmek.
7. Sonuçları farklı rastgele tohumlarla tekrarlayıp ortalama ve standart
   sapma ile raporlamak; olasılık kullanılan modellerde kalibrasyonu
   değerlendirmek.

### Sonuç bölümünde kullanılan Python araçları

Sonuç tablosu ve yorumları önceki aşamada `train_classifiers.py` ile üretilen
ölçümlerden alınmıştır. Model başarımını hesaplayan ve sonuçları sıralayan
önemli Python kodları/kütüphaneleri şunlardır:

- `sklearn.metrics.accuracy_score`, eğitim ve test doğruluklarını hesaplar.
- `sklearn.metrics.precision_recall_fscore_support`, macro precision, recall
  ve F1 değerlerini hesaplar.
- `sklearn.metrics.roc_auc_score`, gerçek olumlu/olumsuz etiketleri ile
  olumlu sınıf skorlarını karşılaştırarak ROC-AUC değerini hesaplar.
- `csv.DictReader` bölünmüş CSV dosyalarından etiketleri okur; `csv.DictWriter`
  ölçümleri ve yorum tahminlerini CSV olarak kaydeder.
- Python `sorted` fonksiyonu, `test_accuracy`, `f1_macro`, `roc_auc` ve
  `precision_macro` sırasına göre modelleri karşılaştırma dosyasında sıralar.
- Modeller için `fit`, `predict`, `predict_proba` ve
  `decision_function` metotları kullanılır; son ikisi ROC-AUC için olumlu
  sınıf skorlarını sağlar.
- Farklı çapraz doğrulama katlarından elde edilecek ölçümlerin ortalama ve
  standart sapmasını raporlamak için gelecekte `statistics.mean` ve
  `statistics.stdev` eklenebilir; mevcut çalışmada bu fonksiyonlar
  kullanılmamıştır.

Model sınıfları, TF-IDF matrislerinin yüklenmesi ve her metriğin hesaplanma
akışı önceki `train_classifiers.py` bölümünde açıklanmıştır.

### Çalıştırma ve çıktılar

Önce veri hazırlama adımını, sonra modelleri çalıştırın:

```bash
python prepare_dataset.py
python train_classifiers.py
```

Modelleme sonuçları varsayılan olarak `veri_seti/siniflandirma/` klasörüne
yazılır:

- `test_tahminleri.csv`: Her test yorumunun metni, gerçek etiketi ve dört
  algoritmanın tahmini.
- `model_metrikleri.csv`: Her algoritma için eğitim doğruluğu, test doğruluğu,
  macro precision, macro recall, macro F1 ve olumlu sınıfa göre ROC-AUC;
  test başarımına göre sıralanmış karşılaştırma.
- `naive_bayes.joblib`, `lojistik_regresyon.joblib`, `svm.joblib` ve
  `random_forest.joblib`: Eğitilmiş modeller.

Veri klasörü ve çıktı klasörü komut satırından değiştirilebilir:

```bash
python train_classifiers.py veri_seti sonuc_klasoru
```

## Yerel analiz paneli

Projedeki ön işleme açıklamaları, veri seti örnekleri, sınıflandırıcı
sonuçları ve metrik karşılaştırmaları Streamlit panelinde görüntülenebilir.
Panel ayrıca yeni bir Türkçe yorumu seçilen eğitilmiş modelle sınıflandırır.

Bağımlılıkları kurup paneli proje klasöründen başlatın:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini açın. Sol menüdeki bölümler:

- **Genel Bakış:** Yorum ve etiket sayıları, sınıf dağılımı ve işlem adımları.
- **Metin Ön İşleme:** README’deki yöntem açıklamaları ve ön işlenmiş yorum
  örnekleri.
- **Veri Setinin Hazırlanması:** Eğitim/test örnekleri, sınıf dağılımı ve
  TF-IDF matris bilgileri.
- **Makine Öğrenmesi ve Duygu Analizi:** Algoritma açıklamaları, test
  tahminleri ve yeni yorum için model seçmeli duygu tahmini.
- **Eğitim ve Model Başarımı:** Eğitim süreci ve modellerin kaydedilmiş
  başarım sonuçları; panelden modelleri yeniden eğitme.
- **Değerlendirme Metrikleri:** Eğitim/test ölçüm tablosu, metrik grafiği ve
  seçilen modelin karışıklık matrisi.
- **Sonuç ve Karşılaştırma:** En başarılı modeller, güçlü/zayıf yönler,
  araştırma yorumu, kısıtlar ve gelecek çalışma önerileri.

Panel açıklamaları bu README dosyasından, tablo ve grafikler ise proje CSV/NPZ
çıktılarından yüklenir. Gerekli çıktılar yoksa veri seti hazırlama ve model
eğitme düğmeleriyle oluşturulabilir. Yeni yorum tahmini için eğitim aşamasında
kaydedilmiş TF-IDF vektörleştiricisi ve model dosyaları kullanılır.
