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
