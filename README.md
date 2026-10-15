# belediye-bertopic

Belediye hizmet binası yorumlarını BERTopic ile analiz eden Streamlit uygulaması.

- Excel/CSV yükleme ve yorum sütunu seçimi
- Çok dilli MiniLM embedding, UMAP, HDBSCAN, c-TF-IDF ve KeyBERTInspired
- Hedef 10 konu (doğal küme azsa yapay konu üretilmez), düzenlenebilir konu adları
- Aykırı yorum sayısı/oranı ve isteğe bağlı `reduce_outliers`
- Plotly grafikleri ve Excel çıktısı

## Kurulum (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

İlk çalıştırmada embedding modeli Hugging Face'ten indirilir.
