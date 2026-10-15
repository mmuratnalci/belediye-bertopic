"""Belediye hizmet binası yorumları için BERTopic konu analizi (Streamlit)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_ADI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TR_DURAK = [
    "ve", "veya", "ile", "bir", "bu", "şu", "o", "da", "de", "ki", "mi", "mı", "mu", "mü",
    "için", "çok", "daha", "en", "gibi", "ama", "fakat", "ancak", "ya", "ne", "var", "yok",
    "olan", "olarak", "oldu", "olduğu", "olması", "değil", "her", "tüm", "bütün", "kadar",
    "sonra", "önce", "ben", "biz", "siz", "onlar", "bana", "bize", "ise", "hem", "çünkü",
    "bile", "sadece", "yani", "şey", "bunu", "bunlar", "daha", "aynı", "diye", "den", "dan",
    "ın", "in", "un", "ün", "nın", "nin", "nun", "nün", "ı", "i", "u", "ü", "ya", "ye",
    "belediye", "belediyesi", "bina", "binası", "hizmet", "hizmetleri",
]

st.set_page_config(page_title="Belediye Yorum Konu Analizi", layout="wide")


@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def modeli_yukle():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_ADI)


@st.cache_data(show_spinner="Yorumlar vektörleştiriliyor...")
def embedding_uret(metinler: tuple[str, ...]) -> np.ndarray:
    return modeli_yukle().encode(
        list(metinler), show_progress_bar=False, convert_to_numpy=True, batch_size=64
    )


def dosya_oku(dosya) -> dict[str, pd.DataFrame]:
    ad = dosya.name.lower()
    if ad.endswith(".csv"):
        ham = dosya.getvalue()
        for kodlama in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
            try:
                return {"CSV": pd.read_csv(io.BytesIO(ham), sep=None, engine="python", encoding=kodlama)}
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV kodlaması çözülemedi.")
    return pd.read_excel(dosya, sheet_name=None)


def turkce_kucuk(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def analiz_yap(metinler: list[str], hedef: int, min_boyut: int, n_komsu: int, seed: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    emb = embedding_uret(tuple(metinler))
    umap_model = UMAP(
        n_neighbors=n_komsu, n_components=5, min_dist=0.0, metric="cosine", random_state=seed
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_boyut, metric="euclidean", cluster_selection_method="eom",
        prediction_data=True,
    )
    vektorlayici = CountVectorizer(
        ngram_range=(1, 2), stop_words=TR_DURAK, min_df=2, max_df=0.8, lowercase=False,
        preprocessor=turkce_kucuk, token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    )
    model = BERTopic(
        language="turkish", embedding_model=None, umap_model=umap_model, hdbscan_model=hdbscan_model,
        vectorizer_model=vektorlayici, nr_topics=None, top_n_words=10,
        calculate_probabilities=False, verbose=False,
    )
    konular, _ = model.fit_transform(metinler, embeddings=emb)
    dogal = len({k for k in konular if k != -1})
    # Doğal küme sayısı hedeften azsa yapay konu üretilmez.
    if dogal > hedef:
        # BERTopic, nr_topics değerine aykırı (-1) konuyu da dahil eder.
        model.reduce_topics(metinler, nr_topics=hedef + (1 if -1 in konular else 0))
        konular = model.topics_
    return model, np.array(konular), emb, dogal


def sonuc_tablolari(model, metinler, konular, df_kaynak):
    bilgi = model.get_topic_info()
    satirlar = []
    for _, r in bilgi.iterrows():
        k = int(r["Topic"])
        kelimeler = [w for w, _ in (model.get_topic(k) or []) if w]
        satirlar.append({
            "Konu No": k,
            "Konu Adı": "Aykırı / Konusuz" if k == -1 else f"Konu {k}",
            "Anahtar Kelimeler": ", ".join(kelimeler[:10]),
            "Yorum Sayısı": int(r["Count"]),
            "Oran (%)": round(100 * int(r["Count"]) / len(metinler), 1),
        })
    konu_df = pd.DataFrame(satirlar)
    yorum_df = df_kaynak.copy()
    yorum_df["Konu No"] = konular
    return konu_df, yorum_df


def ornekler(model, metinler, konular, k, n=3):
    idx = np.where(konular == k)[0]
    try:
        if k != -1:
            temsil = model.get_representative_docs(k)
            if temsil:
                return temsil[:n]
    except Exception:
        pass
    return [metinler[i] for i in idx[:n]]


def excel_uret(konu_df, yorum_df, ornek_df) -> bytes:
    cikti = io.BytesIO()
    with pd.ExcelWriter(cikti, engine="xlsxwriter") as w:
        konu_df.to_excel(w, sheet_name="Konu Özeti", index=False)
        ornek_df.to_excel(w, sheet_name="Örnek Yorumlar", index=False)
        yorum_df.to_excel(w, sheet_name="Yorumlar", index=False)
        for ad, df in (("Konu Özeti", konu_df), ("Örnek Yorumlar", ornek_df), ("Yorumlar", yorum_df)):
            ws = w.sheets[ad]
            for i, c in enumerate(df.columns):
                genislik = min(80, max(len(str(c)), int(df[c].astype(str).str.len().quantile(0.9)) if len(df) else 10) + 2)
                ws.set_column(i, i, genislik)
    return cikti.getvalue()


def main():
    st.title("Belediye Hizmet Binası Yorumları - Konu Analizi")
    st.caption("BERTopic: çok dilli MiniLM + UMAP + HDBSCAN + c-TF-IDF")

    with st.sidebar:
        st.header("Ayarlar")
        hedef = st.number_input("Hedef konu sayısı", 2, 30, 10)
        min_boyut = st.slider("Minimum küme boyutu", 3, 50, 10)
        n_komsu = st.slider("UMAP komşu sayısı", 5, 50, 15)
        seed = st.number_input("Rastgelelik tohumu", 0, 9999, 42)

    dosya = st.file_uploader("Excel veya CSV dosyası yükleyin", type=["xlsx", "xls", "csv"])
    if not dosya:
        st.info("Başlamak için yorumları içeren bir dosya yükleyin.")
        return

    try:
        sayfalar = dosya_oku(dosya)
    except Exception as e:
        st.error(f"Dosya okunamadı: {e}")
        return

    sayfa = list(sayfalar)[0]
    if len(sayfalar) > 1:
        sayfa = st.selectbox("Sayfa", list(sayfalar))
    df = sayfalar[sayfa].dropna(how="all").reset_index(drop=True)
    st.write(f"{len(df)} satır, {len(df.columns)} sütun")
    st.dataframe(df.head(10), use_container_width=True)

    sutun = st.selectbox("Yorum sütunu", list(df.columns))
    if not st.button("Analizi başlat", type="primary"):
        return

    df = df[df[sutun].notna()].copy()
    df[sutun] = df[sutun].astype(str).str.strip()
    df = df[df[sutun].str.len() > 0].reset_index(drop=True)
    metinler = df[sutun].tolist()
    if len(metinler) < max(2 * min_boyut, 20):
        st.error("Analiz için yeterli yorum yok; daha fazla yorum veya daha küçük küme boyutu gerekir.")
        return

    with st.spinner("Konular çıkarılıyor..."):
        try:
            model, konular, emb, dogal = analiz_yap(metinler, int(hedef), min_boyut, n_komsu, int(seed))
        except Exception as e:
            st.exception(e)
            return

    konu_df, yorum_df = sonuc_tablolari(model, metinler, konular, df)
    gercek = int((konu_df["Konu No"] != -1).sum())
    msg = f"Doğal küme sayısı: {dogal}. Sonuçta {gercek} konu var."
    if dogal > hedef:
        st.success(msg + f" Hedefe ({int(hedef)}) indirgendi.")
    else:
        st.warning(msg + " Doğal küme sayısı hedefi aşmadığından yapay konu oluşturulmadı.")

    st.subheader("Konu özeti")
    st.dataframe(konu_df, use_container_width=True, hide_index=True)

    grafik = konu_df.copy()
    grafik["Etiket"] = grafik["Konu Adı"] + ": " + grafik["Anahtar Kelimeler"].str.split(", ").str[:3].str.join(", ")
    st.plotly_chart(
        px.bar(grafik.sort_values("Yorum Sayısı"), x="Yorum Sayısı", y="Etiket", orientation="h",
               title="Konulara göre yorum sayısı"),
        use_container_width=True,
    )

    # Anahtar kelime puanları
    puan = []
    for k in konu_df["Konu No"]:
        if k == -1:
            continue
        for kelime, p in [x for x in model.get_topic(k) if x[0]][:8]:
            puan.append({"Konu": f"Konu {k}", "Kelime": kelime, "c-TF-IDF": p})
    if puan:
        pdf = pd.DataFrame(puan)
        st.plotly_chart(
            px.bar(pdf, x="c-TF-IDF", y="Kelime", color="Konu", facet_col="Konu", facet_col_wrap=3,
                   orientation="h", height=300 * ((gercek + 2) // 3), title="Konu anahtar kelimeleri (c-TF-IDF)")
            .update_yaxes(matches=None, showticklabels=True).update_xaxes(matches=None),
            use_container_width=True,
        )

    # 2B UMAP görselleştirme
    from umap import UMAP

    xy = UMAP(n_neighbors=n_komsu, n_components=2, min_dist=0.1, metric="cosine",
              random_state=int(seed)).fit_transform(emb)
    gdf = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "Konu": [f"Konu {k}" if k != -1 else "Aykırı" for k in konular],
                        "Yorum": [m[:150] for m in metinler]})
    st.plotly_chart(
        px.scatter(gdf, x="x", y="y", color="Konu", hover_data=["Yorum"], title="Yorum haritası (UMAP 2B)"),
        use_container_width=True,
    )

    st.subheader("Örnek yorumlar")
    ornek_satir = []
    for _, r in konu_df.iterrows():
        k = int(r["Konu No"])
        orn = ornekler(model, metinler, konular, k)
        with st.expander(f"{r['Konu Adı']} - {r['Yorum Sayısı']} yorum - {r['Anahtar Kelimeler']}"):
            for o in orn:
                st.write("• " + o)
        for o in orn:
            ornek_satir.append({"Konu No": k, "Konu Adı": r["Konu Adı"], "Örnek Yorum": o})
    ornek_df = pd.DataFrame(ornek_satir)

    st.download_button(
        "Sonuçları Excel olarak indir",
        data=excel_uret(konu_df, yorum_df, ornek_df),
        file_name="konu_analizi_sonuclari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


main()
