"""Belediye hizmet binası yorumları için BERTopic konu analizi (Streamlit)."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

MODEL_ADI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Yalnızca işlev sözcükleri: "belediye", "bina", "hizmet", "yok", "değil", "bekleme" gibi anlam taşıyanlar bilinçli olarak yok.
TR_DURAK = sorted(set("""
ve veya ile ya yahut ama fakat ancak lakin oysa halbuki yalnız hem de da ki mi mı mu mü
bir bu şu o bunu şunu onu bunun şunun onun buna şuna ona bunda onda bundan ondan bunlar şunlar onlar
bunları onları bunlara onlara ben sen biz siz bana sana bize size beni seni bizi sizi benim senin bizim sizin
bence sence bize kendi kendim kendimiz kendisi
için gibi kadar göre rağmen dolayı ötürü üzere beri karşı doğru tarafından
çok daha en pek oldukça fazla az biraz hiç hep her tüm bütün bazı birçok bir_çok herkes kimse hiçbir
ise idi imiş iken ise olan olarak olarak oldu olduğu olduğunu olması olmuş olur olacak olabilir olsa olsun
ise ile var_mı mıydı miydi
ne neden niçin nasıl nerede nereye kim hangi kaç
sonra önce şimdi artık zaten hâlâ hala yine bile sadece yalnızca yani şey şeyi şeyler
aynı diye çünkü eğer şayet belki tabi tabii evet hayır
bugün dün yarın gün sabah öğle akşam
ın in un ün nın nin nun nün ı i u ü ya ye na ne
den dan ten tan de da te ta
lütfen sanki kez defa kere
""".replace("_", " ").split()))

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


def turkce_baslik(s: str) -> str:
    if not s:
        return s
    ilk = {"i": "İ", "ı": "I"}.get(s[0], s[0].upper())
    return ilk + s[1:]


def vektorlayici_olustur():
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(
        ngram_range=(1, 2), stop_words=TR_DURAK, min_df=2, max_df=0.8, lowercase=False,
        preprocessor=turkce_kucuk, token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    )


def temsili_yorumlar(emb: np.ndarray, konular: np.ndarray, n: int = 5) -> dict[int, list[int]]:
    """Her konu için merkeze (ortalama embedding) en yakın n gerçek yorumun indeksi."""
    norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sonuc = {}
    for k in sorted(set(konular.tolist())):
        idx = np.where(konular == k)[0]
        merkez = norm[idx].mean(axis=0)
        benzerlik = norm[idx] @ merkez
        sonuc[k] = idx[np.argsort(-benzerlik)[:n]].tolist()
    return sonuc


def anahtar_kelimeler(model, k: int, n: int = 10) -> list[str]:
    asp = model.get_topics(full=True).get("KeyBERT", {}).get(k)
    kel = [w for w, _ in (asp or []) if w]
    if not kel:
        kel = [w for w, _ in (model.get_topic(k) or []) if w]
    return kel[:n]


def konu_adi_oner(kelimeler: list[str], k: int) -> str:
    sozcukler = []
    for kel in kelimeler:
        for s in kel.split():
            if s not in sozcukler:
                sozcukler.append(s)
    return " / ".join(turkce_baslik(s) for s in sozcukler[:3]) or f"Konu {k}"


def konu_tablosu(model, konular: np.ndarray, emb: np.ndarray, n_toplam: int):
    temsil = temsili_yorumlar(emb, konular, 5)
    bilgi = model.get_topic_info()
    satirlar, ornek_idx = [], {}
    for _, r in bilgi.iterrows():
        k = int(r["Topic"])
        kel = anahtar_kelimeler(model, k)
        ctf = [w for w, _ in (model.get_topic(k) or []) if w][:10]
        sayi = int((konular == k).sum())
        satirlar.append({
            "Konu No": k,
            "Konu Adı": "Aykırı / Konusuz" if k == -1 else konu_adi_oner(kel, k),
            "Anahtar Kelimeler": ", ".join(kel),
            "c-TF-IDF Kelimeleri": ", ".join(ctf),
            "Yorum Sayısı": sayi,
            "Oran (%)": round(100 * sayi / n_toplam, 1),
        })
        ornek_idx[k] = temsil.get(k, [])
    return pd.DataFrame(satirlar), ornek_idx


def analiz_yap(metinler, hedef, min_boyut, n_komsu, seed, yeniden_ata, esik):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired
    from hdbscan import HDBSCAN
    from umap import UMAP

    emb = embedding_uret(tuple(metinler))
    temsil_modeli = {"KeyBERT": KeyBERTInspired(top_n_words=10, random_state=seed)}
    model = BERTopic(
        language="turkish", embedding_model=modeli_yukle(),
        umap_model=UMAP(n_neighbors=n_komsu, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_boyut, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=vektorlayici_olustur(),
        representation_model=temsil_modeli,
        nr_topics=None, top_n_words=10, calculate_probabilities=False, verbose=False,
    )
    konular, _ = model.fit_transform(metinler, embeddings=emb)
    konular = np.array(konular)
    dogal = len(set(konular.tolist()) - {-1})
    if dogal > hedef:
        # BERTopic, nr_topics değerine aykırı (-1) konuyu da dahil eder.
        model.reduce_topics(metinler, nr_topics=hedef + (1 if -1 in konular else 0))
        konular = np.array(model.topics_)

    onceki_df, onceki_idx = konu_tablosu(model, konular, emb, len(metinler))
    onceki_konular = konular.copy()
    sonuc = {"dogal": dogal, "onceki_df": onceki_df, "yeniden_atandi": False}

    if yeniden_ata and (konular == -1).any():
        yeni = model.reduce_outliers(metinler, konular.tolist(), strategy="embeddings",
                                     embeddings=emb, threshold=esik)
        model.update_topics(metinler, topics=yeni, vectorizer_model=vektorlayici_olustur(),
                            representation_model=temsil_modeli)
        konular = np.array(model.topics_)
        sonuc["yeniden_atandi"] = True

    konu_df, ornek_idx = konu_tablosu(model, konular, emb, len(metinler))
    sonuc.update({
        "model": model, "konular": konular, "onceki_konular": onceki_konular,
        "konu_df": konu_df, "ornek_idx": ornek_idx, "emb": emb,
        "ctfidf": {int(k): [(w, p) for w, p in (model.get_topic(int(k)) or []) if w][:8]
                   for k in konu_df["Konu No"] if k != -1},
    })
    return sonuc


def excel_uret(konu_df, yorum_df, ornek_df, ozet_df) -> bytes:
    cikti = io.BytesIO()
    with pd.ExcelWriter(cikti, engine="xlsxwriter") as w:
        sayfalar = (("Konu Özeti", konu_df), ("Aykırı Özeti", ozet_df),
                    ("Örnek Yorumlar", ornek_df), ("Yorumlar", yorum_df))
        for ad, df in sayfalar:
            df.to_excel(w, sheet_name=ad, index=False)
            ws = w.sheets[ad]
            for i, c in enumerate(df.columns):
                q = int(df[c].astype(str).str.len().quantile(0.9)) if len(df) else 10
                ws.set_column(i, i, min(80, max(len(str(c)), q) + 2))
    return cikti.getvalue()


def main():
    st.title("Belediye Hizmet Binası Yorumları - Konu Analizi")
    st.caption("BERTopic: çok dilli MiniLM + UMAP + HDBSCAN + c-TF-IDF + KeyBERTInspired")

    with st.sidebar:
        st.header("Ayarlar")
        hedef = st.number_input("Hedef konu sayısı", 2, 30, 10)
        min_boyut = st.slider("Minimum küme boyutu", 3, 50, 10)
        n_komsu = st.slider("UMAP komşu sayısı", 5, 50, 15)
        seed = st.number_input("Rastgelelik tohumu", 0, 9999, 42)
        st.divider()
        yeniden_ata = st.checkbox("Aykırı yorumları en yakın konulara yeniden ata", value=False,
                                  help="BERTopic reduce_outliers (embeddings). Kapalıyken varsayılan sonuçlar korunur.")
        esik = st.slider("Yeniden atama benzerlik eşiği", 0.0, 0.9, 0.3, 0.05, disabled=not yeniden_ata)

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
    st.dataframe(df.head(10), width="stretch")

    sutun = st.selectbox("Yorum sütunu", list(df.columns))
    if st.button("Analizi başlat", type="primary"):
        df = df[df[sutun].notna()].copy()
        df[sutun] = df[sutun].astype(str).str.strip()
        df = df[df[sutun].str.len() > 0].reset_index(drop=True)
        metinler = df[sutun].tolist()
        if len(metinler) < max(2 * min_boyut, 20):
            st.error("Analiz için yeterli yorum yok; daha fazla yorum veya daha küçük küme boyutu gerekir.")
            return
        with st.spinner("Konular çıkarılıyor..."):
            try:
                s = analiz_yap(metinler, int(hedef), min_boyut, n_komsu, int(seed), yeniden_ata, esik)
            except Exception as e:
                st.exception(e)
                return
            from umap import UMAP

            s["xy"] = UMAP(n_neighbors=n_komsu, n_components=2, min_dist=0.1, metric="cosine",
                           random_state=int(seed)).fit_transform(s["emb"])
        s.update({"metinler": metinler, "df": df, "hedef": int(hedef), "n_komsu": n_komsu})
        st.session_state["sonuc"] = s
        st.session_state["duzenleme_no"] = st.session_state.get("duzenleme_no", 0) + 1

    if "sonuc" in st.session_state:
        sonuclari_goster(st.session_state["sonuc"])


def sonuclari_goster(s):
    metinler, konular, hedef = s["metinler"], s["konular"], s["hedef"]
    model = s["model"]
    n = len(metinler)
    konu_df = s["konu_df"].copy()

    # Sonuç özeti ve hedef kontrolü
    gercek = int((konu_df["Konu No"] != -1).sum())
    if s["dogal"] > hedef:
        st.success(f"Doğal küme sayısı {s['dogal']} idi; hedef olan {hedef} konuya indirgendi.")
    else:
        st.warning(
            f"{hedef} ayrı anlamlı konu bulunamadı: veride yalnızca {gercek} doğal konu var. "
            "Yapay konu oluşturulmadı. Daha az sayıda konu bulunmasını beklemek gerekir; "
            "istenirse minimum küme boyutunu küçültmeyi deneyin."
        )

    # Aykırı yorumlar
    st.subheader("Aykırı yorumlar (-1)")
    once = int((s["onceki_konular"] == -1).sum())
    sonra = int((konular == -1).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Aykırı yorum sayısı", sonra, delta=(sonra - once) if s["yeniden_atandi"] else None, delta_color="inverse")
    c2.metric("Aykırı oranı", f"%{100 * sonra / n:.1f}")
    c3.metric("Toplam yorum", n)
    if s["yeniden_atandi"]:
        st.info(f"Yeniden atama öncesi: {once} (%{100 * once / n:.1f}) - sonrası: {sonra} (%{100 * sonra / n:.1f})")
        onceki = s["onceki_df"][["Konu No", "Yorum Sayısı"]].rename(columns={"Yorum Sayısı": "Önce"})
        kar = konu_df[["Konu No", "Yorum Sayısı"]].rename(columns={"Yorum Sayısı": "Sonra"}).merge(onceki, on="Konu No", how="outer").fillna(0)
        kar["Değişim"] = kar["Sonra"] - kar["Önce"]
        with st.expander("Konu bazında önce / sonra yorum sayıları"):
            st.dataframe(kar.astype(int), hide_index=True, width="stretch")
    elif sonra:
        st.caption("Yeniden atama için kenar çubuğundaki seçeneği işaretleyip analizi yeniden başlatın.")

    # Düzenlenebilir konu adları
    st.subheader("Konu adları")
    st.caption("Önerilen adları tablodan değiştirebilirsiniz; grafikler, örnekler ve Excel çıktısı güncellenir.")
    duzenle = konu_df[["Konu No", "Konu Adı"]].copy()
    duzenlenen = st.data_editor(
        duzenle, hide_index=True, width="stretch", disabled=["Konu No"],
        key=f"adlar_{st.session_state.get('duzenleme_no', 0)}",
    )
    adlar = dict(zip(duzenlenen["Konu No"], duzenlenen["Konu Adı"].fillna("").astype(str)))
    konu_df["Konu Adı"] = [adlar.get(k) or f"Konu {k}" for k in konu_df["Konu No"]]
    ad_harita = dict(zip(konu_df["Konu No"], konu_df["Konu Adı"]))

    st.subheader("Konu özeti")
    st.dataframe(konu_df, width="stretch", hide_index=True)

    grafik = konu_df.copy()
    grafik["Etiket"] = grafik["Konu No"].astype(str) + " - " + grafik["Konu Adı"]
    st.plotly_chart(
        px.bar(grafik.sort_values("Yorum Sayısı"), x="Yorum Sayısı", y="Etiket", orientation="h",
               title="Konulara göre yorum sayısı"),
        width="stretch",
    )

    puan = [{"Konu": f"{k} - {ad_harita[k]}", "Kelime": w, "c-TF-IDF": p}
            for k, lst in s["ctfidf"].items() for w, p in lst]
    if puan:
        st.plotly_chart(
            px.bar(pd.DataFrame(puan), x="c-TF-IDF", y="Kelime", color="Konu", facet_col="Konu",
                   facet_col_wrap=3, orientation="h", height=300 * ((gercek + 2) // 3),
                   title="Konu anahtar kelimeleri (c-TF-IDF)")
            .update_yaxes(matches=None, showticklabels=True).update_xaxes(matches=None),
            width="stretch",
        )

    gdf = pd.DataFrame({
        "x": s["xy"][:, 0], "y": s["xy"][:, 1],
        "Konu": [ad_harita[k] if k != -1 else "Aykırı" for k in konular],
        "Yorum": [m[:150] for m in metinler],
    })
    st.plotly_chart(
        px.scatter(gdf, x="x", y="y", color="Konu", hover_data=["Yorum"], title="Yorum haritası (UMAP 2B)"),
        width="stretch",
    )

    st.subheader("Konu anahtar kelimeleri ve en temsilî 5 yorum")
    ornek_satir = []
    for _, r in konu_df.iterrows():
        k = int(r["Konu No"])
        yorumlar = [metinler[i] for i in s["ornek_idx"].get(k, [])]
        with st.expander(f"{r['Konu Adı']} - {r['Yorum Sayısı']} yorum"):
            st.markdown(f"**Anahtar kelimeler (KeyBERT):** {r['Anahtar Kelimeler']}")
            st.markdown(f"**c-TF-IDF:** {r['c-TF-IDF Kelimeleri']}")
            for y in yorumlar:
                st.write("• " + y)
        for sira, y in enumerate(yorumlar, 1):
            ornek_satir.append({"Konu No": k, "Konu Adı": r["Konu Adı"], "Sıra": sira, "Örnek Yorum": y})
    ornek_df = pd.DataFrame(ornek_satir)

    yorum_df = s["df"].copy()
    yorum_df["Konu No"] = konular
    yorum_df["Konu Adı"] = [ad_harita[k] for k in konular]
    yorum_df["Önceki Konu No"] = s["onceki_konular"]
    ozet_df = pd.DataFrame({
        "Ölçü": ["Toplam yorum", "Aykırı (önce)", "Aykırı oranı önce (%)", "Aykırı (sonra)",
                 "Aykırı oranı sonra (%)", "Yeniden atama uygulandı", "Doğal konu sayısı", "Son konu sayısı"],
        "Değer": [n, once, round(100 * once / n, 1), sonra, round(100 * sonra / n, 1),
                  "Evet" if s["yeniden_atandi"] else "Hayır", s["dogal"], gercek],
    })

    st.download_button(
        "Sonuçları Excel olarak indir",
        data=excel_uret(konu_df, yorum_df, ornek_df, ozet_df),
        file_name="konu_analizi_sonuclari.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


main()
