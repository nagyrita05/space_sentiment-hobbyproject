#!/usr/bin/env python
# coding: utf-8

# ## A `utils.py` szerepe:
# 
# A `utils.py` a közös, ismétlődő kódokat tartalmazza:
# 
# - országlisták: `countries` (angol), `non_english_countries` (nem angol),
#   és az `iso2_to_iso3` kódmapping (a térképekhez kell az ISO3);
# - letöltés: `fetch_gnews_for_countries(topic_query, country_list, ...)`;
# - érzelemelemzés:
#   - EN: `score_sentiment_vader` (VADER modellel),
#   - NON-EN: `load_xlmr_model` + `predict_sentiment_batch_xlmr` (XLM-R);
# - kényelmi helperek: `add_iso3(df)` és `get_countries_df()`;
# - aggregálás + 4 térkép egy hívással: `plot_four_maps_for_topic(all_news, topic_name)`;
# - példacímek listázása: `top_examples_all(all_news, label="positive"/"negative")`.
# 
# Előny: a 4 altéma notebookja egységes, rövid, jól olvasható. Újrafuttatáskor csak a 
# `TOPIC_NAME` és a `topic_query` változik.
# 

# In[2]:


# utils.py

# Közös segédfüggvények + országlisták + ISO mapping

import os
import pandas as pd

# ---------- Országlisták és ISO kód mapping ----------

# Angol nyelvű országok: (country_name, iso2, hl, gl, ceid)
countries = [
    ("United States","US","en-US","US","US:en"),
    ("United Kingdom","GB","en-GB","GB","GB:en"),
    ("Canada","CA","en-CA","CA","CA:en"),
    ("Australia","AU","en-AU","AU","AU:en"),
    ("India","IN","en-IN","IN","IN:en"),
    ("Ireland","IE","en-IE","IE","IE:en"),
    ("South Africa","ZA","en-ZA","ZA","ZA:en"),
    ("Singapore","SG","en-SG","SG","SG:en"),
    ("New Zealand","NZ","en-NZ","NZ","NZ:en"),
    ("Philippines","PH","en-PH","PH","PH:en"),
    ("Nigeria","NG","en-NG","NG","NG:en"),
    ("Kenya","KE","en-KE","KE","KE:en"),
    ("Hong Kong","HK","en-HK","HK","HK:en"),
    ("Malaysia","MY","en-MY","MY","MY:en"),
    ("United Arab Emirates","AE","en-AE","AE","AE:en"),
    ("Pakistan","PK","en-PK","PK","PK:en"),
    ("Bangladesh","BD","en-BD","BD","BD:en"),
    ("Ghana","GH","en-GH","GH","GH:en"),
    ("Tanzania","TZ","en-TZ","TZ","TZ:en"),
    ("Uganda","UG","en-UG","UG","UG:en"),
    ("Jamaica","JM","en-JM","JM","JM:en"),
]

# Nem angol országok (példák az altémához): (country_name, iso2, hl, gl, ceid)
non_english_countries = [
    ("China",  "CN", "zh-CN", "CN", "CN:zh-Hans"),
    ("Russia", "RU", "ru-RU", "RU", "RU:ru"),
    ("Germany","DE", "de-DE", "DE", "DE:de"),
    ("France", "FR", "fr-FR", "FR", "FR:fr"),
    ("Spain",  "ES", "es-ES", "ES", "ES:es"),
]

# ISO2 -> ISO3
iso2_to_iso3 = {
    "US":"USA","GB":"GBR","CA":"CAN","AU":"AUS","IN":"IND","IE":"IRL","ZA":"ZAF",
    "SG":"SGP","NZ":"NZL","PH":"PHL","NG":"NGA","KE":"KEN","HK":"HKG",
    "MY":"MYS","AE":"ARE","PK":"PAK","BD":"BGD","GH":"GHA","TZ":"TZA","UG":"UGA","JM":"JAM",
    "CN":"CHN","RU":"RUS","DE":"DEU","FR":"FRA","ES":"ESP"
}

# ---------- Kényelmi helperek: országlista DF és ISO3 hozzáadás ----------

def get_countries_df():
    """Országlista DataFrame-ként + iso3 oszlop."""
    cols = ["country_name","iso2","hl","gl","ceid"]
    df = pd.DataFrame(countries, columns=cols)
    df["iso3"] = df["iso2"].map(iso2_to_iso3)
    return df

def add_iso3(df: pd.DataFrame, iso2_col="iso2"):
    """Hozzáadja/kitölti az iso3 oszlopot a megadott iso2 oszlop alapján."""
    out = df.copy()
    out["iso3"] = out[iso2_col].map(iso2_to_iso3)
    return out

# ---------- RSS letöltés Google Newsból ----------

def fetch_gnews_for_country(topic_query: str, hl: str, gl: str, ceid: str, limit=120):
    """Egy ország (hl/gl/ceid) RSS-e egy queryre. Vissza: DataFrame(title, link, published, source)."""
    import feedparser
    from urllib.parse import quote_plus
    url = f"https://news.google.com/rss/search?q={quote_plus(topic_query)}&hl={hl}&gl={gl}&ceid={ceid}"
    feed = feedparser.parse(url)
    rows = []
    for e in feed.entries[:limit]:
        rows.append({
            "title": e.get("title",""),
            "link": e.get("link",""),
            "published": e.get("published",""),
            "source": (getattr(e, "source", {}).get("title")
                       if hasattr(e, "source") and isinstance(getattr(e,"source"), dict)
                       else "")
        })
    return pd.DataFrame(rows)

def fetch_gnews_for_countries(topic_query: str, country_list, limit=120, source_lang="EN"):
    """
    Több ország lekérése és összefűzése.
    country_list: list of tuples (country_name, iso2, hl, gl, ceid)
    """
    frames = []
    for name, iso2, hl, gl, ceid in country_list:
        df = fetch_gnews_for_country(topic_query, hl, gl, ceid, limit=limit)
        df["country"] = name
        df["iso2"] = iso2
        df["source_lang"] = source_lang
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(subset=["title","link"]).reset_index(drop=True)
    return out

# ---------- VADER az angol headline-okra ----------

def score_sentiment_vader(text: str):
    """VADER compound score (-1..+1). Notebookban töltsd le a lexicont."""
    if not isinstance(text, str): 
        return 0.0
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        return sia.polarity_scores(text)["compound"]
    except Exception:
        return 0.0

# ---------- Többnyelvű (XLM-R) modell a nem angol címekhez ----------

def load_xlmr_model(model_name="cardiffnlp/twitter-xlm-roberta-base-sentiment"):
    """
    Betölti a tokenizer+modellt. Lassabb, mert use_fast=False (nincs tokenizers build).
    CUDA ha elérhető.
    """
    import torch
    from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=False)
    mdl = XLMRobertaForSequenceClassification.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl, device

def predict_sentiment_batch_xlmr(texts, tok, mdl, device, batch_size=24, max_length=128):
    """Címkék: negative / neutral / positive. Vissza: lista címkékkel."""
    import torch
    labels = ["negative","neutral","positive"]
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = [t if isinstance(t,str) and t.strip() else "" for t in texts[i:i+batch_size]]
        enc = tok(batch, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = mdl(**enc).logits
            idx = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        preds.extend([labels[j] for j in idx])
    return preds

# ---------- Ország-szintű aggregálások + 4 térkép + CSV ----------

def make_country_aggregates(all_news: pd.DataFrame):
    """
    Elvárás: all_news tartalmazza a 'country', 'sentiment' (számszerű), 'sentiment_label' oszlopokat.
    Vissza: dict {median, mean, pos_ratio, neg_ratio} DataFrame-ekkel.
    """
    med = (all_news.groupby("country", as_index=False)["sentiment"]
           .median().rename(columns={"sentiment":"median_sentiment"}))
    mean = (all_news.groupby("country", as_index=False)["sentiment"]
            .mean().rename(columns={"sentiment":"mean_sentiment"}))
    pos = (all_news.groupby("country")["sentiment_label"]
           .apply(lambda s: (s=="positive").mean())
           .reset_index().rename(columns={"sentiment_label":"positive_ratio"}))
    neg = (all_news.groupby("country")["sentiment_label"]
           .apply(lambda s: (s=="negative").mean())
           .reset_index().rename(columns={"sentiment_label":"negative_ratio"}))
    return {"median": med, "mean": mean, "pos_ratio": pos, "neg_ratio": neg}

def _maybe_add_iso3(df, all_news):
    if "iso3" in all_news.columns:
        return df.merge(all_news[["country","iso3"]].drop_duplicates(), on="country", how="left")
    return df

def plot_four_maps_for_topic(all_news: pd.DataFrame, topic_name: str, use_iso3=True):
    """
    Négy plotly choropleth: medián, átlag, pozitív és negatív arány (EN+NON_EN összevonva).
    A CSV-ket az outputs/<topic>/csv mappába menti.
    """
    import plotly.express as px
    os.makedirs(f"outputs/{topic_name}/csv", exist_ok=True)

    aggs = make_country_aggregates(all_news)
    med = _maybe_add_iso3(aggs["median"], all_news)
    avg = _maybe_add_iso3(aggs["mean"], all_news)
    pos = _maybe_add_iso3(aggs["pos_ratio"], all_news)
    neg = _maybe_add_iso3(aggs["neg_ratio"], all_news)

    def _plot(df, value_col, title):
        if use_iso3 and "iso3" in df.columns:
            fig = px.choropleth(df, locations="iso3", color=value_col,
                                hover_name="country",
                                color_continuous_scale="RdYlGn",
                                range_color=(df[value_col].min(), df[value_col].max()),
                                title=title)
        else:
            fig = px.choropleth(df, locations="country", locationmode="country names",
                                color=value_col,
                                color_continuous_scale="RdYlGn",
                                range_color=(df[value_col].min(), df[value_col].max()),
                                title=title)
        fig.show()

    _plot(med, "median_sentiment", f"{topic_name} – Median sentiment (EN + NON_EN)")
    _plot(avg, "mean_sentiment",   f"{topic_name} – Mean sentiment (EN + NON_EN)")
    _plot(pos, "positive_ratio",   f"{topic_name} – Positive share (EN + NON_EN)")
    _plot(neg, "negative_ratio",   f"{topic_name} – Negative share (EN + NON_EN)")

    safe = topic_name
    med.to_csv(f"outputs/{safe}/csv/{safe}_median_by_country.csv", index=False)
    avg.to_csv(f"outputs/{safe}/csv/{safe}_mean_by_country.csv", index=False)
    pos.to_csv(f"outputs/{safe}/csv/{safe}_positive_ratio_by_country.csv", index=False)
    neg.to_csv(f"outputs/{safe}/csv/{safe}_negative_ratio_by_country.csv", index=False)

# ---------- TOP példacímek ----------

def top_examples_all(all_news, label="positive", k=10, per_country=False):
    df = all_news[all_news["sentiment_label"].eq(label)].copy()
    if "sentiment" in df.columns:
        df = df.sort_values("sentiment", ascending=(label=="negative"))
    keep = [c for c in ["country","source_lang","title","sentiment","sentiment_label","source","link"] if c in df.columns]
    df = df[keep]
    if per_country:
        return df.groupby("country", group_keys=False).head(k).reset_index(drop=True)
    return df.head(k).reset_index(drop=True)


# In[ ]:




