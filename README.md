# 🛰️ Space Media Sentiment Analysis – Hobby Project

This project analyzes **space-related news articles** in four subtopics using:
- 🔍 **Sentiment classification in Python** (automatic, keyword-based)
- 📊 **Interactive dashboards in Power BI**
- 🧠 A focused **manual vs automatic comparison** for one subtopic using ChatGPT

---

## 📚 Subtopics Analyzed

| Subtopic          | Description |
|-------------------|-------------|
| `moon_mars`       | News related to Moon and Mars missions and scientific goals |
| `tech_launch`     | Technical developments and satellite/rocket launches |
| `success_failure` | Headlines reflecting success, failure, or risk in space projects |
| `policy_finance`  | Space policy, funding, contracts, partnerships |

---

## 🧪 Method Overview

- **Automatic sentiment scoring** of headlines (Python, keyword-based)
- **Descriptive statistics and visuals** per subtopic in Jupyter and Power BI
- **Country-level aggregation** using ISO codes and sentiment labels

---

## 🤖 vs 🧠 Deep Dive: Manual vs Automatic Sentiment

A special manual analysis (with ChatGPT) was conducted for the `policy_finance` subtopic:
- 30 randomly sampled headlines were rated manually (positive / neutral / negative)
- Compared against automated sentiment classification
- Country-level **choropleth maps** were generated to visualize differences

➡️ Key finding: manual evaluation detects social/emotional context better (e.g., women in space), while automated is faster and more scalable.

---

## 📁 Project Structure

```plaintext
📂 notebooks/
   └── sentiment_analysis.ipynb
   └── data_cleaning.ipynb

📂 powerbi/
   └── dashboard.pbix

📂 visuals/
   └── policy_finance_sentiment_map.png

📄 presentation_en.pptx      # English summary slides
📄 policy_finance_sentiment_analysis.md  # Detailed markdown analysis
📄 README.md                 # This file
```

---

## 🧾 Notes

- Jupyter notebooks remain in mixed language (EN/HU) – not fully translated
- The manual-vs-automated comparison is only included for `policy_finance`
- This is a hobby project created for learning and experimentation

---

## 💡 Author

Rita (GitHub: `nagyrita05`) | 2025
