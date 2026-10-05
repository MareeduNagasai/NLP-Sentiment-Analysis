# 🧠 AI Sentiment Analyzer

**Classify mobile phone reviews as Positive, Neutral or Negative using NLP, TF-IDF and Machine Learning, served through an interactive Streamlit app.**

### 🚀 [Live Demo on Streamlit Community Cloud](https://nlp-sentiment-analysis-sie2nvgz36s5tudmk8f5hx.streamlit.app/)

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Deployment](#-deployment)
- [Future Improvements](#-future-improvements)
- [Contact](#-contact)

---

## 📌 About the Project

Online reviews are a goldmine of customer opinion, but reading thousands of them by hand is impractical.

This project builds an **end-to-end sentiment analysis pipeline** on mobile phone reviews:

1. Data exploration and text preprocessing
2. Feature engineering
3. Model comparison
4. A deployed web app where anyone can type a review and instantly see the predicted sentiment with a confidence breakdown

Sentiment labels are derived from star ratings:

| Rating | Sentiment   |
| :----: | :---------- |
| 4 – 5  | 😊 Positive |
| 3      | 😐 Neutral  |
| 1 – 2  | 😞 Negative |

---

## ✨ Features

- **Sentiment-aware preprocessing**: keeps negation words (`not`, `never`, `no`, …) so "not good" is not read as "good"
- **Detailed EDA**: rating distribution, word clouds, review length analysis, correlation heatmap
- **VADER comparison**: rating-based labels vs. lexicon-based VADER sentiment
- **TF-IDF with bigrams** (1–2 n-grams, sublinear TF)
- **Multiple models compared**, including a transformer (DistilBERT) baseline
- **Interactive Streamlit UI** with a colour-coded result, per-class percentages and progress bars

---

## 📊 Dataset

| Property        | Value                                    |
| --------------- | ---------------------------------------- |
| Total reviews   | **1,440**                                |
| Columns         | `title`, `rating`, `body`                |
| Missing values  | 0                                        |
| Duplicate rows  | 0                                        |
| Rating scale    | 1 – 5                                    |
| Languages       | Mostly English, with a few Hindi reviews |

The `title` and `body` columns are combined into a single `review_text` field for modelling.

**Class distribution (after mapping ratings to sentiment):**

| Sentiment   | Reviews | Share |
| ----------- | ------: | ----: |
| 😊 Positive |     729 | 50.6% |
| 😞 Negative |     512 | 35.6% |
| 😐 Neutral  |     199 | 13.8% |

---

## 🔍 Exploratory Data Analysis

### Rating Sentiment Distribution

The dataset is **imbalanced**: Positive reviews dominate, while Neutral is the minority class. This is why **F1-score** (not just accuracy) is used to compare models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/888109ab-e99c-4995-b038-3c9cd86f0da1" alt="Rating sentiment distribution" />
</p>

### Overall Word Cloud

Frequent terms such as *phone*, *camera*, *battery*, *performance* and *price* show which product aspects customers talk about most.

<p align="center">
  <img src="https://github.com/user-attachments/assets/09175901-8d65-4284-a0fb-5828f444b9e4" alt="Overall word cloud" />
</p>

### Rating-based vs. VADER Sentiment

| Rating-based ↓ / VADER → | Negative | Neutral | Positive |
| ------------------------ | -------: | ------: | -------: |
| **Negative**             |      361 |      21 |      130 |
| **Neutral**              |       81 |       6 |      112 |
| **Positive**             |       45 |      17 |      667 |

VADER agrees well on clearly positive and negative reviews but struggles with mixed or neutral ones. This motivated training a supervised model on the rating-based labels.

<p align="center">
  <img src="https://github.com/user-attachments/assets/897030dc-7744-4ff9-9ad9-bd8b9408c0c8" alt="Rating-based vs VADER sentiment" />
</p>

---

## 🛠️ Methodology

```
Raw reviews ─► Combine title + body ─► Rating → Sentiment labels
      │
      ▼
Text preprocessing ─► Feature engineering ─► TF-IDF ─► Model training ─► Evaluation ─► Streamlit app
```

### 1. Text Preprocessing

1. Lowercasing
2. Remove URLs, @mentions, HTML tags and numbers
3. Handle hashtags and remove punctuation
4. Normalise whitespace
5. Tokenisation (NLTK)
6. Stop-word removal, **keeping negation words**
7. Lemmatisation (WordNet)
8. Remove single-character tokens

### 2. Feature Engineering

- Review length, word count, unique word count, average word length, sentence count
- Exclamation, question and hashtag counts
- VADER `neg`, `neu`, `pos` and `compound` scores

### 3. Vectorisation

```python
TfidfVectorizer(max_features=3000, min_df=2, max_df=0.95,
                ngram_range=(1, 2), sublinear_tf=True)
```

### 4. Models Compared

| Feature set                          | Models                                   |
| ------------------------------------ | ---------------------------------------- |
| TF-IDF                               | Logistic Regression, Naive Bayes, Linear SVM |
| TF-IDF + engineered features         | Logistic Regression, Linear SVM          |
| Word2Vec                             | Logistic Regression, Linear SVM          |
| Transformer baseline                 | DistilBERT                               |

---

## 🏆 Model Performance

> **Selected model: Linear SVM (TF-IDF)**

Linear SVM was chosen as the final model based on **F1-score and the other evaluation metrics**.

**Why not BERT?** DistilBERT was fine-tuned for only 3 epochs. Training longer was not pursued because its **high computational cost** (training time, memory and inference latency) makes it a poor fit for a lightweight, freely hosted app. Linear SVM delivers competitive performance at a tiny fraction of the cost.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd1989a8-953e-497f-bd83-e4b50d295df9" alt="Model performance comparison" />
</p>

---

## 🧰 Tech Stack

| Category         | Tools                                           |
| ---------------- | ----------------------------------------------- |
| Language         | Python                                          |
| Data & Analysis  | Pandas, NumPy                                   |
| Visualisation    | Matplotlib, Seaborn, WordCloud                  |
| NLP              | NLTK (tokenisation, stop-words, WordNet, VADER) |
| Machine Learning | scikit-learn (TF-IDF, classifiers)              |
| Deep Learning    | Word2Vec embeddings, DistilBERT (baseline)      |
| Web App          | Streamlit                                       |
| Model Saving     | Joblib                                          |

---

## 📁 Project Structure

```
NLP-Sentiment-Analysis/
├── images/
│   ├── rating_sentiment_distribution.png
│   └── wordcloud_overall.png
├── app.py                             # Streamlit web app
├── Sentiment_Analysis_Project.ipynb   # EDA, preprocessing, feature engineering & modelling
├── svm_model.pkl                      # Trained classifier
├── tfidf_vectorizer.pkl               # Fitted TF-IDF vectorizer
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MareeduNagasai/NLP-Sentiment-Analysis.git
cd NLP-Sentiment-Analysis

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the App Locally

```bash
streamlit run app.py
```

---

## 💡 Usage

1. Type or paste a review into the text box.
2. Click **Analyze Sentiment**.
3. View the predicted sentiment, the percentage breakdown across Negative / Neutral / Positive, and the progress bars.

**Examples**

| Input                                                | Output      |
| ---------------------------------------------------- | ----------- |
| `Battery life is amazing and the camera is superb!`  | 😊 POSITIVE |
| `Worst phone ever. It hangs and the display is poor.`| 😞 NEGATIVE |
| `It is okay. not good, not bad`                      | 😐 NEUTRAL  |

---

## ☁️ Deployment

The app is hosted on **Streamlit Community Cloud**:

1. Push the repository to GitHub (including `app.py`, `requirements.txt` and both `.pkl` files).
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Select the repository, set `app.py` as the main file, and click **Deploy**.

---

## 🔮 Future Improvements

- [ ] Fine-tune a lightweight transformer (e.g. DistilBERT) to close the gap with BERT at lower cost
- [ ] Handle class imbalance (class weights / SMOTE) to improve Neutral recall
- [ ] Add multilingual support (Hindi reviews are present in the data)
- [ ] Aspect-based sentiment (camera, battery, display, price)

---

## 📬 Contact

**Mareedu Naga Sai**

- 📧 Email: nagasaimareedu45@gmail.com
- 💼 LinkedIn: [linkedin.com/in/mareedu-naga-sai-981019378](https://linkedin.com/in/mareedu-naga-sai-981019378)
- 🐙 GitHub: [MareeduNagasai](https://github.com/MareeduNagasai)
- 🔗 Project: [NLP-Sentiment-Analysis](https://github.com/MareeduNagasai/NLP-Sentiment-Analysis)
