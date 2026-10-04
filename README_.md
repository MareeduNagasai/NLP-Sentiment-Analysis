#  AI Sentiment Analyzer

**Classify customer reviews as Positive, Neutral or Negative using NLP, TF-IDF and Machine Learning, served through an interactive Streamlit app.**

###  [Live Demo on Streamlit Community Cloud](#)
---

##  Table of Contents

- [About the Project](#about-the-project)
- [Live Demo](#live-demo)
- [Features](#features)
- [Dataset](#dataset)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Deployment](#deployment-on-streamlit-community-cloud)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

##  About the Project

Online reviews are a goldmine of customer opinion, but reading thousands of them by hand is impractical. This project builds an **end-to-end sentiment analysis pipeline** on mobile phone reviews: from data exploration and text preprocessing, to feature engineering, model comparison, and a deployed web app where anyone can type a review and instantly see its predicted sentiment along with a confidence breakdown.

Sentiment labels are derived from star ratings:

| Rating | Sentiment |
| :----: | :-------- |
| 4 – 5  |  Positive |
| 3      |  Neutral |
| 1 – 2  |  Negative |

##  Live Demo

 **[Open the app](#)**
##  Features

-  **Sentiment-aware preprocessing**: keeps negation words (`not`, `never`, `no`, …) so "not good" is not read as "good"
-  **Detailed EDA**: rating distribution, word clouds, review length analysis, correlation heatmap
-  **VADER comparison**: rating-based labels vs. lexicon-based VADER sentiment
-  **TF-IDF with bigrams** (`1–2` n-grams, sublinear TF)
-  **Multiple models compared**, including a transformer (BERT) baseline
- ️ **Interactive Streamlit UI** with colour-coded result, per-class percentages and progress bars

##  Dataset

| Property | Value |
| -------- | ----- |
| Total reviews | **1,440** |
| Columns | `title`, `rating`, `body` |
| Missing values | 0 |
| Duplicate rows | 0 |
| Rating scale | 1 – 5 |
| Languages | Mostly English, with a few Hindi reviews |

The `title` and `body` columns are combined into a single `review_text` field for modelling.

**Class distribution (after mapping ratings to sentiment):**

| Sentiment | Reviews | Share |
| --------- | ------: | ----: |
|  Positive | 729 | 50.6% |
|  Negative | 512 | 35.6% |
|  Neutral | 199 | 13.8% |

##  Exploratory Data Analysis

### Rating Sentiment Distribution

The dataset is **imbalanced**: Positive reviews dominate, while Neutral reviews are the minority class. This is why F1-score (not just accuracy) is used to compare models.

### Overall Word Cloud

Frequent terms such as *phone*, *camera*, *battery*, *performance* and *price* show which product aspects customers talk about most.

### Rating-based vs. VADER sentiment

| Rating-based ↓ / VADER → | Negative | Neutral | Positive |
| ------------------------ | -------: | ------: | -------: |
| **Negative** | 361 | 21 | 130 |
| **Neutral**  | 81  | 6  | 112 |
| **Positive** | 45  | 17 | 667 |

VADER agrees well on clearly positive and negative reviews but struggles with mixed or neutral reviews, which motivated training a supervised model on the rating-based labels.

##  Methodology

```
Raw reviews ─► Combine title + body ─► Rating → Sentiment labels
      │
      ▼
Text preprocessing ─► Feature engineering ─► TF-IDF ─► Model training ─► Evaluation ─► Streamlit app
```

### 1. Text preprocessing

1. Lowercasing
2. Remove URLs, @mentions, HTML tags and numbers
3. Handle hashtags and remove punctuation
4. Normalise whitespace
5. Tokenisation (NLTK)
6. Stop-word removal, **keeping negation words**
7. Lemmatisation (WordNet)
8. Remove single-character tokens

### 2. Feature engineering

Review length, word count, unique word count, average word length, sentence count, exclamation / question / hashtag counts, and VADER `neg`, `neu`, `pos` and `compound` scores.

### 3. Vectorisation

`TfidfVectorizer(max_features=3000, min_df=2, max_df=0.95, ngram_range=(1, 2), sublinear_tf=True)`

### 4. Models compared

Classical machine-learning models on TF-IDF features were compared with a fine-tuned **BERT** model.

##  Model Performance

>  **Selected model: Logistic Regression**

Logistic Regression was chosen as the final model based on **F1-score and the other evaluation metrics**. BERT achieved slightly better results, but its **high computational cost** (training time, memory, and inference latency) makes it a poor fit for a lightweight, freely hosted app. Logistic Regression delivers competitive performance at a tiny fraction of the cost.
| Model | Accuracy | Precision | Recall | F1-score |
| ----- | :------: | :-------: | :----: | :------: |
| **Logistic Regression**  | XX.XX | XX.XX | XX.XX | XX.XX |
| BERT | XX.XX | XX.XX | XX.XX | XX.XX |
| Other models | XX.XX | XX.XX | XX.XX | XX.XX |

**Why not BERT?**

| | Logistic Regression | BERT |
| -- | :--: | :--: |
| Performance | Strong | Slightly higher |
| Training cost | Seconds | High (GPU recommended) |
| Inference speed | Milliseconds | Slower |
| Deployment on free tier |  Easy |  Heavy |

##  Tech Stack

| Category | Tools |
| -------- | ----- |
| Language | Python |
| Data & Analysis | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| NLP | NLTK (tokenisation, stop-words, WordNet, VADER) |
| Machine Learning | scikit-learn (TF-IDF, classifiers) |
| Deep Learning (baseline) | BERT / Hugging Face Transformers |
| Web App | Streamlit |
| Model persistence | Joblib |

##  Project Structure

```
sentiment-analysis/
├── images/
│   ├── rating_sentiment_distribution.png
│   └── wordcloud_overall.png
├── app.py                          # Streamlit web app
├── Sentiment_Analysis_Project.ipynb  # EDA, preprocessing, feature engineering & modelling
├── svm_model.pkl                   # Trained classifier
├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
├── requirements.txt                # Python dependencies
└── README.md
```

##  Getting Started

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/USERNAME/REPO.git
cd REPO

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the app locally

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Run the notebook

```bash
pip install jupyter wordcloud nltk seaborn matplotlib pandas openpyxl
jupyter notebook Sentiment_Analysis_Project.ipynb
```

The notebook downloads the required NLTK data (`vader_lexicon`, `stopwords`, `wordnet`, `punkt`, `punkt_tab`) on first run.

##  Usage

1. Type or paste a review into the text box.
2. Click ** Analyze Sentiment**.
3. View the predicted sentiment, the percentage breakdown across Negative / Neutral / Positive, and progress bars.

**Examples**

| Input | Output |
| ----- | ------ |
| `Battery life is amazing and the camera is superb!` |  POSITIVE |
| `Worst phone ever. It hangs and the display is poor.` |  NEGATIVE |
| `It is okay. Nothing special, does the job.` |  NEUTRAL |

## ️ Deployment on Streamlit Community Cloud

1. Push this repository to GitHub (include `app.py`, both `.pkl` files and `requirements.txt`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and select your repository, branch (`main`) and main file (`app.py`).
4. Click **Deploy**.
5. Copy the generated URL and paste it into the **Live Demo** links at the top of this README.

##  Future Improvements

- [ ] Fine-tune a lightweight transformer (e.g. DistilBERT) to close the gap with BERT at lower cost
- [ ] Handle class imbalance (class weights / SMOTE) to improve Neutral recall
- [ ] Add multilingual support (Hindi reviews are present in the data)
- [ ] Aspect-based sentiment (camera, battery, display, price)
- [ ] Batch prediction from uploaded CSV files

##  Contributing

Contributions, issues and feature requests are welcome!

1. Fork the project
2. Create your branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

Distributed under the MIT License. See `LICENSE` for details.

##  Contact

**Your Name**: your.email@example.com
LinkedIn: [your-profile](https://linkedin.com/in/your-profile) · GitHub: [@USERNAME](https://github.com/USERNAME)

Project Link: [https://github.com/USERNAME/REPO](https://github.com/USERNAME/REPO)

---

⭐ **If you found this project useful, please give it a star!** ⭐
