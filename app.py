import streamlit as st
import joblib
import numpy as np

# LOAD MODEL AND TF-IDF VECTORIZER


model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")



# PAGE SETTINGS

st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)



# CUSTOM CSS# 

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #dbeafe, #fce7f3, #fef3c7);
}

/* Main title */
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #7c3aed;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #475569;
    margin-bottom: 30px;
}

/* Result box */
.result {
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 25px;
}

/* Positive */
.positive {
    background-color: #bbf7d0;
    color: #166534;
    border: 2px solid #22c55e;
}

/* Negative */
.negative {
    background-color: #fecaca;
    color: #991b1b;
    border: 2px solid #ef4444;
}

/* Neutral */
.neutral {
    background-color: #fde68a;
    color: #92400e;
    border: 2px solid #f59e0b;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: bold;
    color: #4f46e5;
    margin-top: 20px;
}

/* Info box */
.info-box {
    background-color: rgba(255,255,255,0.7);
    padding: 15px;
    border-radius: 12px;
    margin-top: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# HEADER


st.markdown(
    '<div class="title">💬 AI Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze text using TF-IDF and Support Vector Machine (SVM)</div>',
    unsafe_allow_html=True
)



# TEXT INPUT

text = st.text_area(
    "📝 Enter your review:",
    height=150,
    placeholder="Example: I really enjoyed this product!"
)


# ANALYZE BUTTON

if st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True
):

    # Check empty input
    if text.strip() == "":
        st.warning("⚠️ Please enter some text.")

    else:

        
        # TF-IDF TRANSFORMATION

        text_vector = vectorizer.transform([text])



        # SVM PREDICTION

        prediction = model.predict(text_vector)[0]

        # Get SVM decision scores
        scores = model.decision_function(text_vector)[0]

        # Get class names
        classes = model.classes_


        # FIND HIGHEST SCORE

        highest_index = np.argmax(scores)

        prediction = classes[highest_index]


        # RESULT STYLE

        if prediction == "Positive":

            emoji = "😊"
            css_class = "positive"

        elif prediction == "Negative":

            emoji = "😞"
            css_class = "negative"

        else:

            emoji = "😐"
            css_class = "neutral"


        # FINAL PREDICTION

        st.markdown(
            f"""
            <div class="result {css_class}">
                {emoji} {prediction.upper()}
            </div>
            """,
            unsafe_allow_html=True
        )


    
        # SENTIMENT PERCENTAGES
        
        # Convert SVM decision scores into
        # relative percentages.

        exp_scores = np.exp(scores - np.max(scores))

        percentages = (
            exp_scores / np.sum(exp_scores)
        ) * 100


        percentage_dict = {
            str(classes[i]): float(percentages[i])
            for i in range(len(classes))
        }

        # DISPLAY PERCENTAGES

        st.markdown(
            '<div class="section-title">📊 Sentiment Distribution</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "😞 Negative",
                f"{percentage_dict.get('Negative', 0):.2f}%"
            )

        with col2:

            st.metric(
                "😐 Neutral",
                f"{percentage_dict.get('Neutral', 0):.2f}%"
            )

        with col3:

            st.metric(
                "😊 Positive",
                f"{percentage_dict.get('Positive', 0):.2f}%"
            )


        # PROGRESS BARS

        st.write("")

        st.write("😞 **Negative**")
        st.progress(
            int(percentage_dict.get("Negative", 0))
        )

        st.write("😐 **Neutral**")
        st.progress(
            int(percentage_dict.get("Neutral", 0))
        )

        st.write("😊 **Positive**")
        st.progress(
            int(percentage_dict.get("Positive", 0))
        )

        # ANALYZED TEXT

        st.markdown(
            '<div class="section-title">📝 Analyzed Text</div>',
            unsafe_allow_html=True
        )

        st.info(text)


# FOOTER
st.markdown(
    """
    <div class="footer">
        🤖 Built using Python • TF-IDF • SVM • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

