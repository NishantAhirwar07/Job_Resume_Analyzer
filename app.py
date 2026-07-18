import streamlit as st
import plotly.graph_objects as go
import PyPDF2
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------------------
# NLTK setup
# ----------------------------------------------------------------------------
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# ----------------------------------------------------------------------------
# Page Setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Job Match Scorer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Custom CSS — gives the app a polished, "designed" look
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
    }
    .hero h1 {
        color: white;
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.02rem;
        margin: 0;
    }

    /* Section card */
    .card {
        background: #ffffff;
        border: 1px solid #eef0f4;
        border-radius: 16px;
        padding: 1.6rem 1.6rem;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.04);
        margin-bottom: 1.2rem;
    }

    .verdict-box {
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-weight: 600;
        font-size: 0.98rem;
        margin-top: 0.6rem;
    }

    /* Keyword pills */
    .pill {
        display: inline-block;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.22rem;
    }
    .pill-match {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #86efac;
    }
    .pill-missing {
        background: #fee2e2;
        color: #b91c1c;
        border: 1px solid #fca5a5;
    }

    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.7rem;
    }

    div[data-testid="stFileUploader"] section {
        border-radius: 12px;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📄 Resume Job Match Scorer</h1>
    <p>Upload your resume and paste a job description to instantly see how well they align,
    powered by TF-IDF + Cosine Similarity NLP matching.</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This tool helps you:\n\n"
        "- Measure how your resume matches a job description\n"
        "- Identify important job keywords\n"
        "- Spot missing terms to improve your resume"
    )
    st.header("⚙️ How It Works")
    st.write(
        "1. Upload your resume (PDF)\n"
        "2. Paste the job description\n"
        "3. Click **Analyze Match**\n"
        "4. Review your score & suggestions"
    )
    st.divider()
    st.caption("Built with Streamlit · scikit-learn · NLTK")

# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text = text + (page.extract_text() or "")
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])


def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed


def extract_keywords(text, num_keywords=15):
    words = word_tokenize(text)
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    nouns = [w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    word_freq = Counter(nouns)
    return word_freq.most_common(num_keywords)


def score_theme(score):
    """Returns (color, verdict text, bg color) based on score."""
    if score < 40:
        return "#dc2626", "⚠️ Low Match — consider tailoring your resume more closely to this role.", "#fef2f2"
    elif score < 70:
        return "#d97706", "👍 Good Match — your resume aligns fairly well with this role.", "#fffbeb"
    else:
        return "#16a34a", "🎉 Excellent Match — your resume strongly aligns with this role!", "#f0fdf4"


def make_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
        },
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


# ----------------------------------------------------------------------------
# Main App
# ----------------------------------------------------------------------------
def main():
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📎 Upload Resume</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧾 Job Description</div>', unsafe_allow_html=True)
        job_description = st.text_area(
            "Paste the job description", height=185, label_visibility="collapsed",
            placeholder="Paste the full job description here..."
        )
        st.markdown('</div>', unsafe_allow_html=True)

    analyze_clicked = st.button("🔍 Analyze Match", use_container_width=True, type="primary")

    if analyze_clicked:
        if not uploaded_file:
            st.warning("Please upload your resume")
            return
        if not job_description:
            st.warning("Please paste the job description")
            return

        with st.spinner("Analyzing your resume against the job description..."):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from PDF. Please try another PDF.")
                return

            similarity_score, resume_processed, job_processed = calculate_similarity(
                resume_text, job_description
            )
            color, verdict, bg = score_theme(similarity_score)

        st.markdown("### 📊 Results")

        res_col1, res_col2 = st.columns([1, 1.3], gap="large")

        with res_col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.plotly_chart(make_gauge(similarity_score, color), use_container_width=True)
            st.markdown(
                f'<div class="verdict-box" style="background:{bg};color:{color};">{verdict}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔑 Keyword Analysis</div>', unsafe_allow_html=True)

            job_keywords = [w for w, _ in extract_keywords(job_processed, num_keywords=20)]
            resume_words = set(resume_processed.split())

            matched = [kw for kw in job_keywords if kw in resume_words]
            missing = [kw for kw in job_keywords if kw not in resume_words]

            m1, m2 = st.columns(2)
            m1.metric("Matched Keywords", len(matched))
            m2.metric("Missing Keywords", len(missing))

            st.write("**✅ Found in your resume:**")
            if matched:
                st.markdown(
                    "".join([f'<span class="pill pill-match">{kw}</span>' for kw in matched]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("No overlapping keywords found.")

            st.write("")
            st.write("**❌ Missing from your resume:**")
            if missing:
                st.markdown(
                    "".join([f'<span class="pill pill-missing">{kw}</span>' for kw in missing]),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("Great — no major keywords missing!")

            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🧠 View processed text (debug)"):
            t1, t2 = st.columns(2)
            with t1:
                st.caption("Processed Resume Text")
                st.text_area("Resume", resume_processed, height=200, label_visibility="collapsed")
            with t2:
                st.caption("Processed Job Description Text")
                st.text_area("Job Description", job_processed, height=200, label_visibility="collapsed")


if __name__ == "__main__":
    main()
