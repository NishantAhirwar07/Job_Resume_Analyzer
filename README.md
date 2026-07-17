# 📄 Resume Job Match Scorer

A machine learning powered web app that analyzes how well a resume matches a job description using **TF-IDF vectorization** and **Cosine Similarity**. Upload a resume (PDF), paste a job description, and instantly get a match score along with keyword insights to help improve resume alignment.

**🔗 Live Demo:** []

---

## ✨ Features

- 📤 Upload resume in PDF format
- 📝 Paste any job description as plain text
- 🎯 Get an instant **Match Score (%)** using TF-IDF + Cosine Similarity
- 📊 Visual gauge chart showing match strength (Low / Good / Excellent)
- 🔑 Extract top keywords from the job description (nouns & adjectives via POS tagging)
- 💡 Actionable feedback to help tailor your resume

---

## 🖥️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Web Framework | [Streamlit](https://streamlit.io/) |
| NLP | [NLTK](https://www.nltk.org/) (tokenization, stopwords, POS tagging) |
| Machine Learning | [scikit-learn](https://scikit-learn.org/) (TF-IDF Vectorizer, Cosine Similarity) |
| PDF Parsing | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| Visualization | [Matplotlib](https://matplotlib.org/) |

---

## 🧠 How It Works

1. **Text Extraction** – The uploaded PDF resume is parsed and converted to raw text using `PyPDF2`.
2. **Text Cleaning** – Both the resume and job description are lowercased, stripped of special characters, and normalized.
3. **Stopword Removal** – Common connector words (e.g. "is", "the", "to") are removed using NLTK's stopword corpus.
4. **Vectorization** – The cleaned resume and job description are converted into numeric vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
5. **Similarity Scoring** – **Cosine Similarity** is calculated between the two TF-IDF vectors to produce a match percentage.
6. **Keyword Extraction** – Nouns and adjectives are extracted from the job description using POS tagging to highlight the most important terms.

---

## 📂 Project Structure

```
resume-job-match-scorer/
│
├── app.py                 # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md                # Project documentation
└── resume_job_eligiblity_checker.ipynb   # Original notebook (development/prototyping)
```

---

## ⚙️ Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/resume-job-match-scorer.git
cd resume-job-match-scorer
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app locally
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

> **Note:** On first run, the app downloads a few small NLTK resources (`punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`). This requires an internet connection the first time you run it.

---

## 🚀 Deploying to Streamlit Community Cloud

1. Push this project to a **public GitHub repository**, making sure it includes `app.py` and `requirements.txt` at the root.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with your GitHub account.
3. Click **"New app"** and select:
   - **Repository:** `<your-username>/resume-job-match-scorer`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**. Streamlit Cloud will install the dependencies from `requirements.txt` and launch your app automatically.
5. Once live, copy the app URL and add it to the **Live Demo** link at the top of this README.

---

## 📸 Screenshots

> Add screenshots of the app here after deployment, for example:
> ```markdown
> ![App Screenshot](assets/screenshot.png)
> ```

---

## 🔮 Future Improvements

- Support for `.docx` resumes in addition to PDF
- Multi-resume comparison against a single job description
- Suggested resume edits based on missing high-weight keywords
- Support for multiple job descriptions at once (batch scoring)
- Replace TF-IDF with sentence embeddings (e.g. `sentence-transformers`) for semantic matching

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or an issue.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request



## 👤 Author

**[Nishant Ahirwar]**
- GitHub: [@your-username](https://github.com/NishantAhirwar07)
- LinkedIn: [your-linkedin](https://www.linkedin.com/in/nishant-ahirwar-6aa7a8333?utm_source=share_via&utm_content=profile&utm_medium=member_android)

---

