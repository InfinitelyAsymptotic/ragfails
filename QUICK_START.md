# Quick Start Guide

Get the demo running in 5 minutes!

## 1. Clone and Install

```bash
git clone https://github.com/yourusername/ragfails.git
cd ragfails
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your keys:
# - OPENAI_API_KEY (required)
# - COHERE_API_KEY (required)
```

Get keys from:
- OpenAI: https://platform.openai.com/api-keys
- Cohere: https://dashboard.cohere.ai/api-keys

## 3. Run the Demo

```bash
streamlit run app.py
```

## 4. Try the Test Query

In the web interface, ask:

> "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

Compare the responses from Naive vs Advanced RAG agents!

## What to Look For

**Naive RAG** (🔴):
- May miss key information
- May not cite both sources correctly
- May give confident but wrong answers

**Advanced RAG** (🟢):
- Accurately finds information from both documents
- Correctly identifies 14.5% (10-Q) vs "approximately 15%" (CEO)
- Explains the nuance clearly

## Need Help?

- See `SETUP_GUIDE.md` for detailed setup
- See `MASTERCLASS_GUIDE.md` for instructor notes
- Contact: pranjal.alias@gmail.com
