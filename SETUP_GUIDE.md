# Setup Guide for RAG Fallacy Masterclass

This guide will help you set up the demo environment for the masterclass.

## Prerequisites Checklist

- [ ] Python 3.9 or higher installed
- [ ] Git installed
- [ ] OpenAI API account with credits
- [ ] Cohere API account (free tier works)
- [ ] LangFuse account (optional, free tier available)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ragfails.git
cd ragfails
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected installation time:** 2-3 minutes

### 4. Get API Keys

#### OpenAI API Key (Required)

1. Go to https://platform.openai.com/api-keys
2. Sign in or create an account
3. Click "Create new secret key"
4. Copy the key (starts with `sk-`)
5. **Note:** You'll need credits in your account (~$1-2 for the demo)

#### Cohere API Key (Required)

1. Go to https://dashboard.cohere.ai/api-keys
2. Sign in or create an account
3. Copy your API key from the dashboard
4. **Note:** Free tier includes 1000 API calls/month

#### LangFuse Keys (Optional)

1. Go to https://cloud.langfuse.com
2. Create an account
3. Create a new project
4. Go to Settings → API Keys
5. Copy both the Public Key and Secret Key

### 5. Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```bash
# Required
OPENAI_API_KEY=sk-...your-key...
COHERE_API_KEY=...your-key...

# Optional (for observability demo)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

### 6. Verify Setup

Run the test script to verify everything works:

```bash
python scripts/test_agents.py
```

**Expected output:**
- Should see "Initializing Naive RAG Agent..."
- Should see document indexing progress
- Should receive responses from both agents
- No error messages

**If you see errors:**
- Check API keys are correct in `.env`
- Ensure virtual environment is activated
- Verify all dependencies installed correctly

### 7. Launch the Web Application

```bash
streamlit run app.py
```

The browser should open automatically to `http://localhost:8501`

**Troubleshooting:**
- If port 8501 is busy: `streamlit run app.py --server.port 8502`
- If browser doesn't open: manually navigate to the URL shown in terminal

## Pre-Masterclass Checklist

Before the live demo, verify:

- [ ] Web app loads without errors
- [ ] Both agents can be queried successfully
- [ ] Test query returns different results from naive vs advanced
- [ ] Retrieved context is visible in expandable sections
- [ ] (Optional) LangFuse traces appear in dashboard

## Cost Estimates

**Per demo run (one query to both agents):**
- Embeddings: ~$0.001
- LLM calls: ~$0.01
- Cohere reranking: Free (within limits)
- **Total: ~$0.011 per query**

**For 90-minute masterclass:**
- Estimated queries: 10-20
- Estimated cost: $0.10 - $0.20

## Common Issues and Solutions

### Issue: "OPENAI_API_KEY is required"
**Solution:** Make sure `.env` file exists and contains the API key

### Issue: "Reranking failed"
**Solution:** Check COHERE_API_KEY in `.env` file

### Issue: ChromaDB errors
**Solution:** Delete `.chroma_db` directory and restart

### Issue: Port already in use
**Solution:** Use a different port: `streamlit run app.py --server.port 8502`

### Issue: Slow response times
**Solution:**
- First query is always slower (indexing)
- Subsequent queries should be faster
- Check internet connection

## Technical Architecture

### Data Flow - Naive RAG
```
User Query
  ↓
OpenAI Embedding API (query)
  ↓
ChromaDB Vector Search (top 3)
  ↓
OpenAI Chat API (GPT-4o-mini)
  ↓
Response
```

### Data Flow - Advanced RAG
```
User Query
  ↓
OpenAI Embedding API (query)
  ↓
ChromaDB Vector Search (top 10)
  ↓
Cohere Rerank API (top 3)
  ↓
OpenAI Chat API (GPT-4o-mini)
  ↓
Response
```

## Files Overview

- `app.py` - Streamlit web application
- `src/agents/naive_rag.py` - Naive RAG implementation
- `src/agents/advanced_rag.py` - Advanced RAG implementation
- `src/utils/document_loader.py` - Document loading utilities
- `src/config.py` - Configuration management
- `data/*.txt` - Sample financial documents
- `scripts/test_agents.py` - Testing script
- `scripts/test_with_langfuse.py` - Observability demo script

## Support

If you encounter issues during setup:

1. Check this guide's troubleshooting section
2. Verify all prerequisites are met
3. Review error messages carefully
4. Contact: pranjal.alias@gmail.com

## Next Steps

Once setup is complete:
1. Familiarize yourself with the web interface
2. Try the test query multiple times
3. Examine the retrieved context from both agents
4. (Optional) Set up LangFuse and review traces
5. Read through the instructor notes in `MASTERCLASS_GUIDE.md`
