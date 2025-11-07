# 🧠 The RAG Fallacy: An Architectural Clinic for Trustworthy Agents

A 90-minute masterclass demonstrating why "just having data" isn't enough for trustworthy AI agents, and how to fix it with advanced RAG architectures.

**Instructor:** Pranjal Joshi ([pranjal.alias@gmail.com](mailto:pranjal.alias@gmail.com) | [LinkedIn](https://linkedin.com/in/jpranjal))

## 🎯 What is the RAG Fallacy?

The **RAG Fallacy** is the dangerous assumption that simply having data in a vector database means your AI agent can reliably use it. This demo shows:

1. **The Failure**: A "smart" financial analyst agent with naive RAG that gives confident but wrong answers
2. **The Diagnosis**: Using observability tools to identify the exact architectural failure point
3. **The Fix**: Re-architecting with sentence-window retrieval and reranking
4. **The Proof**: Demonstrating reliable, accurate responses with the advanced architecture

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- OpenAI API key (required)
- Cohere API key (required for reranking)
- LangFuse account (optional, for observability)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ragfails.git
cd ragfails
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### Running the Demo

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 📊 Demo Flow

### Part 1: The "Smart" Agent That Fails (15 min)

The demo presents a Financial Analyst Bot built with naive RAG. Ask it:

> **"Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."**

**The Naive Agent's Failure:**
- May give a confident but incorrect answer
- May miss the nuance that the 10-Q says "14.5%" while the CEO said "approximately 15%"
- May fail to cite both sources correctly
- May say it "can't find the information" despite having the data

This is the **RAG Fallacy** in action.

### Part 2: Architectural Triage with Observability (30 min)

Enable LangFuse tracing to see exactly what went wrong:

```bash
# Set LangFuse credentials in .env
python scripts/test_with_langfuse.py
```

**What You'll See:**
- The exact chunks retrieved by the naive agent
- Irrelevant or fragmented context
- "Lost in the middle" problem where important info is buried
- The LLM never had a chance - the architecture failed it

### Part 3: The Architectural Fix (30 min)

The demo switches to the **Advanced RAG Agent** which implements:

1. **Sentence-Window Retrieval**
   - Retrieves precise sentences for matching
   - Includes surrounding context window
   - Better semantic boundaries

2. **Reranking**
   - Uses Cohere's rerank-english-v3.0
   - Re-scores retrieved chunks for relevance
   - Ensures best context reaches the LLM

### Part 4: The Result & The Funnel (15 min)

Ask the same question to the Advanced Agent:

> **"Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."**

**The Advanced Agent's Success:**
- Accurately identifies 14.5% from the 10-Q
- Finds the CEO's "approximately 15%" statement
- Explains the discrepancy clearly
- Cites both sources correctly

**Observability Proof:**
- LangFuse trace shows clean, relevant context chunks
- Reranking scores demonstrate proper prioritization
- The LLM receives exactly what it needs

## 🏗️ Architecture

### Naive RAG Agent

```
Document → Fixed Chunks (1000 chars) → Vector DB
Query → Vector Search (Top 3) → LLM → Response
```

**Problems:**
- Fixed chunking breaks semantic boundaries
- No reranking - first-pass results go straight to LLM
- Limited context window
- Information spread across chunks gets lost

### Advanced RAG Agent

```
Document → Sentences + Windows → Vector DB
Query → Vector Search (Top 10) → Reranking (Top 3) → LLM → Response
```

**Improvements:**
- Sentence-level retrieval with context windows
- Two-stage retrieval: broad search → precise reranking
- Better context assembly
- Handles multi-chunk information effectively

## 📁 Project Structure

```
ragfails/
├── app.py                          # Streamlit web application
├── data/                           # Sample financial documents
│   ├── acme_corp_q1_2024_10q.txt
│   └── acme_corp_q1_2024_earnings_call.txt
├── src/
│   ├── agents/
│   │   ├── naive_rag.py           # Naive RAG implementation
│   │   └── advanced_rag.py        # Advanced RAG implementation
│   ├── utils/
│   │   └── document_loader.py     # Document loading utilities
│   └── config.py                   # Configuration
├── scripts/
│   ├── test_agents.py             # Test script for both agents
│   └── test_with_langfuse.py      # LangFuse integration demo
├── requirements.txt
└── README.md
```

## 🧪 Testing

### Basic Testing

Test both agents with a sample query:

```bash
python scripts/test_agents.py
```

### With LangFuse Observability

```bash
# Make sure LangFuse credentials are in .env
python scripts/test_with_langfuse.py
```

Then visit your LangFuse dashboard to see the traces.

## 🎓 Key Takeaways

1. **Vector Search ≠ Architecture**: Just having embeddings and vector search is not an architecture
2. **Observability is Critical**: You can't fix what you can't see
3. **Retrieval is the Bottleneck**: Most RAG failures happen at retrieval, not generation
4. **Advanced Techniques Matter**: Sentence-window + reranking dramatically improves reliability

## 🛠️ Technologies Used

- **LLM**: OpenAI GPT-4
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector DB**: ChromaDB
- **Reranking**: Cohere rerank-english-v3.0
- **Observability**: LangFuse
- **UI**: Streamlit

## 📚 Further Learning

This masterclass is Part 0 of a 12-week course on building production-ready AI agents. The full course covers:

- **Milestone 1**: Giving agents tools and memory
- **Milestone 2**: Multi-agent coordination and planning
- **Milestone 3**: Advanced RAG and knowledge systems (this masterclass)
- **Milestone 4**: Production deployment, governance, and cost control

Interested? Contact Pranjal Joshi at [pranjal.alias@gmail.com](mailto:pranjal.alias@gmail.com)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Pranjal Joshi**
- Email: pranjal.alias@gmail.com
- LinkedIn: [linkedin.com/in/jpranjal](https://linkedin.com/in/jpranjal)

---

*"The single biggest failure point in agentic systems is bad knowledge retrieval. Fix the architecture, fix the agent."*