# RAG Fallacy Masterclass - Instructor Guide

## Pre-Class Preparation (30 minutes before)

### Technical Setup
- [ ] Start the Streamlit app: `streamlit run app.py`
- [ ] Verify both agents respond correctly
- [ ] Open LangFuse dashboard in a separate tab (if using)
- [ ] Test screen sharing and resolution
- [ ] Have both documents open for reference

### Materials Ready
- [ ] This guide open
- [ ] Streamlit app running
- [ ] LangFuse dashboard (optional)
- [ ] Sample documents accessible

## Detailed Flow

### Part 1: The "Smart" Agent That Fails (15 minutes)

#### Introduction (3 min)
**Script:**
> "Welcome to The RAG Fallacy masterclass. Today we're going to explore the single biggest failure point in agentic systems: bad knowledge retrieval.
>
> I'm going to show you something dangerous. I have a 'smart' Financial Analyst Bot right here. It has access to a company's 10-Q report and earnings call transcript. It uses vector embeddings, a vector database, and GPT-4. By all accounts, this should work perfectly.
>
> But it doesn't. And that's the RAG Fallacy."

**Action:** Show the Streamlit interface

#### The Demo Query (5 min)

**Show the test query:**
> "Compare the Q1 revenue growth mentioned in the 10-Q report with the CEO's statements in the earnings call."

**Explain why this is hard:**
> "This question requires:
> 1. Finding specific numbers from the 10-Q (14.5%)
> 2. Finding the CEO's statement from the earnings call ('approximately 15%')
> 3. Understanding they're describing the same thing
> 4. Explaining why there's a discrepancy in how it's presented"

**Action:** Run the query on the **Naive RAG Agent only**

#### Analyzing the Failure (7 min)

**Examine the response:**
- Is it accurate?
- Does it cite both sources?
- Does it explain the nuance?
- Is it confident despite being wrong?

**Show the retrieved chunks:**
> "Now let's look at what the agent actually saw. Click 'View Retrieved Context'."

**Point out issues:**
- Chunks might not contain the key information
- Information might be fragmented
- Context might be cut off mid-sentence
- Relevant information might be ranked low

**Key teaching moment:**
> "The LLM didn't fail. GPT-4 is perfectly capable of answering this question IF it had the right context. The architecture that feeds it failed. This is the RAG Fallacy - just because you have the data doesn't mean your agent can use it."

---

### Part 2: Architectural Triage with Observability (30 minutes)

#### Opening the Hood (10 min)

**Explain observability:**
> "In production systems, you can't fix what you can't see. Let's use observability tools to diagnose exactly what went wrong."

**Option A - If using LangFuse:**
- Open LangFuse dashboard
- Show the trace for the failed query
- Walk through the retrieval span
- Show the exact chunks that were retrieved
- Highlight the reranking scores (or lack thereof)

**Option B - Without LangFuse:**
- Use the "Retrieved Context" section in the Streamlit app
- Manually analyze each retrieved chunk
- Discuss what's missing

#### Root Cause Analysis (10 min)

**Walk through the naive architecture:**
```
Document → Fixed Chunks (1000 chars) → Vector DB
Query → Vector Search (top 3) → LLM → Response
```

**Identify the failure points:**

1. **Chunking Strategy**
   > "Fixed-size chunks ignore semantic boundaries. We're splitting on arbitrary character counts, not on meaning. Look at this chunk - it cuts off mid-sentence."

2. **No Reranking**
   > "Vector similarity is a good first pass, but it's not enough. We retrieve the top 3 by cosine similarity and send them straight to the LLM. What if the most relevant information is in position 4? We'll never know."

3. **Limited Context**
   > "With only 3 chunks, we're betting everything on those three retrievals being perfect. For complex questions that span multiple documents, this is a losing bet."

**Key teaching moment:**
> "Vector search is not an architecture. It's one component of an architecture. What we need is a systematic approach to ensure the right information reaches the LLM."

#### Introducing Advanced Patterns (10 min)

**Explain the two key improvements:**

**1. Sentence-Window Retrieval**
```
Instead of:  [----1000 char chunk----][----1000 char chunk----]
We use:      [sentence] with [surrounding window of 3 sentences]
```

> "We embed small, precise sentences for matching, but we retrieve larger context windows. This gives us precision in matching and context in delivery."

**Visual on whiteboard or slides:**
- Show document split into sentences
- Show how each sentence has a window
- Show how retrieval works

**2. Reranking**
> "Two-stage retrieval: Cast a wide net with vector search (top 10), then use a specialized reranking model to intelligently re-score and select the best 3.
>
> Cohere's rerank model is specifically trained for this task. It understands the query context better than pure vector similarity."

---

### Part 3: The Architectural Fix (30 minutes)

#### Live Implementation Walkthrough (15 min)

**Show the code structure:**
> "Let me show you how these improvements are implemented."

**Navigate to:** `src/agents/advanced_rag.py`

**Key sections to highlight:**

1. **Sentence Splitting** (line ~80-110 in document_loader.py)
```python
def sentence_splitter(text: str) -> List[Dict[str, any]]:
    # Creates sentence + window pairs
```

> "Notice we're storing both the sentence and its window. The sentence is what we embed and search on. The window is what we show to the LLM."

2. **Retrieval** (line ~120-150 in advanced_rag.py)
```python
def retrieve(self, query: str, top_k: int = 10):
    # Retrieve top 10 sentences
```

> "First pass: cast a wide net. We get 10 candidates, not 3."

3. **Reranking** (line ~150-180 in advanced_rag.py)
```python
def rerank(self, query: str, retrieved_items: List[Dict], top_k: int = 3):
    # Use Cohere to rerank to top 3
```

> "Second pass: intelligent filtering. Cohere's rerank model looks at the query and each candidate together, scoring true relevance."

**Key teaching moment:**
> "This is what an architecture looks like. Not just 'use embeddings', but a systematic pipeline with checks and balances."

#### The Switch (2 min)

**In the Streamlit app:**
> "Now, watch this. I'm going to toggle to the Advanced RAG Agent and ask the exact same question."

**Action:** Uncheck "Naive Agent", ensure "Advanced Agent" is checked

#### The Demo (13 min)

**Run the same query on Advanced Agent:**

**Examine the response:**
- Is it accurate?
- Does it cite both sources correctly?
- Does it explain the nuance?
- Compare with the naive response

**Show the retrieved context:**
> "Now let's see what this agent retrieved. Click 'View Retrieved Context'."

**Point out improvements:**
- More precise sentence matches
- Context windows provide surrounding information
- Reranking scores show relevance
- Information from both documents is present

**Compare side-by-side:**
- Show naive response vs advanced response
- Highlight accuracy differences
- Show retrieved context differences

**Key teaching moment:**
> "Same data. Same LLM. Different architecture. Dramatically different results. This is the proof that architecture matters more than people think."

---

### Part 4: The Result & The Funnel (15 minutes)

#### Observability Proof (5 min)

**If using LangFuse:**
- Show the trace for the advanced query
- Compare with the naive trace
- Highlight the quality difference in retrieved context
- Show reranking scores

**Key point:**
> "This is why observability matters. Without seeing these traces, you'd just know that one worked and one didn't. But you wouldn't know WHY. And you can't improve what you can't measure."

#### Summary of Improvements (5 min)

**Create a comparison table on screen or whiteboard:**

| Aspect | Naive RAG | Advanced RAG |
|--------|-----------|--------------|
| Chunking | Fixed 1000 chars | Sentences + windows |
| Initial retrieval | Top 3 | Top 10 |
| Reranking | None | Cohere rerank-v3 |
| Final context | 3 chunks | 3 reranked windows |
| Accuracy on test query | Poor | Excellent |

**Key teaching moments:**
> "These improvements aren't complex. They're not using exotic models or techniques. They're systematic architectural choices.
>
> The sentence-window pattern is well-documented. Reranking has been around for years. But most RAG implementations don't use them. Why? Because people think 'vector database = RAG = solved.' That's the fallacy."

#### The Funnel to Full Course (5 min)

**Transition to bigger picture:**
> "We just spent 90 minutes fixing this agent's knowledge retrieval. That's Milestone 3 in the full 12-week course.
>
> But this agent still can't:
> - Use tools (Milestone 1: Tool use and memory)
> - Work with other agents (Milestone 2: Multi-agent coordination)
> - Run in production safely (Milestone 4: Governance and deployment)
>
> Today you saw how to build a trustworthy agent's 'brain' - its knowledge system. The full course shows you how to build the complete agent: tools, planning, memory, coordination, and production deployment."

**Show course structure briefly:**
- **Milestone 1** (Weeks 1-3): Tool-using agents with memory
- **Milestone 2** (Weeks 4-6): Multi-agent systems and planning
- **Milestone 3** (Weeks 7-9): Advanced RAG and knowledge systems ← Today's topic
- **Milestone 4** (Weeks 10-12): Production deployment and governance

**Call to action:**
> "If you found this valuable and want to go deeper, I'm running the full course starting [DATE]. We'll build production-ready agents from scratch, with real-world examples and hands-on coding.
>
> You'll walk away with working agents you can deploy, and more importantly, the architectural thinking to build trustworthy AI systems.
>
> Contact me at pranjal.alias@gmail.com or connect on LinkedIn."

---

## Q&A Preparation

### Expected Questions and Answers

**Q: "Why not just use a better embedding model?"**
A: Embedding quality helps, but it doesn't solve the core problems. Even perfect embeddings can't overcome poor chunking or lack of reranking. Architecture > models.

**Q: "What about other vector databases like Pinecone or Weaviate?"**
A: The vector database choice matters less than how you use it. These patterns work with any vector DB. The failure isn't the database, it's the pipeline.

**Q: "Doesn't GPT-4 have a huge context window now? Why not just stuff everything in?"**
A: Three reasons: (1) Cost - tokens are expensive, (2) Performance - "lost in the middle" problem where LLMs miss info in long contexts, (3) Latency - larger contexts = slower responses.

**Q: "How do you handle documents larger than the context window?"**
A: That's exactly why retrieval matters. You MUST retrieve selectively. The techniques shown today - sentence-window + reranking - scale to any document size.

**Q: "Is sentence-window always better than fixed chunks?"**
A: Usually, but not always. For very structured documents (tables, forms), you might want different strategies. The key lesson is: think architecturally about chunking, don't use defaults blindly.

**Q: "What's the cost of using Cohere for reranking?"**
A: Very reasonable. Free tier includes 1000 calls/month. Production pricing is $2 per 1000 calls. Given the accuracy improvement, it's worth it.

**Q: "Can I use open-source models for reranking?"**
A: Yes! Sentence-transformers has cross-encoder models for reranking. Cohere's commercial model is better, but open-source works.

**Q: "How do I know if my RAG is failing in production?"**
A: Observability! LangFuse, LangSmith, or custom logging. Track: retrieval relevance, answer accuracy, user feedback. Without observability, you're flying blind.

**Q: "What about fine-tuning the LLM instead?"**
A: Fine-tuning can help, but it doesn't fix bad retrieval. If the LLM isn't seeing the right context, no amount of fine-tuning will help. Fix retrieval first.

## Backup Materials

### If Demo Fails
- Have screenshots of successful runs
- Walk through the code instead
- Use the test scripts to show output in terminal

### If Questions Run Short
- Demo the LangFuse integration
- Show the data documents and discuss how they were created
- Do a live code walkthrough of one of the agents

### If Time Runs Over
- Skip the detailed code walkthrough
- Focus on the demo and results
- Provide code as take-home material

## Post-Class Follow-Up

### Materials to Share
- GitHub repository link
- Slides (if any)
- LangFuse dashboard access (if they want to experiment)
- Contact information
- Course information and registration link

### Success Metrics
- Participants understand what the RAG Fallacy is
- Participants can articulate why naive RAG fails
- Participants know what sentence-window and reranking are
- Participants are interested in the full course

---

**Good luck with the masterclass! 🚀**

*For questions or support: pranjal.alias@gmail.com*
