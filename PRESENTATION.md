# Presentation: Influence-Aware RAG Pipeline 🔬

This document provides a conceptual breakdown of the core mechanics behind the **Influence-Aware RAG** project, tailored for presentations or architectural reviews.

---

## 1. How THIS RAG Pipeline Works (In Maximum Detail)

Unlike standard RAG architectures that simply retrieve and generate, this application is a **diagnostic testing harness**. Its purpose is to test the _alignment_ between the Retrieval engine and the Generation engine.

The pipeline executes in 7 distinct algorithmic phases per run:

1. **Phase 1: Data Structuring:** We load a set of pre-answered factual questions from a real-world dataset. The original long document for each question is split into discrete "passages" of exactly ~100 words. These passages act as our corpus.
2. **Phase 2: Lexical Retrieval (BM25):** We ask the BM25 retrieval engine to search the corpus and rank the top 10 most relevant passages for the question. We record these as `Retrieval Rank 1` through `10`.
3. **Phase 3: Baseline Generation:** We combine the user's question with all 10 passages and ask the LLM (e.g., `openai/gpt-oss-20b`) to answer the question. This forms our "Baseline Answer."
4. **Phase 4: Systematic Ablation (The Drop-One Loop):** This is the core engine. We run 10 separate, independent LLM calls. In each call, we hide exactly _one_ of the 10 passages. We record the 10 new, slightly different answers.
5. **Phase 5: Scoring the Damage (ROUGE-L):** We mathematically compare each of the 10 new answers against the original Baseline Answer to see how much the text changed.
6. **Phase 6: Influence Ranking:** We assign an `Influence Score` to each passage based on the damage its removal caused. The passage that caused the biggest change gets `Influence Rank 1`. The passage that caused zero change gets `Influence Rank 10`.
7. **Phase 7: Alignment Evaluation (Spearman's ρ):** We statistically compare the `Retrieval Ranks` (what the retriever thought was important) against the `Influence Ranks` (what the LLM actually used).

This entire 7-phase loop is repeated for (e.g.) 50 different queries, generating a comprehensive statistical analysis of the system's behavior.

### What the Simulation Mode Tests (In a Nutshell)

When running the dashboard in **Simulation Mode** (instead of passing real LLM data through a CSV), the system bypasses the 7-phase loop above and uses random number generators to mimic the math outputs.

**The Simulation does NOT test the LLM or the Retriever models at all.** It is purely a UI/UX test harness designed to:

1. Preview how the frontend dashboard responds to statistical variations natively.
2. Ensure the mathematical UI logic (such as flagging ρ < 0.7 as Divergent) works flawlessly before spending money on huge LLM ablation runs.
3. Verify failure mode detectors behave correctly under randomized mathematical constraints.

---

## 2. The Retrieval Engine: How BM25 Works

**What is it?**
To keep this project lightweight, fast, and focused purely on ablation testing, we use **BM25 (Best Matching 25)** instead of vector databases (like Pinecone) or deep neural embeddings (like OpenAI `text-embedding-v3`).

BM25 is a sparse, lexical (keyword-based) search algorithm. It creates an in-memory index of the document terms on the fly using the `rank_bm25` library based on term frequency (how often a word appears in the document) and inverse document frequency (how rare the word is across all documents).

**What exactly is the system testing?**
BM25 is famously "dumb" in that it relies strictly on exact keyword matching, not semantic meaning. The system is specifically testing whether the LLM is smart enough to find the answer buried in BM25's potentially flawed top-10 list, or if the LLM blindly relies on whatever BM25 puts simply at Rank #1.

**Example Query Mechanics:**

- **Query:** _"who played the main character in the movie gladiator?"_
- **BM25 Behavior:** It strips stop words and looks for high frequencies of "played", "main", "character", "movie", "gladiator".
- **The Retrieval:**
  - **Rank 1 Passage:** Might be a trivia paragraph containing the words "movie", "movie", "gladiator", "gladiator" constantly, but completely missing the actor's name.
  - **Rank 4 Passage:** Might simply say: _"Russell Crowe played Maximus, the main character."_
- **The Test:** Our ablation logic will reveal this discrepancy. Removing Rank 1 won't change the LLM's answer at all (Influence Rank 10). Removing Rank 4 will break the LLM's answer completely (Influence Rank 1).

---

## 3. The Dataset: Google Natural Questions (NQ)

The project uses the **Google Natural Questions (NQ)** dataset, accessed via the HuggingFace `datasets` library.

**What is it?**
NQ is a massive, gold-standard dataset of real, organically issued queries typed into the Google search engine by real users.

**How does it work in this project?**

1. **Real User Queries:** Each row contains a real question (e.g., _"where did they film high school musical two"_).
2. **Wikipedia Grounding:** Each question is paired with the full Wikipedia article that contains the answer.
3. **Short Answer Extraction:** We aggressively filter the dataset to only include questions annotated by humans as having a specific, factual "short answer" (e.g., in this case, the span _"Utah"_).
4. **Streaming & Cleaning:** The dataset is enormous (~40GB). We use HuggingFace's streaming mode to pull records over the network one by one, stopping when we have 50 valid samples. Because NQ documents contain raw HTML tags (`<H1>`, `<P>`), we use the dataset's `is_html` mask to strip out tags, leaving clean text.

**Example Row from Dataset:**

- **Query:** _"where did they film high school musical two"_
- **Full Document:** (Entire Wikipedia page for High School Musical 2)
- **Extracted Passages:** We chunk that page into 15 passages of 100 words each.
- **Human Short Answer Guarantee:** We know the answer "Utah" is hidden somewhere inside one of those 15 passages.

---

## 4. Evaluation Metrics Breakdown

The application evaluates pipeline health using three primary metrics calculated on a per-query basis.

### Metric 1: Influence Score

- **What is it?** The raw score of how much a single passage impacted the final LLM output.
- **How it works:** We calculate the ROUGE-L F1 score (longest common subsequence overlap) between the Baseline Answer and the Ablated Answer. The Influence Score is the inverse of this overlap:
  `Influence = 1.0 - ROUGE-L(Baseline, Ablated)`
- **Example:**
  - _Baseline Answer:_ "The capital of France is Paris."
  - _Ablated Answer 1:_ "The capital of France is Paris." -> ROUGE = 1.0 -> **Influence = 0.0 (Ignored)**
  - _Ablated Answer 2:_ "I don't know the capital of France." -> ROUGE = 0.2 -> **Influence = 0.8 (Highly Influential)**

### Metric 2: Dominance Ratio

- **What is it?** A measure of whether the LLM synthesized information from _many_ documents, or simply copy-pasted from a _single_ document.
- **How it works:** `Max(Influence Scores) / Sum(Influence Scores)`
- **Examples:**
  - **High Dominance (e.g., 0.90):** The sum of all influence scores is 1.0. Passage Rank 3 had an influence of 0.9, and the other 9 passages had 0.01. The LLM relied entirely on one specific passage.
  - **Low Dominance (e.g., 0.20):** Five different passages all had an influence score of 0.2. The LLM synthesized the answer by pulling pieces of context from half the retrieved database.

### Metric 3: Spearman's Rank Correlation Coefficient (ρ)

- **What is it?** A statistical measure (ranging from -1.0 to 1.0) evaluating how perfectly the Lexical Retrieval Ranks (BM25) aligned with the LLM's actual Usage Ranks (Influence Ranks).
- **How it works:** It measures monotonic relationships between two ranked variables.
- **Examples for our RAG Pipeline:**
  - **ρ = 0.95 (Perfect Harmony):** The BM25 algorithm perfectly handed the LLM the most important data first. The document BM25 ranked #1 was the LLM's #1 most used document. BM25's #2 was the LLM's #2, and so on.
  - **ρ = 0.10 (Random Noise):** The LLM's usage of the documents appears totally arbitrary compared to how the retriever sorted them. The retriever's highly-ranked top 5 were mostly ignored, and the LLM hunted for answers in the bottom 5.
  - **ρ = -0.60 (Negative Correlation):** Total pipeline failure. The documents the retriever scored the worst (Ranks 8, 9, 10) were actually the _most_ useful to the LLM.

If ρ falls below `0.7`, the Dashboard flags the query as **"Divergent,"** signaling a failure mode where the retrieval algorithm and the generation model operate on misaligned assumptions of relevance.
