-------------------------

# MAT 496: CapStone Project

## Title: News Bias Detector

## Overview

This project contains an Agent that can read a news article and check whether it is biased. The Agent will:
* Read article text from provided URL(via user Break-point).
* Write a short neutral summary.
* Identify factual statements and opinions.
* Check factual claims using a web search API.
* Look for emotional or one-sided language in the article.
* Calculate a bias score between 0 and 1.
* Predict the likely stance (for example: pro-government, anti-corporate).

The final result will be stored in a structured JSON format.
Some steps will pause for human review if the confidence is low or if the facts found online are unclear.
This will be done by the Jupyter-Notebook file "NewsBiasDetector.ipynb"

## Reason for picking up this project

I wanted to evaluate how different news networks present information and how much factual accuracy or emotional framing their articles contain. Many news websites appear neutral but may subtly
push certain perspectives. This project helps automate the detection of such bias by checking claims, tone, and stance.

This problem fits well with the topics taught, it uses:
* Prompting to instruct the LLM to summarize, classify claims, analyze tone, and compute bias.
* Structured Output (via Pydantic) to capture summaries, extracted claims, fact-check results, and final scores.
* Semantic Search to find relevant information online when verifying factual claims.
* Retrieval Augmented Generation (RAG) to combine retrieved search results with LLM reasoning for fact-checking.
* Tool Calling when the LLM interacts with Tavily (web search API).
* LangGraph (State, Nodes, Graph) to build a step-by-step pipeline that processes the article.
* Human-in-the-loop using LangGraph interrupts, asks for URL input, another time where the system pauses for approval if confidence is low.
* LangSmith for tracing, debugging, verifying execution flow, and observing intermediate states.

This project shows a practical way to analyse news using tools from the course.

## Video Summary Link
* Google Drive:
https://drive.google.com/file/d/120ZxQw4tbZSpnMM9d1TETuI_lRAzy146/view?usp=drivesdk
* YouTube:
https://youtu.be/5guZK03lcJM?si=r0X6LgefiaSfeqCj

## Plan

I plan to excecute these steps to complete my project.

- [DONE] Step 1: Create a virtual environment, install required libraries, and set up the initial LangGraph state.
- [DONE] Step 2: Build a summarization node that produces a short neutral summary.
- [DONE] Step 3: Build a node that extracts claims and separates facts from opinions in a structured format.
- [DONE] Step 4: Add a web search tool (like Tavily or SerpAPI) and send selected claims for checking.
- [DONE] Step 5: Build a fact-checking node that marks claims as supported, contradicted or unclear, along with source links.
- [DONE] Step 6: Build a language analysis node that detects loaded or emotional wording.
- [DONE] Step 7: Build a scoring node that combines facts and language cues to produce a final bias score and stance.
- [DONE] Step 8: Add human reviewing breakpoints for low-confidence or conflicting results.
- [DONE] Step 9: Connect LangSmith for tracing, debugging and small-scale evaluation.

## Working
### Workflow Steps (NewsBiasDetector.ipynb)
  1. URL Input Node (Breakpoint #1)
     * Purpose: Accepts news article URL from user
     * Process:
       * Workflow pauses at startup waiting for URL input
       * User enters URL interactively
       * Fetches article content using BeautifulSoup web scraping
       * Extracts clean text from paragraphs, removing scripts/styles/navigation
     * Output: Raw article text stored in state
  2. Summarization Node
     * Purpose: Generate neutral baseline summary
     * Process:
       * Uses structured output (Pydantic Summary model)
       * Prompts LLM to create 2-3 sentence factual summary
       * Estimates word count
     * Output: Neutral summary and word count
  3. Claims Extraction Node
     * Purpose: Identify verifiable facts vs opinions
     * Process:
       * Uses structured output (Pydantic ClaimsExtraction model)
       * LLM extracts 5-10 key statements from article
       * Classifies each as "fact" or "opinion" with confidence score
     * Output: Lists of factual claims and opinion statements
  4. Fact Checking Node
     * Purpose: Verify factual claims using web search
     * Process:
       * Selects top 3-5 factual claims for verification
       * For each claim:
         * Queries Tavily API for relevant search results
         * LLM analyzes search context to determine: supported, contradicted, or unclear
         * Assigns confidence score (0-1)
       * Flags needs_review if confidence < 0.3 or status unclear
     * Output: Fact-check results with status, evidence, sources, and confidence
  5. Language Analysis Node
     * Purpose: Detect emotional or loaded language
     * Process:
       * Uses structured output (Pydantic LanguageAnalysis model)
       * LLM identifies:
         * Emotionally charged words/phrases
         * Overall tone (neutral/positive/negative/inflammatory)
         * Language bias score (0=neutral, 1=extremely biased)
         * Example biased sentences
     * Output: Tone assessment, bias score, loaded phrases list
  6. Bias Scoring Node
     * Purpose: Calculate final bias assessment
     * Process:
       * Compiles all previous analysis (summary, fact-checks, language)
       * LLM generates structured BiasReport with:
         * Overall bias score (0-1 scale)
         * Predicted stance (e.g., "pro-government", "neutral")
         * Confidence in assessment
         * Key contributing factors
         * Reader recommendation
       * Triggers review if bias > 0.7 OR confidence < 0.5
     * Output: Final bias report with actionable recommendations
  7. Human Review Node (Breakpoint #2 - Conditional)
     * Purpose: Manual verification for edge cases
     * Trigger Conditions:
       * High bias score (> 0.7)
       * Low confidence (< 0.5)
       * Unclear fact-check results
     * Process:
       * Displays current analysis and reason for review
       * Pauses execution
       * User approves/rejects to continue
     * Output: Human approval flag
  8. Export & Display
     * Purpose: Save results and present findings
     * Process:
       * Generates formatted console report
       * Exports complete analysis to JSON file
       * Includes all intermediate outputs and final scores

### State Management
  The BiasDetectorState TypedDict tracks:
  * Inputs: article_url, article_text
  * Processing: summary, claims, fact_checks, language_analysis, bias_report
  * Control Flow: needs_human_review, review_reason, human_approved
  * Tracing: messages list for debugging

### LangGraph Studio Integration (news_bias_graph.py)
  The standalone Python file adapts the notebook for LangGraph Studio deployment:
  Key Differences:
  * No URL Input Breakpoint: Starts directly with fetch_article_node (URL provided in initial state)
  * Platform Checkpointer: Uses cloud-native persistence instead of local MemorySaver
  * Single Breakpoint: Only human_review node for conditional intervention
  * Exported Graph: graph = create_graph() enables Studio visualization

  Studio Workflow:
  * User provides URL in initial state configuration
  * Graph executes automatically through scoring node
  * Pauses at human_review only if confidence thresholds met
  * Results viewable in Studio's state inspector

## Technologies Used
- LangGraph: State management, conditional routing, breakpoints
- Groq (llama-3.3-70b): LLM for analysis with structured outputs
- Tavily API: Web search for fact verification
- Pydantic: Structured output validation
- LangSmith: Execution tracing and debugging
- BeautifulSoup: Web scraping for article extraction

## Conclusion:

  I had planned to build a complete news-analysis system that could:
  * Read and extract article text
  * Identify claims
  * Perform fact-checking using search results
  * Analyze language and tone
  * Calculate a bias score
  * Include human oversight.
  I believe I have successfully achieved these goals.

  Why the project is satisfactory:
  * Each topic mentioned been applied.
  * The system works end-to-end with real URLs.
  * The results are structured, interpretable, and easy to debug via LangSmith.
  * The human-review mechanism makes it safer and more realistic.

  What could be better:
  * If I had additional time, I would add a UI and batch-analysis support for comparing sources.
  * Add a Web Interface for easier use.
-------------------------
