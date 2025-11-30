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
If I had additional time, I would add a UI and batch-analysis support for comparing sources.
-------------------------
