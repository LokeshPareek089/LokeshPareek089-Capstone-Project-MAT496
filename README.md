Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report Template

## Title: News Bias Detector

## Overview

This project builds an AI system that can read a news article and check whether it is biased. The system will:
* Read article text or a provided URL.
* Write a short neutral summary.
* Identify factual statements and opinions.
* Check factual claims using a web search API.
* Look for emotional or one-sided language in the article.
* Calculate a bias score between 0 and 1.
* Predict the likely stance (for example: pro-government, anti-corporate).

The final result will be stored in a structured JSON format.
Some steps will pause for human review if the confidence is low or if the facts found online are unclear.

## Reason for picking up this project

I want to evaluate articles from different News-Networks, to check their credibiltiy and confirm their stance.

This problem fits well with the topics taught in MAT496. It uses:
* Prompting to instruct the AI to summarize, extract claims and score bias.
* Structured Output to store summaries, claims and scores in JSON.
* Semantic Search to check claims using online information.
* Retrieval Augmented Generation (RAG) to combine AI reasoning with web search results.
* Tool Calling to run a web search API inside the LangGraph workflow.
* LangGraph (State, Nodes, Graph) to build a step-by-step agent to analyse the article.
* LangSmith to trace runs, debug mistakes, and evaluate results.

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

I had planned to achieve {this this}. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.

----------

# Added instructions:

- This is a `solo assignment`. Each of you will work alone. You are free to talk, discuss with chatgpt, but you are responsible for what you submit. Some students may be called for viva. You should be able to each and every line of work submitted by you.

- `commit` History maintenance.
  - Fork this respository and build on top of that.
  - For every step in your plan, there has to be a commit.
  - Change [TODO] to [DONE] in the plan, before you commit after that step. 
  - The commit history should show decent amount of work spread into minimum two dates. 
  - **All the commits done in one day will be rejected**. Even if you are capable of doing the whole thing in one day, refine it in two days.  
 
 - Deadline: Nov 30, Sunday 11:59 pm


# Grading: total 25 marks

- Coverage of most of topics in this class: 20
- Creativity: 5
  
