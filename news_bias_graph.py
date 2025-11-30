import os
from typing import TypedDict, List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
# NOTE: Do NOT import MemorySaver unconditionally; we will import it only when explicitly requested.
from tavily import TavilyClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

# Initialize clients
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Summary(BaseModel):
    """Neutral summary of the article"""
    summary: str = Field(description="A short, neutral summary of the article (2-3 sentences)")
    word_count: int = Field(description="Approximate word count of original article")

class Claim(BaseModel):
    """Individual claim or statement"""
    text: str = Field(description="The claim or statement")
    type: str = Field(description="Either 'fact' or 'opinion'")
    confidence: float = Field(description="Confidence in classification (0-1)")

class ClaimsExtraction(BaseModel):
    """Extracted claims from article"""
    factual_claims: List[Claim] = Field(description="List of factual claims")
    opinions: List[Claim] = Field(description="List of opinions")
    total_claims: int = Field(description="Total number of claims extracted")

class FactCheck(BaseModel):
    """Fact check result for a claim"""
    claim: str = Field(description="The original claim")
    status: str = Field(description="Either 'supported', 'contradicted', or 'unclear'")
    evidence: str = Field(description="Summary of evidence found")
    sources: List[str] = Field(description="URLs of sources")
    confidence: float = Field(description="Confidence in fact check (0-1)")

class FactCheckResults(BaseModel):
    """All fact check results"""
    checks: List[FactCheck] = Field(description="List of fact check results")
    needs_review: bool = Field(description="Whether human review is needed")

class LanguageAnalysis(BaseModel):
    """Analysis of language bias"""
    loaded_phrases: List[str] = Field(description="Emotionally loaded or biased phrases")
    tone: str = Field(description="Overall tone: neutral, positive, negative, inflammatory")
    language_bias_score: float = Field(description="Language bias score (0-1, 0=neutral)")
    examples: List[str] = Field(description="Example sentences showing bias")

class BiasReport(BaseModel):
    """Final bias analysis report"""
    bias_score: float = Field(description="Overall bias score (0-1, 0=unbiased)")
    stance: str = Field(description="Predicted stance or position")
    confidence: float = Field(description="Confidence in assessment (0-1)")
    key_factors: List[str] = Field(description="Key factors contributing to bias score")
    recommendation: str = Field(description="Recommendation for readers")

# ============================================================================
# STATE DEFINITION
# ============================================================================

class BiasDetectorState(TypedDict):
    """State for the bias detector workflow"""
    # Input
    article_url: str
    article_text: str
    
    # Processing stages
    summary: Optional[Summary]
    claims: Optional[ClaimsExtraction]
    fact_checks: Optional[FactCheckResults]
    language_analysis: Optional[LanguageAnalysis]
    bias_report: Optional[BiasReport]
    
    # Control flow
    needs_human_review: bool
    review_reason: Optional[str]
    human_approved: bool
    
    # Messages for tracing
    messages: List[str]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def fetch_article_from_url(url: str) -> str:
    """Fetch article text from a URL using web scraping."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Get text from paragraphs
        paragraphs = soup.find_all('p')
        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        if len(article_text) < 100:
            # Fallback to all text
            article_text = soup.get_text(separator='\n', strip=True)
        
        return article_text
    
    except Exception as e:
        return f"Error fetching article: {str(e)}"

# ============================================================================
# GRAPH NODES
# ============================================================================

def fetch_article_node(state: BiasDetectorState) -> BiasDetectorState:
    """Fetch article from URL"""
    print(f"\n📥 Fetching article from: {state['article_url']}")
    
    article_text = fetch_article_from_url(state['article_url'])
    
    if article_text.startswith("Error"):
        print(f"❌ {article_text}")
        state['article_text'] = ""
        state['messages'].append(f"Failed to fetch URL: {state['article_url']}")
    else:
        state['article_text'] = article_text
        state['messages'].append(f"Fetched {len(article_text)} characters from URL")
        print(f"✓ Fetched {len(article_text)} characters")
    
    return state

def summarize_node(state: BiasDetectorState) -> BiasDetectorState:
    """Generate a neutral summary of the article."""
    print("\n🔄 Running: Summarization Node")
    
    structured_llm = llm.with_structured_output(Summary)
    
    prompt = f"""You are a neutral news analyst. Read the following article and provide a short, neutral summary.
Focus on the main facts and events without adding interpretation or opinion.

Article:
{state['article_text'][:3000]}

Provide a 2-3 sentence neutral summary and estimate the word count."""

    summary = structured_llm.invoke([HumanMessage(content=prompt)])
    
    state['summary'] = summary
    state['messages'].append(f"Summary created: {len(summary.summary)} chars")
    
    print(f"✓ Summary created")
    
    return state

def extract_claims_node(state: BiasDetectorState) -> BiasDetectorState:
    """Extract factual claims and opinions from the article."""
    print("\n🔄 Running: Claims Extraction Node")
    
    structured_llm = llm.with_structured_output(ClaimsExtraction)
    
    prompt = f"""You are a critical analyst. Read this article and extract key claims.

Separate them into:
1. FACTUAL CLAIMS: Statements that can be verified (dates, events, statistics, quotes)
2. OPINIONS: Judgments, interpretations, predictions, or subjective statements

Article:
{state['article_text'][:4000]}

For each claim, provide:
- The exact text of the claim
- Type: 'fact' or 'opinion'
- Confidence: How confident you are in the classification (0-1)

Extract 5-10 of the most important claims."""

    claims = structured_llm.invoke([HumanMessage(content=prompt)])
    
    state['claims'] = claims
    state['messages'].append(f"Extracted {claims.total_claims} claims")
    
    print(f"✓ Extracted {len(claims.factual_claims)} factual claims, {len(claims.opinions)} opinions")
    
    return state

def fact_check_node(state: BiasDetectorState) -> BiasDetectorState:
    """Fact-check claims using Tavily web search."""
    print("\n🔄 Running: Fact Checking Node")
    
    claims = state['claims']
    fact_checks = []
    needs_review = False
    
    # Select top 3-5 most important factual claims
    factual_claims = claims.factual_claims[:5]
    
    for claim_obj in factual_claims:
        claim_text = claim_obj.text
        print(f"  🔍 Checking: {claim_text[:60]}...")
        
        try:
            # Search for information about the claim
            search_results = tavily_client.search(
                query=claim_text,
                max_results=3
            )
            
            # Analyze results with LLM
            context = "\n\n".join([
                f"Source: {r.get('url', 'Unknown')}\n{r.get('content', '')}" 
                for r in search_results.get('results', [])
            ])
            
            analysis_prompt = f"""Based on the following web search results, determine if this claim is supported, contradicted, or unclear.

Claim: {claim_text}

Search Results:
{context[:2000]}

Determine:
1. Status: 'supported', 'contradicted', or 'unclear'
2. Brief evidence summary
3. Confidence (0-1)

Be conservative: if evidence is mixed or insufficient, mark as 'unclear'."""

            response = llm.invoke([HumanMessage(content=analysis_prompt)])
            
            # Parse response
            status = "unclear"
            if "supported" in response.content.lower():
                status = "supported"
            elif "contradicted" in response.content.lower():
                status = "contradicted"
            
            # Extract confidence
            confidence = 0.5
            if "high confidence" in response.content.lower() or "clearly" in response.content.lower():
                confidence = 0.8
            elif "unclear" in status or "insufficient" in response.content.lower():
                confidence = 0.3
                needs_review = True
            
            fact_check = FactCheck(
                claim=claim_text,
                status=status,
                evidence=response.content[:300],
                sources=[r.get('url', '') for r in search_results.get('results', [])[:3]],
                confidence=confidence
            )
            
            fact_checks.append(fact_check)
            print(f"    ✓ Status: {status}")
            
        except Exception as e:
            print(f"    ✗ Error: {str(e)}")
            fact_checks.append(FactCheck(
                claim=claim_text,
                status="unclear",
                evidence=f"Error during fact check: {str(e)}",
                sources=[],
                confidence=0.0
            ))
            needs_review = True
    
    state['fact_checks'] = FactCheckResults(
        checks=fact_checks,
        needs_review=needs_review
    )
    state['needs_human_review'] = needs_review
    if needs_review:
        state['review_reason'] = "Low confidence in fact checking results"
    
    state['messages'].append(f"Fact-checked {len(fact_checks)} claims")
    
    return state

def language_analysis_node(state: BiasDetectorState) -> BiasDetectorState:
    """Analyze language for emotional or loaded wording."""
    print("\n🔄 Running: Language Analysis Node")
    
    structured_llm = llm.with_structured_output(LanguageAnalysis)
    
    prompt = f"""You are a linguistic analyst. Analyze this article for biased or emotionally loaded language.

Look for:
- Emotionally charged words (e.g., "catastrophic", "hero", "villain")
- Loaded adjectives and adverbs
- One-sided framing
- Inflammatory rhetoric
- Persuasive language

Article excerpt:
{state['article_text'][:4000]}

Provide:
1. List of loaded phrases (with context)
2. Overall tone: neutral, positive, negative, or inflammatory
3. Language bias score (0 = perfectly neutral, 1 = extremely biased)
4. Example sentences showing bias"""

    analysis = structured_llm.invoke([HumanMessage(content=prompt)])
    
    state['language_analysis'] = analysis
    state['messages'].append(f"Language analysis: tone={analysis.tone}")
    
    print(f"✓ Tone: {analysis.tone}, Bias: {analysis.language_bias_score:.2f}")
    
    return state

def bias_scoring_node(state: BiasDetectorState) -> BiasDetectorState:
    """Calculate final bias score and predict stance."""
    print("\n🔄 Running: Bias Scoring Node")
    
    structured_llm = llm.with_structured_output(BiasReport)
    
    # Compile all analysis
    summary_text = state['summary'].summary if state['summary'] else "No summary"
    
    fact_summary = "\n".join([
        f"- {fc.claim}: {fc.status} (conf: {fc.confidence:.2f})"
        for fc in state['fact_checks'].checks
    ]) if state['fact_checks'] else "No fact checks"
    
    lang_summary = f"Tone: {state['language_analysis'].tone}, Score: {state['language_analysis'].language_bias_score}" if state['language_analysis'] else "No analysis"
    
    prompt = f"""You are a media bias expert. Based on all the analysis, provide a final bias assessment.

SUMMARY:
{summary_text}

FACT CHECK RESULTS:
{fact_summary}

LANGUAGE ANALYSIS:
{lang_summary}
Loaded phrases: {', '.join(state['language_analysis'].loaded_phrases[:5]) if state['language_analysis'] else 'None'}

Calculate:
1. Overall bias score (0-1):
   - 0.0-0.2: Minimal bias
   - 0.2-0.4: Low bias
   - 0.4-0.6: Moderate bias
   - 0.6-0.8: High bias
   - 0.8-1.0: Extreme bias

2. Predicted stance (e.g., "pro-government", "anti-corporate", "left-leaning", "right-leaning", "neutral")

3. Confidence in your assessment (0-1)

4. Key factors contributing to the score

5. Recommendation for readers

Consider:
- Contradicted facts increase bias
- Emotional language increases bias
- One-sided coverage increases bias
- Missing context increases bias"""

    report = structured_llm.invoke([HumanMessage(content=prompt)])
    
    state['bias_report'] = report
    state['messages'].append(f"Final bias score: {report.bias_score:.2f}")
    
    # Check if we need review due to high bias or low confidence
    if report.bias_score > 0.7 or report.confidence < 0.5:
        state['needs_human_review'] = True
        state['review_reason'] = f"High bias score or low confidence"
    
    print(f"✓ Bias Score: {report.bias_score:.2f}, Stance: {report.stance}")
    
    return state

def human_review_node(state: BiasDetectorState) -> BiasDetectorState:
    """Pause for human review if needed."""
    print("\n⚠️  HUMAN REVIEW NODE REACHED")
    print(f"Reason: {state.get('review_reason', 'Unknown')}")
    
    return state

# ============================================================================
# ROUTING FUNCTION
# ============================================================================

def should_review(state: BiasDetectorState) -> str:
    """Decide whether to route to human review or final output."""
    if state.get('needs_human_review', False) and not state.get('human_approved', False):
        return "review"
    return "end"

# ============================================================================
# BUILD GRAPH
# ============================================================================

def create_graph():
    """Create and return the compiled graph"""
    workflow = StateGraph(BiasDetectorState)
    
    # Add nodes
    workflow.add_node("fetch_article", fetch_article_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("extract_claims", extract_claims_node)
    workflow.add_node("fact_check", fact_check_node)
    workflow.add_node("language_analysis", language_analysis_node)
    workflow.add_node("bias_scoring", bias_scoring_node)
    workflow.add_node("human_review", human_review_node)
    
    # Define edges
    workflow.set_entry_point("fetch_article")
    workflow.add_edge("fetch_article", "summarize")
    workflow.add_edge("summarize", "extract_claims")
    workflow.add_edge("extract_claims", "fact_check")
    workflow.add_edge("fact_check", "language_analysis")
    workflow.add_edge("language_analysis", "bias_scoring")
    
    # Conditional edge for human review
    workflow.add_conditional_edges(
        "bias_scoring",
        should_review,
        {
            "review": "human_review",
            "end": END
        }
    )
    
    workflow.add_edge("human_review", END)
    
    # Compile the workflow.
    # IMPORTANT: Do not pass a custom checkpointer by default — the platform handles persistence.
    # If you want a local in-memory checkpointer for dev only, set USE_LOCAL_CHECKPOINTER=1 in your env.
    use_local = os.getenv("USE_LOCAL_CHECKPOINTER", "").lower() in ("1", "true", "yes")
    if use_local:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            memory = MemorySaver()
            return workflow.compile(
                checkpointer=memory,
                interrupt_before=["human_review"]
            )
        except Exception as e:
            print("Warning: failed to create local MemorySaver, compiling without custom checkpointer:", e)
            return workflow.compile(interrupt_before=["human_review"])
    else:
        # No custom checkpointer passed — let the platform handle persistence
        return workflow.compile(
            interrupt_before=["human_review"]
        )

# Export the graph for LangGraph Studio
graph = create_graph()
