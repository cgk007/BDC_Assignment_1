import os
import json
import time
import argparse
import requests
from typing import List, Literal, Optional

import mlflow
from pydantic import BaseModel, Field

from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from langchain_community.utilities.brave_search import BraveSearchWrapper
from langchain_core.output_parsers import StrOutputParser

# --------------------------
# Pydantic structured schema
# --------------------------
class SentimentSchema(BaseModel):
    companyname: str = Field(..., description="Original company name")
    stockcode: str = Field(..., description="Resolved stock code/ticker")
    newsdesc: List[str] = Field(..., description="Concise bullet summaries of latest news")
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(..., description="Overall market sentiment")
    peoplenames: List[str] = Field(default_factory=list, description="People mentioned")
    placesnames: List[str] = Field(default_factory=list, description="Places mentioned")
    othercompaniesreferred: List[str] = Field(default_factory=list, description="Other companies mentioned")
    relatedindustries: List[str] = Field(default_factory=list, description="Related industries")
    marketimplications: str = Field(..., description="Short rationale of likely market impact")
    confidencescore: float = Field(..., ge=0.0, le=1.0, description="Confidence between 0.0 and 1.0")

# --------------------------
# Helpers
# --------------------------
STATIC_TICKER_MAP = {
    "Apple Inc.": "AAPL",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Microsoft Corp.": "MSFT",
    "Alphabet": "GOOGL",
    "Alphabet Inc.": "GOOGL",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "NVIDIA": "NVDA",
}

def yahoo_symbol_suggest(query: str) -> Optional[str]:
    # Unofficial suggestion endpoint; robust error handling for classroom/demo use
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": query, "quotesCount": 1, "newsCount": 0}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.ok:
            data = resp.json()
            quotes = data.get("quotes", [])
            if quotes:
                symbol = quotes[0].get("symbol")
                # Basic sanity check to avoid indices/ETFs for simple demos
                if symbol and symbol.isupper():
                    return symbol
    except Exception:
        return None
    return None

def resolve_ticker(company: str) -> str:
    # Try static first, then Yahoo suggest
    if company in STATIC_TICKER_MAP:
        return STATIC_TICKER_MAP[company]
    suggestion = yahoo_symbol_suggest(company)
    return suggestion or company.replace(" ", "").upper()[:5]

def brave_fetch_news(ticker: str, k: int = 8) -> List[dict]:
    # Use LangChain BraveSearch wrapper to get recent web results
    wrapper = BraveSearchWrapper(api_key=BRAVE_SEARCH_API_KEY)
    # Form a newsy query
    q = f"{ticker} stock company news latest headlines past week"
    results = wrapper.results(q, count=k)
    # Normalize to a simple list of dicts
    news = []
    for r in results or []:
        title = r.get("title") or ""
        url = r.get("url") or ""
        snippet = r.get("description") or r.get("snippet") or ""
        news.append({"title": title, "url": url, "summary": snippet})
    return news

def compress_news_items(items: List[dict], max_items: int = 8) -> List[str]:
    bullets = []
    for it in items[:max_items]:
        title = (it.get("title") or "").strip()
        summary = (it.get("summary") or "").strip()
        if title and summary:
            bullets.append(f"{title} — {summary}")
        elif title:
            bullets.append(title)
        elif summary:
            bullets.append(summary)
    return bullets

# --------------------------
# Build the chain
# --------------------------
def build_chain(model_name: str = "gemini-2.0-flash-001"):
    llm = ChatVertexAI(
        model_name=model_name,
        project=GCP_PROJECT or None,
        location=GCP_LOCATION or None,
        temperature=0.2,
    )

    # Structured output binding
    structured_llm = llm.with_structured_output(SentimentSchema)

    prompt = PromptTemplate(
        template=(
            "Task: Given recent news bullets for a company and its stock code, produce a concise, structured sentiment profile.\n"
            "Company: {company}\n"
            "Ticker: {ticker}\n"
            "News bullets:\n{news_bullets}\n\n"
            "Instructions:\n"
            "- Return a compact analysis, deduplicating entities and keeping bullets terse.\n"
            "- sentiment must be one of Positive, Negative, or Neutral.\n"
            "- confidencescore must be between 0.0 and 1.0.\n"
        ),
        input_variables=["company", "ticker", "news_bullets"],
    )

    # Steps: resolve ticker -> fetch news -> compress -> prompt LLM -> structured JSON
    def _resolve(inputs):
        company = inputs["company"]
        with mlflow.start_run(nested=True, run_name="resolve_ticker"):
            ticker = resolve_ticker(company)
            mlflow.log_param("company", company)
            mlflow.log_param("resolved_ticker", ticker)
        return {"company": company, "ticker": ticker}

    def _news(inputs):
        ticker = inputs["ticker"]
        with mlflow.start_run(nested=True, run_name="fetch_news"):
            news_items = brave_fetch_news(ticker)
            mlflow.log_param("news_items_count", len(news_items))
            # Keep raw news as artifact for debugging
            mlflow.log_text(json.dumps(news_items, indent=2), artifact_file="news_raw.json")
        return {"ticker": ticker, "news_items": news_items, "company": inputs["company"]}

    def _compress(inputs):
        bullets = compress_news_items(inputs["news_items"])
        with mlflow.start_run(nested=True, run_name="compress_news"):
            mlflow.log_param("compressed_count", len(bullets))
            mlflow.log_text("\n".join(f"- {b}" for b in bullets), artifact_file="news_bullets.txt")
        return {"company": inputs["company"], "ticker": inputs["ticker"], "news_bullets": "\n".join(f"- {b}" for b in bullets)}

    # Runnable graph
    graph = (
        RunnablePassthrough()
        | RunnableLambda(_resolve)
        | RunnableLambda(_news)
        | RunnableLambda(_compress)
        | {
            "company": lambda x: x["company"],
            "ticker": lambda x: x["ticker"],
            "news_bullets": lambda x: x["news_bullets"],
        }
        | prompt
        | RunnableLambda(lambda p: mlflow.log_text(p.to_string(), artifact_file="prompt.txt") or p)
        | structured_llm
    )

    return graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--company", required=True, help="Company name, e.g., 'Google' or 'Apple Inc.'")
    parser.add_argument("--model", default="gemini-2.0-flash-001")
    parser.add_argument("--experiment", default="real_time_market_sentiment")
    args = parser.parse_args()

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(args.experiment)

    chain = build_chain(model_name=args.model)

    with mlflow.start_run(run_name=f"sentiment_pipeline_{int(time.time())}"):
        mlflow.log_param("gcp_location", os.getenv("GCP_LOCATION", ""))
        mlflow.log_param("model_name", args.model)
        mlflow.log_param("search_provider", "brave")

        result = chain.invoke({"company": args.company})
        # Log final result
        mlflow.log_text(result.model_dump_json(indent=2), artifact_file="sentiment.json")

        print(json.dumps(result.model_dump(), indent=2))

if __name__ == "__main__":
    main()
