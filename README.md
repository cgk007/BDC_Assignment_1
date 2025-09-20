# BDC_Assignment_1
This repo contains the code for Assignment 1 (Real-Time Market Sentiment  Analyzer Using LangChain Chains)

# Real-Time Market Sentiment Analyzer (LangChain + Gemini + mlflow)

## Prerequisites
- Python 3.10, GCP project with Vertex AI enabled, and service-account credentials with access to Gemini 2.0 Flash [gemini-2.0-flash-001] [file:21][web:37].
- Brave Search API key (or switch to Exa if preferred) and an mlflow tracking backend/URI for observability as required by the assignment [file:21][web:22].

## Install
pip install -r requirements.txt

## Configure
# Option 1: export environment variables
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
export GCP_PROJECT=your-project-id
export GCP_LOCATION=us-central1
export BRAVE_SEARCH_API_KEY=your-brave-key
export MLFLOW_TRACKING_URI=http://localhost:5000

# Option 2: copy .env.example to .env and use python-dotenv
cp .env.example .env

## Run
python market_sentiment_analyzer.py --company "Apple Inc." --model "gemini-2.0-flash-001"

## Outputs
- Stdout: structured JSON sentiment profile with fields required by the brief [file:21].
- mlflow: nested runs for resolve_ticker, fetch_news, compress_news, final sentiment.json, and prompt.txt artifacts [file:21].

