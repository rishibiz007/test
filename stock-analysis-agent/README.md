# stock-analysis-agent

This repo contains source code for a stock analysis agent. It takes a stock ticker symbol and can perform various analysis based on user request. 

Tech: LangChain, Python

1. Get 1-year stock performance
2. Compare a stock with the S&P500
3. Fetch relevant news articles about a stock
4. Conduct sentiment analysis on retrieved articles


## Set Up Stock Agent

Step 1: Install package requirements

```
pip3 install -r requirements.txt
```

Step 2: Rename config.env.tpl to config.env

```
mv config.env.tpl config.env
```

Step 3: Add your OpenAI API key
- Get your [OpenAI API key](https://platform.openai.com/api-keys)
- Add it to config.env: `OPENAI_API_KEY=your_key_here`

Step 4 (one-time): Download TextBlob's NLP corpora (needed for sentiment analysis)
```
python3 -m textblob.download_corpora
```

Step 5: Run the Stock Analysis Agent

One-step launcher (recommended) — handles `cd` + venv activation + start:
```
./run.sh
```

Or manually:
```
source .venv/bin/activate && python3 agent.py
```

You'll get a numbered menu — pick `1`-`4`, type a ticker, see the result. Type `q` to quit.

```
============================================
 Stock Analysis Agent
============================================
  1) 1-year stock performance
  2) Compare ticker to S&P 500
  3) Recent news articles
  4) News + sentiment analysis
  5) Key ratio comparable (BUY/SELL)
  q) Quit
Pick [1-5 / q]:
```

### About option 5 — Key ratio comparable

Pulls trailing P/E, PEG, and ROE for the ticker from yfinance, compares each against a sector-average benchmark, and scores each ratio bullish/neutral/bearish (+1 / 0 / -1). The summed score maps to:

| Score | Recommendation |
|---|---|
| +2 or +3 | STRONG BUY |
| +1 | BUY |
| 0 | HOLD |
| -1 | SELL |
| -2 or -3 | STRONG SELL |

Sector benchmarks are hardcoded approximations of S&P sector composites (~2024–2025), edit `SECTOR_BENCHMARKS` at the top of `agent.py` to refresh. Lower P/E and PEG vs sector are bullish; higher ROE vs sector is bullish. Not financial advice — this is a heuristic, not a model.

### Natural-language mode (LangChain agent)

If you'd rather type free-form questions and let the LLM route to the right tool:

```
python3 agent.py --ask
```

Then ask things like `Give performance of AMZN`, `Compare AMZN to S&P500`, `Fetch relevant news articles about AMZN`, `Conduct sentiment analysis about AMZN`.

`--ask` mode uses `gpt-4o-mini` and incurs OpenAI API cost per request; the default menu mode does not (it calls the tool functions directly).
