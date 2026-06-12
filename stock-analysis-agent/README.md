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

Step 4: Run the Stock Analysis Agent
```
python3 agent.py
```

## Example Questions

1. Give performance of AMZN
2. Compare AMZN to S&P500
3. Fetch relevant news articles about AMZN
4. Conduct sentiment analysis about AMZN
