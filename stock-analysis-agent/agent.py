# Standard library imports for basic functionality
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List  # Type hints for better code clarity

# Third-party library imports
import yfinance as yf  # Yahoo Finance API for stock data
from dotenv import load_dotenv  # For loading environment variables
from langchain.agents import AgentExecutor, create_openai_functions_agent  # For creating AI agents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # For structuring AI prompts
from langchain_openai import ChatOpenAI  # OpenAI's chat model interface
from langchain.tools import Tool  # For creating tools the AI agent can use
from newspaper import Article  # For scraping and parsing news articles
from textblob import TextBlob  # For sentiment analysis of text

# Load environment variables from config.env file
# This keeps sensitive data like API keys secure
load_dotenv("config.env")

# Approximate sector-average ratios used as benchmarks for the key-ratio
# comparison tool. These are rounded values drawn from broadly cited industry
# data (Damodaran / S&P sector composites, ~2024–2025). They are not real-time
# — refresh periodically. P/E and PEG: lower-than-sector is bullish.
# ROE: higher-than-sector is bullish. ROE is stored as a decimal (0.22 = 22%).
SECTOR_BENCHMARKS = {
    "Technology":             {"pe": 28.0, "peg": 2.0, "roe": 0.22},
    "Healthcare":             {"pe": 22.0, "peg": 1.8, "roe": 0.16},
    "Financial Services":     {"pe": 14.0, "peg": 1.4, "roe": 0.12},
    "Consumer Cyclical":      {"pe": 24.0, "peg": 1.7, "roe": 0.18},
    "Consumer Defensive":     {"pe": 21.0, "peg": 2.5, "roe": 0.20},
    "Communication Services": {"pe": 20.0, "peg": 1.6, "roe": 0.18},
    "Industrials":            {"pe": 22.0, "peg": 1.9, "roe": 0.17},
    "Energy":                 {"pe": 12.0, "peg": 1.0, "roe": 0.15},
    "Utilities":              {"pe": 18.0, "peg": 3.0, "roe": 0.09},
    "Real Estate":            {"pe": 35.0, "peg": 2.8, "roe": 0.08},
    "Basic Materials":        {"pe": 18.0, "peg": 1.5, "roe": 0.13},
    # Fallback when yfinance returns a sector name we don't recognize —
    # S&P 500 broad-market averages.
    "_DEFAULT":               {"pe": 22.0, "peg": 1.8, "roe": 0.15},
}


def _score_ratio(stock_value, sector_value, lower_is_better: bool):
    """Return (verdict, score_int) for one ratio comparison.

    score_int is +1 (bullish), 0 (neutral), -1 (bearish), or None if the
    stock or sector value is missing. The 15% band around the sector
    average is treated as 'in line' (neutral).
    """
    if stock_value is None or sector_value is None or sector_value == 0:
        return ("N/A", None)
    ratio = stock_value / sector_value
    if lower_is_better:
        if ratio < 0.85:
            return ("Bullish (cheaper than sector)", 1)
        if ratio > 1.15:
            return ("Bearish (richer than sector)", -1)
        return ("Neutral (in line with sector)", 0)
    else:
        if ratio > 1.15:
            return ("Bullish (more profitable than sector)", 1)
        if ratio < 0.85:
            return ("Bearish (less profitable than sector)", -1)
        return ("Neutral (in line with sector)", 0)


def _score_to_recommendation(total: int, valid_count: int) -> str:
    """Map summed -3..+3 score into a human-readable recommendation."""
    if valid_count == 0:
        return "INSUFFICIENT DATA"
    if total >= 2:
        return "STRONG BUY"
    if total == 1:
        return "BUY"
    if total == 0:
        return "HOLD"
    if total == -1:
        return "SELL"
    return "STRONG SELL"


# Main class that handles all stock analysis functionality
class StockAnalyzer:
    def __init__(self):
        """Initialize the StockAnalyzer.

        The OpenAI key is only required for the LangChain --ask mode. Menu
        mode calls the tool methods directly with no LLM involvement, so
        we don't enforce the key here. create_stock_agent() validates it
        at the point of need instead.
        """
        # Storage used by the wrapper methods so the LangChain agent can
        # pass news articles from get_news_articles into analyze_sentiment.
        self.shared_data = {}

    def get_stock_performance(self, symbol: str) -> Dict:
        """Get 1-year performance data for a stock
        
        Args:
            symbol (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
            
        Returns:
            Dict: Dictionary containing key performance metrics or error message
        """
        try:
            # Create a Yahoo Finance ticker object for the given stock symbol
            stock = yf.Ticker(symbol)
            
            # Calculate date range for 1 year of historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Fetch historical price data from Yahoo Finance
            hist = stock.history(start=start_date, end=end_date)
            
            # Check if we got any data back
            if hist.empty:
                return {"error": f"No data found for symbol {symbol}"}
            
            # Calculate key metrics
            initial_price = hist.iloc[0]['Close']  # First day's closing price
            current_price = hist.iloc[-1]['Close']  # Most recent closing price
            percent_change = ((current_price - initial_price) / initial_price) * 100
            
            # Return comprehensive performance data
            return {
                "symbol": symbol,
                "initial_price": round(initial_price, 2),
                "current_price": round(current_price, 2),
                "percent_change": round(percent_change, 2),
                "high_52week": round(hist['High'].max(), 2),  # Highest price in last year
                "low_52week": round(hist['Low'].min(), 2)     # Lowest price in last year
            }
        except Exception as e:
            return {"error": f"Error fetching stock data: {str(e)}"}

    def compare_with_sp500(self, symbol: str) -> Dict:
        """Compare stock performance with S&P 500 index
        
        Args:
            symbol (str): Stock ticker symbol to compare
            
        Returns:
            Dict: Comparison metrics between stock and S&P 500
        """
        try:
            # Get data for both the stock and S&P 500 (^GSPC is the ticker for S&P 500)
            stock = yf.Ticker(symbol)
            sp500 = yf.Ticker("^GSPC")
            
            # Set up date range for 1-year comparison
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Get historical data for both
            stock_hist = stock.history(start=start_date, end=end_date)
            sp500_hist = sp500.history(start=start_date, end=end_date)
            
            # Calculate percentage changes for both stock and S&P 500
            stock_change = ((stock_hist.iloc[-1]['Close'] - stock_hist.iloc[0]['Close']) / 
                          stock_hist.iloc[0]['Close']) * 100
            sp500_change = ((sp500_hist.iloc[-1]['Close'] - sp500_hist.iloc[0]['Close']) / 
                           sp500_hist.iloc[0]['Close']) * 100
            
            # Return comparison results
            return {
                "stock_symbol": symbol,
                "stock_performance": round(stock_change, 2),  # Stock's percentage change
                "sp500_performance": round(sp500_change, 2),  # S&P 500's percentage change
                "difference": round(stock_change - sp500_change, 2)  # How much stock outperformed/underperformed
            }
        except Exception as e:
            return {"error": f"Error comparing with S&P 500: {str(e)}"}

    def get_news_articles(self, symbol: str) -> List[Dict]:
        """Get recent news articles about the stock
        
        Args:
            symbol (str): Stock ticker symbol to fetch news for
            
        Returns:
            List[Dict]: List of dictionaries containing article data (title, text, URL)
        """
        try:
            print(f"\n=== Fetching news for {symbol} ===")
            # Get stock news from Yahoo Finance API
            stock = yf.Ticker(symbol)
            news = stock.news
            
            if not news:
                print("No news found")
                return [{"error": f"No news articles found for {symbol}"}]
                
            print(f"Found {len(news)} total news articles for {symbol}")
            
            articles = []
            # Process only the 5 most recent articles to avoid overwhelming the system
            for i, item in enumerate(news[:5]):
                print(f"\nProcessing article {i+1}/5:")
                try:
                    # Extract the article URL from the news item
                    article_url = item['content']['clickThroughUrl']['url']
                    print(f"Extracted URL: {article_url}")
                    
                    if not article_url:
                        print(f"No URL found in news item {i+1}")
                        continue
                    
                    # Download and parse the article content using newspaper3k library
                    article = Article(article_url)
                    article.download()
                    article.parse()
                    
                    # Skip articles with no text content
                    if not article.text:
                        print(f"No text content found in article {i+1}")
                        continue
                    
                    # Store article data in a structured format
                    article_data = {
                        "title": article.title,
                        "text": article.text,
                        "url": article_url
                    }
                    articles.append(article_data)
                    print(f"Successfully processed article {i+1}: {article.title}")
                    print(f"Article length: {len(article.text)} characters")
                    
                except Exception as article_error:
                    print(f"Error processing article {i+1}:")
                    print(f"Error details: {str(article_error)}")
                    print("Full error:", article_error.__class__.__name__)
                    continue
                    
            print(f"\nSuccessfully processed {len(articles)} articles")
            
            if not articles:
                print("No articles were successfully processed")
                return [{"error": f"Could not process any news articles for {symbol}"}]
                
            return articles
            
        except Exception as e:
            print(f"\nError in get_news_articles:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {str(e)}")
            return [{"error": f"Error fetching news: {str(e)}"}]

    def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Perform sentiment analysis on news articles
        
        Args:
            articles (List[Dict]): List of article dictionaries containing text to analyze
            
        Returns:
            Dict: Sentiment analysis results including average sentiment and label
        """
        try:
            print(f"\nAnalyzing sentiment for {len(articles)} articles")
            if not articles:
                return {"error": "No articles provided for analysis"}
                
            sentiments = []

            # Process each article for sentiment
            for i, article in enumerate(articles):
                try:
                    print(f"\nAnalyzing article {i+1}/{len(articles)}:")
                    
                    # Ensure article data is in correct format
                    if not isinstance(article, dict):
                        print(f"Warning: Article {i+1} is not a dictionary")
                        continue
                        
                    title = article.get('title', 'No title')
                    text = article.get('text', '')
                    
                    print(f"Title: {title}")
                    print(f"Text length: {len(text)} characters")
                    
                    if not text:
                        print(f"No text content found for article {i+1}")
                        continue
                    
                    # Use TextBlob to analyze sentiment
                    # Sentiment polarity ranges from -1 (negative) to 1 (positive)
                    analysis = TextBlob(text)
                    sentiment = analysis.sentiment.polarity
                    print(f"Sentiment score: {sentiment}")
                    sentiments.append(sentiment)
                    
                except Exception as article_error:
                    print(f"Error analyzing article {i+1}: {str(article_error)}")
                    continue
            
            if not sentiments:
                return {"error": "Could not analyze sentiment for any articles"}
                
            # Calculate average sentiment and determine overall label
            avg_sentiment = sum(sentiments) / len(sentiments)
            result = {
                "average_sentiment": round(avg_sentiment, 2),
                "sentiment_label": "Positive" if avg_sentiment > 0.1 
                               else "Negative" if avg_sentiment < -0.1 
                               else "Neutral",
                "number_of_articles": len(sentiments)
            }
            print(f"\nFinal sentiment analysis: {result}")
            return result
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {"error": f"Error in sentiment analysis: {str(e)}"}

    def get_key_ratios(self, symbol: str) -> Dict:
        """Compare a stock's P/E, PEG, and ROE against its sector average and
        return a buy/sell recommendation.

        Pulls trailingPE, pegRatio, and returnOnEquity from yfinance, looks
        up the sector benchmark from SECTOR_BENCHMARKS, scores each ratio
        (+1/0/-1), sums the score, and maps it to one of:
        STRONG BUY / BUY / HOLD / SELL / STRONG SELL.
        """
        try:
            info = yf.Ticker(symbol).info
            if not info or info.get("trailingPE") is None and info.get("sector") is None:
                return {"error": f"No financials found for symbol {symbol}"}

            sector = info.get("sector") or "Unknown"
            stock_pe = info.get("trailingPE")
            stock_peg = info.get("pegRatio") or info.get("trailingPegRatio")
            stock_roe = info.get("returnOnEquity")

            bench = SECTOR_BENCHMARKS.get(sector, SECTOR_BENCHMARKS["_DEFAULT"])
            sector_pe, sector_peg, sector_roe = bench["pe"], bench["peg"], bench["roe"]

            pe_verdict, pe_score = _score_ratio(stock_pe, sector_pe, lower_is_better=True)
            peg_verdict, peg_score = _score_ratio(stock_peg, sector_peg, lower_is_better=True)
            roe_verdict, roe_score = _score_ratio(stock_roe, sector_roe, lower_is_better=False)

            valid_scores = [s for s in (pe_score, peg_score, roe_score) if s is not None]
            total = sum(valid_scores)
            recommendation = _score_to_recommendation(total, len(valid_scores))

            return {
                "symbol": symbol,
                "sector": sector,
                "sector_benchmark_source": "Hardcoded sector composite (~2024–2025)",
                "stock_pe": round(stock_pe, 2) if stock_pe is not None else None,
                "sector_pe": sector_pe,
                "pe_verdict": pe_verdict,
                "stock_peg": round(stock_peg, 2) if stock_peg is not None else None,
                "sector_peg": sector_peg,
                "peg_verdict": peg_verdict,
                "stock_roe_pct": round(stock_roe * 100, 2) if stock_roe is not None else None,
                "sector_roe_pct": round(sector_roe * 100, 2),
                "roe_verdict": roe_verdict,
                "score": total,
                "metrics_scored": f"{len(valid_scores)}/3",
                "recommendation": recommendation,
            }
        except Exception as e:
            return {"error": f"Error fetching key ratios: {str(e)}"}

    def get_news_articles_wrapper(self, symbol: str):
        """Wrapper method for get_news_articles that stores results in shared data
        
        This wrapper allows the agent to maintain state between calls by storing
        the articles in shared_data for later use by analyze_sentiment
        """
        articles = self.get_news_articles(symbol)
        self.shared_data['articles'] = articles
        return f"Retrieved articles for {symbol}."

    def analyze_sentiment_wrapper(self, _):
        """Wrapper method for analyze_sentiment that uses stored articles
        
        This wrapper retrieves articles from shared_data that were stored by
        get_news_articles_wrapper, allowing for sequential processing
        """
        articles = self.shared_data.get('articles', [])
        if not articles:
            return "No articles found. Please run get_news_articles first."
        return self.analyze_sentiment(articles)

def create_stock_agent():
    """Create a LangChain agent for stock analysis
    
    This function sets up an AI agent that can understand natural language requests
    and use the appropriate tools to analyze stocks. It combines OpenAI's language
    model with custom tools for stock analysis.
    
    Returns:
        AgentExecutor: An agent that can process natural language requests about stocks
    """
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found — required for --ask mode. Set it in config.env.")

    # Initialize the analyzer that contains all our stock analysis methods
    analyzer = StockAnalyzer()
    
    # Create the system prompt that defines the agent's behavior and capabilities
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a stock analysis assistant that helps analyze stocks.
        You have access to these independent tools:
        - Get stock performance data
        - Compare with S&P 500
        - Get news articles
        - Analyze sentiment of articles
        - Get key ratio comparable (P/E, PEG, ROE vs sector → buy/sell recommendation)

        IMPORTANT: Only use tools that are specifically relevant to the user's request.
        For example:
        - If user asks for news, only use get_news_articles
        - If user asks for sentiment, use get_news_articles followed by analyze_sentiment
        - If user asks for a buy/sell recommendation, valuation, or fundamentals, use get_key_ratios

        DO NOT use tools that weren't specifically requested."""),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Initialize the language model (gpt-4o-mini is cheaper and stronger than gpt-3.5-turbo)
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # Define the tools available to the agent with specific descriptions
    tools = [
        Tool(
            name="get_stock_performance",
            func=analyzer.get_stock_performance,
            description="Get 1-year performance data for a stock symbol. Use only when performance data is requested."
        ),
        Tool(
            name="compare_with_sp500",
            func=analyzer.compare_with_sp500,
            description="Compare stock's performance with S&P 500. Use only when market comparison is requested."
        ),
        Tool(
            name="get_news_articles",
            func=analyzer.get_news_articles_wrapper,
            description="Get recent news articles about a stock. Use when news or articles are requested, or before sentiment analysis."
        ),
        Tool(
            name="analyze_sentiment",
            func=analyzer.analyze_sentiment_wrapper,
            description="Analyze sentiment of news articles. Use only when sentiment analysis is specifically requested."
        ),
        Tool(
            name="get_key_ratios",
            func=analyzer.get_key_ratios,
            description="Compare stock's P/E, PEG, and ROE against its sector average and produce a buy/sell recommendation. Use when the user asks for fundamentals, valuation, key ratios, or a buy/sell decision."
        )
    ]
    
    # Create the agent with the language model and tools
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor that will run the agent with the tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

MENU_OPTIONS = {
    "1": ("1-year stock performance",          "get_stock_performance"),
    "2": ("Compare ticker to S&P 500",         "compare_with_sp500"),
    "3": ("Recent news articles",              "get_news_articles"),
    "4": ("News + sentiment analysis",         "_news_and_sentiment"),
    "5": ("Key ratio comparable (BUY/SELL)",   "get_key_ratios"),
}


def _print_menu() -> None:
    print()
    print("=" * 44)
    print(" Stock Analysis Agent")
    print("=" * 44)
    for key, (label, _) in MENU_OPTIONS.items():
        print(f"  {key}) {label}")
    print("  q) Quit")


def _print_result(result) -> None:
    """Pretty-print whatever a tool returned (dict, list of articles, etc.)."""
    if isinstance(result, list):
        if result and isinstance(result[0], dict) and result[0].get("error"):
            print(f"  Error: {result[0]['error']}")
            return
        for i, item in enumerate(result, 1):
            print(f"  {i}. {item.get('title', '(no title)')}")
            print(f"     {item.get('url', '')}")
    elif isinstance(result, dict):
        if "error" in result:
            print(f"  Error: {result['error']}")
            return
        # Key-ratio dict gets a structured layout with the recommendation
        # called out at the bottom.
        if "recommendation" in result:
            _print_key_ratios(result)
            return
        for k, v in result.items():
            print(f"  {k:>20}: {v}")
    else:
        print(f"  {result}")


def _print_key_ratios(r: Dict) -> None:
    """Custom layout for the get_key_ratios output."""
    print(f"  Symbol: {r['symbol']}    Sector: {r['sector']}")
    print(f"  ({r['sector_benchmark_source']})")
    print()
    print(f"  {'Metric':<8} {'Stock':>10} {'Sector':>10}   Verdict")
    print(f"  {'-'*8} {'-'*10} {'-'*10}   {'-'*40}")
    rows = [
        ("P/E",  r["stock_pe"],       r["sector_pe"],       r["pe_verdict"]),
        ("PEG",  r["stock_peg"],      r["sector_peg"],      r["peg_verdict"]),
        ("ROE%", r["stock_roe_pct"],  r["sector_roe_pct"],  r["roe_verdict"]),
    ]
    for name, stock_v, sector_v, verdict in rows:
        stock_s = f"{stock_v:.2f}" if isinstance(stock_v, (int, float)) else "N/A"
        sector_s = f"{sector_v:.2f}"
        print(f"  {name:<8} {stock_s:>10} {sector_s:>10}   {verdict}")
    print()
    print(f"  Score: {r['score']:+d}  (across {r['metrics_scored']} metrics)")
    print(f"  >>> RECOMMENDATION: {r['recommendation']}")


def run_menu_mode() -> None:
    """Default mode: numbered menu, no LLM involved."""
    analyzer = StockAnalyzer()
    while True:
        _print_menu()
        choice = input("Pick [1-5 / q]: ").strip().lower()
        if choice == "q":
            print("Bye.")
            return
        if choice not in MENU_OPTIONS:
            print("Invalid choice — pick 1, 2, 3, 4, 5, or q.")
            continue

        ticker = input("Ticker symbol (e.g. AAPL): ").strip().upper()
        if not ticker:
            print("Skipped — empty ticker.")
            continue

        label, method_name = MENU_OPTIONS[choice]
        print(f"\n>>> {label} for {ticker}")
        print("-" * 44)
        if method_name == "_news_and_sentiment":
            articles = analyzer.get_news_articles(ticker)
            if articles and isinstance(articles[0], dict) and articles[0].get("error"):
                _print_result(articles)
            else:
                _print_result(analyzer.analyze_sentiment(articles))
        else:
            _print_result(getattr(analyzer, method_name)(ticker))


def run_ask_mode() -> None:
    """--ask mode: original LangChain natural-language agent loop."""
    agent = create_stock_agent()
    print("LangChain agent mode — ask any question about a stock. Type 'quit' to exit.")
    while True:
        user_request = input("\n> ").strip()
        if user_request.lower() == "quit":
            return
        try:
            result = agent.invoke({"input": user_request})
            print("\nAnalysis Results:")
            print(result["output"])
        except Exception as e:
            print(f"\nError: {str(e)}")


def main():
    """Entry point — menu by default, LangChain agent with --ask."""
    if "--ask" in sys.argv:
        run_ask_mode()
    else:
        run_menu_mode()


if __name__ == "__main__":
    main()