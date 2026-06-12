# Standard library imports for basic functionality
import os
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

# Main class that handles all stock analysis functionality
class StockAnalyzer:
    def __init__(self):
        """Initialize the StockAnalyzer with necessary API keys"""
        # Get OpenAI API key from environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # Instance-specific shared data storage
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

        IMPORTANT: Only use tools that are specifically relevant to the user's request.
        For example:
        - If user asks for news, only use get_news_articles
        - If user asks for sentiment, use get_news_articles followed by analyze_sentiment
        
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
        )
    ]
    
    # Create the agent with the language model and tools
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor that will run the agent with the tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def main():
    """Main function to run the stock analysis program
    
    This function creates the agent and runs an interactive loop where users
    can input requests about stocks and get AI-powered analysis responses.
    """
    # Create the AI agent that will handle stock analysis requests
    agent = create_stock_agent()

    while True:
        user_request = input("\nEnter a stock symbol and the type of analysis you want to perform: ").strip().upper()
        if user_request.lower() == 'quit':
            break
            
        try:
            # Process the user's request using the AI agent
            result = agent.invoke({
                "input": user_request
            })
            print("\nAnalysis Results:")
            print(result["output"])
        except Exception as e:
            print(f"\nError: {str(e)}")
    
if __name__ == "__main__":
    main()