import json 
import os
from datetime import datetime
import streamlit as lt
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st

def fetch_stock_price(ticker):
    stock = yf.download(ticker, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stocks prices for {ticker} from the last year about a specific company from Yahoo Finance API",
    func= lambda ticker: fetch_stock_price(ticker)
)

os.environ['OPENAI_API_KEY']= st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


stock_price_analyst = Agent(
    role= "Senior stock price analyst",
    goal="Find the {ticker} stock price and analyses trends",
    backstory="""You're highly experienced in analyzing the price of an specific stock
    and make predictions about its future price.""",
    verbose=True,
    llm= llm,
    max_iter= 5,
    memory= True,
    tools=[yahoo_finance_tool],
    allow_delegation=False
)

get_stock_price = Task(
    description= "Analyze the stock {ticker} price history and create a trend analyses of up, down or sideways",
    expected_output = """" Specify the current trend stock price - up, down or sideways. 
    eg. stock= 'APPL, price UP'
""",
    agent= stock_price_analyst
)

search_tool = DuckDuckGoSearchResults(backend='news', num_results=10)

news_analyst = Agent(
    role= "Stock News Analyst",
    goal="""Create a short summary of the market news related to the stock {ticker} company. Specify the current trend - up, down or sideways with
    the news context. For each request stock asset, specify a numbet between 0 and 100, where 0 is extreme fear and 100 is extreme greed.""",
    backstory="""You're highly experienced in analyzing the market trends and news and have tracked assest for more then 10 years.

    You're also master level analyts in the tradicional markets and have deep understanding of human psychology.

    You understand news, theirs tittles and information, but you look at those with a health dose of skepticism. 
    You consider also the source of the news articles. 
    """,
    verbose=True,
    llm= llm,
    max_iter= 10,
    memory= True,
    tools=[search_tool],
    allow_delegation=False
)

get_news = Task(
    description= f"""Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually. 

    The current date is {datetime.now()}.

    Compose the results into a helpfull report""",
    expected_output = """"A summary of the overall market and one sentence summary for each request asset. 
    Include a fear/greed score for each asset based on the news. Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>
""",
    agent= news_analyst
)

stock_report_writter = Agent(
    role = "Senior Stock Analyts Writer",
    goal= """"Analyze the trends price and news and write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend. """,
    backstory= """You're widely accepted as the best stock analyst in the market. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences. 

    You understand macro factors and combine multiple theories - eg. cycle theory and fundamental analyses. 
    You're able to hold multiple opinions when analyzing anything.
""",
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory=True,
    allow_delegation = True
)

write_report = Task(
    description = """Use the stock price trend and the stock news report to create an analyses and write the newsletter about the {ticker} company
    that is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
""",
    expected_output= """"An eloquent 3 paragraphs newsletter formated as markdown in an easy readable manner. It should contain:

    - 3 bullets executive summary 
    - Introduction - set the overall picture and spike up the interest
    - Top 5 news titles
    - main part provides the meat of the analysis including the news summary and fead/greed scores
    - summary - key facts and concrete future trend prediction - up, down or sideways.
""",
    agent = stock_report_writter,
    context = [get_stock_price, get_news]
)

spanish_translator = Agent(
    role = "Senior translator",
    goal= """"Translate reports made by Senior Stock Analyts Writer """,
    backstory= """You're widely accepted as the best translator. You understand complex concepts and create compelling stories
    and narratives that resonate with wider audiences. 

""",
    verbose = True,
    llm=llm,
    max_iter = 5,
    memory=True,
    allow_delegation = True
)

translate_report = Task(
    description = """Translate the text from English to Spanish
""",
    expected_output= """"Text generated by stockAnalystWrite translated to spanish
""",
    agent = spanish_translator,
    context = [write_report]
)

crew = Crew(
    agents = [stock_price_analyst, news_analyst, stock_report_writter, spanish_translator],
    tasks = [get_stock_price, get_news, write_report, translate_report], 
    verbose = 2,
    process= Process.hierarchical,
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)

with st.sidebar:
    st.header("Entre con el ticker")

    with st.form(key="research_form"):
        topic=st.text_input("Select the ticker")
        
        submit_button=st.form_submit_button(label="run")
if submit_button:
    if not topic:
        st.error("Seleccione un ticker")   
    else:
        results = crew.kickoff(inputs={'ticker': topic})  
        st.subheader("resultado")   
        st.write(results['final_output'])    