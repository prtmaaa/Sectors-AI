import os
import json
import logging
import requests
import streamlit as st
from dotenv import load_dotenv
from typing import Dict, Optional
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.cache import InMemoryCache
from langchain.memory import ConversationBufferMemory
from datetime import datetime, timedelta
import altair as alt
import pandas as pd

# Load environment variables
# load_dotenv()

# Set up API keys and headers
SECTORS_API_KEY = st.secrets["SECTORS_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]

headers = {"Authorization": SECTORS_API_KEY}

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize in-memory cache
llm_cache = InMemoryCache()

# # Initialize memory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

stock_not_valid="Stock symbol must be 4 characters, e.g 'BBRI'"

# Valid parameters
valid_indices = [
        'ftse', 'idx30', 'idxbumn20', 'idxesgl', 'idxg30', 'idxhidiv20',
        'idxq30', 'idxv30', 'jii70', 'kompas100', 'lq45', 'sminfra18', 'srikehati'
    ]

valid_sections = ["overview", "valuation", "future", "peers", "financials", "dividend",
                            "management", "ownership"]

valid_sub_sections = ["companies", "growth", "market cap", "stability", "statistics", "valuation"]

valid_classification = ["dividend yield", "total dividend", "revenue", "earnings", "market cap"]

valid_class = ['top gainers', 'top losers']

valid_periods = ['1d', '7d', '14d', '30d', '365d']

def fetch_data(url: str) -> Dict:
    """
    Fetch data from the given URL with authorization headers.

    :param url: URL to fetch data from.
    :return: JSON response as a dictionary.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return {"error": "An error occurred while fetching the data."}

def create_altair_chart(data: pd.DataFrame, x: str, y: str, title: str, mark_type: str = 'bar', height: int = 500, width: int = 700) -> alt.Chart:
    """
    Create an Altair chart from a DataFrame.
    
    :param data: DataFrame containing chart data.
    :param x: Column name for x-axis.
    :param y: Column name for y-axis.
    :param title: Title of the chart.
    :param mark_type: Type of mark (e.g., 'line', 'bar', etc.)
    :param height: Height of the chart.
    :param width: Width of the chart.
    :return: Altair Chart object.
    """
    # Map mark types to Altair methods
    marks = {
        'line': alt.Chart(data).mark_line(),
        'bar': alt.Chart(data).mark_bar(),
        'point': alt.Chart(data).mark_point(),
        'area': alt.Chart(data).mark_area()
    }
    
    chart = marks.get(mark_type, alt.Chart(data).mark_line()).encode(
        x=alt.X(x, sort=None),
        y=alt.Y(y, sort=None),
        tooltip=[x, y]
    ).properties(
        title=title,
        height=height,
        width=width
    ).interactive()
    
    # Add text marks for labels
    labels = chart.mark_text(
        dy=-15,
        align='center',
        baseline='alphabetic',
        font='sans-serif',
        fontSize=12,
        color='yellow'
    ).encode(
        text=y
    )
    
    return chart + labels

# LangChain Tool
@tool
def get_subsectors() -> str:
    """
    Helper List.
    Get list of all available subsectors.
    """
    url = "https://api.sectors.app/v1/subsectors/"
    return json.dumps(fetch_data(url))

@tool
def get_industries() -> str:
    """
    Helper List
    Get list of all available industries.
    """
    url = "https://api.sectors.app/v1/industries/"
    return json.dumps(fetch_data(url))

@tool
def get_subindustries() -> str:
    """
    Helper List
    Get list of all available subindustries.
    """
    url = "https://api.sectors.app/v1/subindustries/"
    return json.dumps(fetch_data(url))

@tool
def get_companies_by_idx(index: str) -> Dict:
    """
    Helper List.
    Get list of companies from available stock index.

    :param index: Indonesia stock index to filter by. Must be lowercase.
    :return: List of companies in the specified index.
    """

    if index not in valid_indices:
        raise ValueError(f"Invalid index '{index}'. Must be one of {valid_indices}")

    url = f"https://api.sectors.app/v1/index/{index}/"
    return fetch_data(url)

@tool
def get_companies_by_subsector(sub_sector: str) -> Dict:
    """
    Helper List.
    Get list of companies from a subsector.
    Get the available sub_sector from the get_subsectors tool.

    :param sub_sector: Subsector to filter by.
    :return: List of companies in the specified subsector.
    """
    url = f"https://api.sectors.app/v1/companies/?sub_sector={sub_sector.replace(' ', '-')}"
    return fetch_data(url)

@tool
def get_companies_by_subindustry(sub_industry: str) -> Dict:
    """
    Helper List.
    Get list of companies from a subindustry.

    :param sub_industry: Subindustry to filter by.
    :return: List of companies in the specified subindustry.
    """
    url = f"https://api.sectors.app/v1/companies/?sub_industry={sub_industry.replace(' ', '-')}"
    return fetch_data(url)

@tool
def get_companies_with_revenue_and_cost_segments() -> str:
    """
    Helper List.
    Get list of companies with revenue and cost segments, and their available financial years.
    """
    url = "https://api.sectors.app/v1/companies/list_companies_with_segments/"
    return json.dumps(fetch_data(url))

@tool
def get_company_perf_since_ipo(stock: str):
    """
    Use this tool to answer question related to stock performance since its IPO listing.
    Listing performance data is accessible only for tickers listed after May 2005.
    Get percentage gain since the IPO listing of a given stock symbol.
        chg_7d: The price change in the last 7 days.
        chg_30d: The price change in the last 30 days.
        chg_90d: The price change in the last 90 days.
        chg_365d: The price change in the last 365 days.
    Always show data as percentage and their up/down connotation
    A valid stock parameter consists of 4 letters, optionally followed by ‘.jk’, and is case-insensitive.

    :param stock: 4-character stock symbol.
    :return: Performance data as a JSON string.
    """
    if stock.endswith(('.jk','.JK')):
        return stock[:-3]
    if len(stock) != 4:
        raise ValueError("Stock symbol must be 4 characters, e.g 'BBRI'")
    
    url = f"https://api.sectors.app/v1/listing-performance/{stock}/"
    data = fetch_data(url)

    if "error" not in data:
        # Prepare the data for charting
        perf_data = {
            "Period": ["7d", "30d", "90d", "365d"],
            "Change": [data["chg_7d"], data["chg_30d"], data["chg_90d"], data["chg_365d"]]
        }
        df = pd.DataFrame(perf_data)
        
        # Create a chart
        perf_chart = create_altair_chart(df, x="Period", y="Change", title=f"{stock} Performance Since IPO", mark_type = 'line')

        # Return both the performance data and the chart object
        return {
            "data": data,
            "chart": st.altair_chart(perf_chart)
        }

    return json.dumps({"error": "Unable to fetch performance data."})

@tool
def get_company_report(stock: str, sections: Optional[str] = None) -> str:
    """
    Detailed Reports.
    Use this tool to get company report for a specified stock and sections.
    Get the available stock symbol from the get_companies_by_subsector or the get_companies_by_subindustry endpoints.
    Use comma to separate each section when retrieving data of more than one section. Do not use 'all'.
    A valid stock parameter consists of 4 letters, optionally followed by ‘.jk’, and is case-insensitive.

    :param stock: 4-character stock symbol.
    :param sections: Comma-separated list of sections to retrieve.
        overview: An overview of the listing details and key metrics of the given ticker (listing board, industry, sub industry, sector, sub sector, market cap, market cap rank, address, employee num, listing date, website, phone number, email, last close price, latest close date, and daily close change).
        valuation: Valuation metrics and related information of the given ticker.
        future: Future forecasts related to the given ticker.
        financials: Historical financials and key financial ratios of the given ticker.
        dividend: Historical dividend information of the given ticker.
        management: Key management and their shareholdings of the given ticker (key executives, executives shareholdings).
        ownership: Ownership details including major shareholders and transaction of the given ticker (major shareholders, ownership percentage, top transactions, and monthly net transactions)
        peers: Information about the peers of the given ticker.
    :return: Company report in JSON format as a string.
    """
    if stock.endswith(('.jk','.JK')):
        return stock[:-3]
    if len(stock) != 4:
        raise ValueError(stock_not_valid)
    
    url = f"https://api.sectors.app/v1/company/report/{stock}/"

    if sections == 'all' or sections is None:
        url += f"?sections=overview,valuation,financials,management,ownership"
    else:
        if sections != 'all':
            assert all(section in valid_sections for section in sections.split(',')), f"Invalid sections {sections}. Must be one of {valid_sections}"
            url += f"?sections={sections}"

    return json.dumps(fetch_data(url))

@tool
def get_company_revenue_and_cost_segments(stock: str, financial_year: Optional[int] = None) -> str:
    """
    Detailed Reports.
    Get revenue and cost segments of a given ticker.
    A valid stock parameter consists of 4 letters, optionally followed by ‘.jk’, and is case-insensitive.

    :param stock: 4-character stock symbol.
    :return: Company revenue and cost segments data as a JSON string.
    """
    if stock.endswith(('.jk','.JK')):
        return stock[:-3]
    if len(stock) != 4:
        raise ValueError(stock_not_valid)
    
    url = f"https://api.sectors.app/v1/company/get-segments/{stock}/"

    if financial_year is not None:
        # Add the financial year as a query parameter if it's provided
        url += f"?financial_year={financial_year}"

    return json.dumps(fetch_data(url))

@tool
def get_subsector_reports(sector: str, sub_sections: Optional[str] = None) -> str:
    """
    Detailed Reports.
    Get detailed statistics of a given subsector, organized into 
    distinct sections (eg. statistics, market_cap, stability, valuation, growth, companies, default to 'all')

    :param sector: Sector to filter by.
    :return: Company revenue and cost segments data as a JSON string.
    """
    
    url = f"https://api.sectors.app/v1/subsector/report/sector={sector.replace(' ','-')}/"

    if sub_sections:
        assert sub_sections in valid_sub_sections, f"Invalid section {sub_sections}. Must be one of {valid_sub_sections}"
        url += f"?sections={sub_sections.replace(' ','_')}"

    return json.dumps(fetch_data(url))

@tool
def get_most_traded_stocks_by_volume(start_date: str, end_date: str, sub_sector: str = "", top_n: int = 5):
    """
    Retrieve the Most Traded Stocks by Transaction Volume.
    This tool ranks the most traded stocks based on transaction volume over a specified period.
    You can filter results by sub-sector and retrieve up to the top `n` stocks.
    Automatically adjusts the `end_date` if it falls on a weekend.
    Use this tool for queries like:
      - "Top 5 companies by volume on the 1st of this month."
      - "Most traded stocks yesterday."
      - "Top 7 most traded stocks from June 6th to June 10th."
      - "Top 3 companies by volume over the last 7 days."
    Date format: 'YYYY-MM-DD'.

    :param start_date: Start date for the period in 'YYYY-MM-DD' format.
    :param end_date: End date for the period in 'YYYY-MM-DD' format. If it falls on a weekend, it's adjusted to the next weekday.
    :param sub_sector: Filter by a specific sub-sector (default is no filter).
    :param top_n: Number of top companies to retrieve. Defaults to 5, with a max of 10.
    """

    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}&sub_sector={sub_sector.replace(' ','-')}"
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    original_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    while True:
        data = fetch_data(url)
        
        if data and "error" not in data:  # Data fetched successfully
            break
        
        # If no data is returned, increment end_date by 1 day and retry
        end_date_obj += timedelta(days=1)
        end_date = end_date_obj.strftime('%Y-%m-%d')

        url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}&sub_sector={sub_sector.replace(' ','-')}"

    if not data or "error" in data:
        return json.dumps({"error": "Unable to fetch most traded stocks by volume data."})

    # Prepare the data for charting
    flattened_data = []
    for date,stocks in data.items():
        for stock in stocks:
            stock['date']=date
            flattened_data.append(stock)

    vol_data = pd.DataFrame(flattened_data)
    df = vol_data.groupby(['symbol','company_name'],as_index=False).agg(Volume=pd.NamedAgg(column='volume',aggfunc='sum'),
                                                        Price=pd.NamedAgg(column='price', aggfunc='mean'))
    df = df.rename(columns={'symbol':'Symbol'}).sort_values(by='Volume', ascending=False).iloc[:top_n].reset_index()

    # Create a chart
    vol_chart = create_altair_chart(df, x="Symbol:N", y="Volume:Q", title=f"Most Traded Stocks from {start_date} to {end_date} by Volume", mark_type = 'bar')

    # Return both the most traded stocks by volume data and the chart object
    return {
        "data": df,
        "chart": st.altair_chart(vol_chart),
        "message": f"The end date has been adjusted from {original_end_date} to {end_date} due to the Indonesia Stock Exchange was closed."
    }

@tool
def get_top_companies_ranked(classifications: Optional[str] = None, sub_sector: str = "", top_n: int = 5, year: int = 2024) -> str:
    """
    Company Ranking by Dimensions.
    Get a list of IDX companies in a given year that ranks top on a specified dimension (dividend yield, total dividend, revenue, earnings, or market cap).
    n_stock parameter can be used to specify number of stocks you want to show (default to 5, max. 10).
    year can be used to specify year for which the list of IDX companies is to be retrieved (default to the current year).

    :param classification: 
    :param sub_sector: Sub-sector to filter by.
    :param top_n: Number of top companies to retrieve.
    :param year: 
    :return: Top companies ranked by dimension in JSON format as a string.
    """

    url = f"https://api.sectors.app/v1/companies/top/?n_stock={top_n}&year={year}&sub_sector={sub_sector.replace(' ','-')}"

    if classifications:
        assert classifications in valid_classification, f"Invalid section {classifications}. Must be one of {valid_classification}"
        url = f"https://api.sectors.app/v1/companies/top/?classifications={classifications.replace(' ','_')}&n_stock={top_n}&year={year}&sub_sector={sub_sector.replace(' ','-')}"

    return json.dumps(fetch_data(url))

@tool
def get_top_company_movers(classification: Optional[str] = None, n_stock: int = 5, periods: Optional[str] = None, sub_sector: Optional[str] = None):
    """
    Fetch the n company movers based on significant price changes within a specified period.
    This tool retrieves companies experiencing the largest gains or losses over a defined time frame.
    The results can be filtered by classification ('top gainers', 'top losers'), time period, and sub-sector.
    All filters are optional. The URL is dynamically constructed based on provided parameters.
    Defaults apply when filters are not provided.

    :param classification: Filter by classification ('top gainers', 'top losers'). 
    :param n_stock: Number of top stocks to retrieve (default: 5, max: 10).
    :param periods: Time period for changes. Must be one of '1d', '7d', '14d', '30d', or '365d'.
    :param sub_sector: Filter results by sub-sector.
    """
    valid_periods = ['1d', '7d', '14d', '30d', '365d']

    if periods not in valid_periods:
        raise ValueError(f"Invalid index '{periods}'. Must be one of {valid_periods}")
    url = f"https://api.sectors.app/v1/companies/top-changes/?n_stock={n_stock}&"
    # https://api.sectors.app/v1/companies/top-changes/?classifications={classifications}&n_stock={n_stock}&periods={periods}&sub_sector={sub_sector}
    
    if classification:
        assert classification in valid_class, f"Invalid section {classification}. Must be one of {valid_class}"
        url += f"classifications={classification.replace(' ','_')}&"

    if periods:
        assert periods in valid_periods, f"Invalid section {periods}. Must be one of {valid_periods}"
        url += f"periods={periods}&"

    return json.dumps(fetch_data(url))

@tool
def get_top_companies_by_growth():
    """
    """
    pass

@tool
def get_hist_market_cap():
    """
    """
    pass

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str):
    """
    Use this tool to get daily transaction data for a specific stock within a date range.
    A valid stock parameter consists of 4 letters, optionally followed by ‘.jk’, and is case-insensitive.

    :param stock: 4-character stock symbol.
    :param start_date: Start date for the transaction data.
    :param end_date: End date for the transaction data.
    :return: Daily transaction data in JSON format as a string.
    """
    if stock.endswith(('.jk','.JK')):
        return stock[:-3]
    if len(stock) != 4:
        raise ValueError(stock_not_valid)

    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    original_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    while True:
        data = fetch_data(url)
        
        if data and "error" not in data:  # Data fetched successfully
            break
        
        # If no data is returned, increment end_date by 1 day and retry
        end_date_obj += timedelta(days=1)
        end_date = end_date_obj.strftime('%Y-%m-%d')
        url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"

    if not data or "error" in data:
        return json.dumps({"error": "Unable to fetch most traded stocks by volume data."})

    # Prepare the data for charting
    daily_data = {
        "Date": [entry['date'] for entry in data],
        "Close": [entry['close'] for entry in data]
    }
    df = pd.DataFrame(daily_data)
    
    # Create a chart
    daily_chart = create_altair_chart(df, x="Date", y="Close", title=f"{stock} Daily Transaction from {start_date} to {end_date}", mark_type = 'line')

    # Return both the daily transaction data and the chart object
    return {
        "data": data,
        "chart": st.altair_chart(daily_chart),
        "message": f"The end date has been adjusted from {original_end_date} to {end_date} due to the Indonesia Stock Exchange was closed."
    }

@tool
def get_idx_daily_tx(index: str, start_date: str, end_date: str):
    """
    Use this tool to get daily transaction data for a specific index within a date range.

    :param index: Indonesia stock index to filter by. Must be lowercase.
    :param start_date: Start date for the transaction data.
    :param end_date: End date for the transaction data.
    :return: Daily transaction data in JSON format as a string.
    """
    if index not in valid_indices:
        raise ValueError(f"Invalid index '{index}'. Must be one of {valid_indices}")

    url = f"https://api.sectors.app/v1/index-daily/{index}/?start={start_date}&end={end_date}"
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    original_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    while True:
        data = fetch_data(url)
        
        if data and "error" not in data:  # Data fetched successfully
            break
        
        # If no data is returned, increment end_date by 1 day and retry
        end_date_obj += timedelta(days=1)
        end_date = end_date_obj.strftime('%Y-%m-%d')
        url = f"https://api.sectors.app/v1/index-daily/{index}/?start={start_date}&end={end_date}"

    if not data or "error" in data:
        return json.dumps({"error": "Unable to fetch most traded stocks by volume data."})

    # Prepare the data for charting
    daily_data = {
        "Date": [entry['date'] for entry in data],
        "Price": [entry['price'] for entry in data]
    }
    df = pd.DataFrame(daily_data)
    
    # Create a chart
    daily_chart = create_altair_chart(df, x="Date", y="Price", title=f"{index} Daily Transaction from {start_date} to {end_date}", mark_type = 'line')

    # Return both the daily transaction data and the chart object
    return {
        "data": data,
        "chart": st.altair_chart(daily_chart),
        "message": f"The end date has been adjusted from {original_end_date} to {end_date} due to the Indonesia Stock Exchange was closed."
    }

@tool
def search_google(query: str, num_results: Optional[int] = 10) -> str:
    """
    Searches Google and returns results.

    :param query: Search query string.
    :param num_results: Number of results to return.
    :return: Search results in JSON format as a string.
    """
    try:
        search_wrapper = GoogleSerperAPIWrapper(api_key=SERPER_API_KEY)
        return search_wrapper.run(query, num_results=num_results)
    except Exception as e:
        logging.error(f"Error during Google search: {e}")
        return f"An error occurred: {e}"

# Define tools
tools = [
    get_subsectors, get_industries, get_subindustries, get_companies_by_idx, get_companies_by_subsector, get_companies_by_subindustry, get_companies_with_revenue_and_cost_segments,
    get_company_perf_since_ipo,
    get_company_report, get_company_revenue_and_cost_segments, get_subsector_reports,
    get_most_traded_stocks_by_volume, get_top_companies_ranked, get_top_companies_by_growth,
    get_hist_market_cap, get_daily_tx, get_idx_daily_tx,
    search_google
]

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", f"""
        You are an AI assistant that must prioritize using tools to gather and process information. Always invoke the relevant tools to answer questions.
        You are an AI assistant using multiple tools to answer user queries. Always use the provided tools in sequence to gather, process, and deliver accurate information.
        Tool Invocation Priority: Start with tool invocation for any data retrieval or processing. Summarize and analyze the information from the tools in your response.
        Data Gathering: Start with the most relevant tool to collect data. Pass data between tools if needed. Use the search_google tool only if no other tool can provide an answer.
        Intermediate Results: Store intermediate results in the scratchpad for reference.
        Summarization and Analysis: Provide a clear summary and your analysis of the data.
        Chart Display: Render and explain any included charts.
        Handling Time Periods: Default to the current month and year unless specified otherwise. For single-day data, use the same start and end date. Adjust dates for weekends and public holidays, especially for the Indonesian stock exchange.
        Fallback to Historical Data: Use the most recent relevant data if real-time data is unavailable.
        Formatting & Parameters: Use lowercase for all parameters except stock symbols (4 letters, optionally followed by .jk). Ensure all parameters match the defined format exactly.
        If the query involves stock volume or transaction volume over a specific time period, use get_most_traded_stocks_by_volume.
        If the query involves significant changes in stock price (gainers or losers), use get_top_company_movers. Must have 'top gainers' or 'top losers' on the query.
        Data is exclusively from the Indonesian Stock Exchange; no other international stock exchanges are available.
        If there is a message in the output, always include it at the beginning of the response, rather than as a note at the end.
        If the result is a ranking, always present it using numbers.

        Current Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Today: {datetime.now().strftime('%Y-%m-%d')}
        Current Month: {datetime.now().strftime("%B %Y")}
        Current Year: {datetime.now().strftime('%Y')}
    """
    ),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
    # MessagesPlaceholder(variable_name="chat_history"),
])

# Initialize the LLM and AgentExecutor
llm = ChatGroq(temperature=0, model_name="llama3-groq-70b-8192-tool-use-preview", groq_api_key=GROQ_API_KEY)
# llama-3.1-70b-versatile

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example queries
# queries = [
#     "Get revenue and cost segments of BBRI",
#     "What are the top 3 companies by transaction volume in the 'banks' subsector over the last 7 days?",
#     "Retrieve the top 2 companies by transaction volume in the investment service subsector and get their report",
# 
#     "What are the top 5 companies by transaction volume on the first of this month?",
#     "What are the most traded stock yesterday?",
#     "What are the top 7 most traded stocks between 6th June to 10th June this year?",
#     "What are the top 3 companies by transaction volume over the last 7 days?",
#     "Based on the closing prices of BBCA between 1st and 30th of June 2024, are we seeing an uptrend or downtrend? Try to explain why.",
#     "What is the company with the largest market cap between BBCA and BREN? For said company, retrieve the email, phone number, listing date and website for further research.",
#     "What is the performance of GOTO (symbol: GOTO) since its IPO listing?",
#     "If I had invested into GOTO vs BREN on their respective IPO listing date, which one would have given me a better return over a 90 day horizon?"
# ]