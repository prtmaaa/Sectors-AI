import streamlit as st
from llm_sectors_ai import agent_executor, get_most_traded_stocks_by_volume
import altair as alt
import json

st.set_page_config(
    page_title="SectorsAI",
    page_icon="🤑",
    layout="centered",
    initial_sidebar_state="expanded",
)

def about_page():
    st.title("About Our 🤑 Sectors AI Assistant")

    st.write("""
    ## Overview
    Welcome to our **[Sectors.app](https://sectors.app/) AI Assistant**! This tool is designed to provide you with detailed insights into the most traded stocks on the Indonesian Stock Exchange. 

    ### Features
    - **Advanced Language Processing** Utilizes a state-of-the-art Large Language Model (LLM) to understand and respond to user queries with high accuracy and contextually relevant information.
    - **Intelligent Query Handling** The LLM engine interprets a wide range of user queries, from simple questions to complex multi-step requests, ensuring comprehensive and relevant answers.
    - **Contextual Data Insights** Provides insights and explanations based on the latest data from the Indonesian Stock Exchange, processed through advanced AI algorithms to offer a deeper understanding of market trends.
    - **Filter by Parameter** Refine your results based on specific parameters such as sub-sectors or other criteria. The LLM engine is adept at managing complex filtering criteria to deliver precise and relevant results.
    - **Date Adjustments**: Automatically adjusts end dates for weekends and public holidays to ensure accurate data.
    - **Visualizations**: View your data in an easily interpretable format with interactive charts.

    ### How It Works
    1. **Set Parameters**: Define the number of top stocks or any specific sub-sector filters or any parameters you want to retrieve.
    2. **Fetch Data**: Our AI assistant will call the API, fetch the data, and process it.
    3. **Receive Results**: View the results in both tabular and graphical formats.

    ### Data Source
    Data is exclusively sourced from the Indonesian Stock Exchange collected from [sectors.app](https://sectors.app/) API. No data from other international stock exchanges is available.

    ### Contact
    If you have any questions or need further assistance, please contact our support team at [sectors@supertype.ai](mailto:sectors@supertype.ai).
    """)

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "About"])
    
    if page == "About":
        about_page()
    else:
        st.title("🤑 Sectors AI Assistant Chat")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Enter your question:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the user's query and generate a response
            with st.spinner("🧠 Thinking..."):
                try:
                    result = agent_executor.invoke({"input": prompt})
                    response_data = result.get("output", "No response received.")
                    # st.write(result)

                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_data})

                    # Display AI response in chat message container
                    with st.chat_message("assistant"):
                        st.success(response_data)

                except Exception as e:
                    st.error(f"An error occurred: {e}")