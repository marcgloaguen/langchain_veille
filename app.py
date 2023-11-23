import os

import pandas as pd
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

os.environ["OPENAI_API_KEY"] = open("OPENAI_API_KEY", 'r').read()
st.set_page_config(layout="wide")
st.title('Analyse your CSV')

if "messages" not in st.session_state:
    st.session_state.messages = []

if csv := st.file_uploader('', type='csv'):
    data = pd.read_csv(csv, sep=',')
    chayGPTAgent = create_pandas_dataframe_agent(
        ChatOpenAI(model_name='gpt-4'),
        df=data,
        verbose=True,
        handle_parsing_errors=True
    )

    st.dataframe(data)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Message..."):

        with st.chat_message("human"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            with st.spinner("I'm thinking..."):
                try:
                    answer = chayGPTAgent.run(prompt)
                except:
                    answer = "i can't answer"
                st.markdown(answer)
        st.session_state.messages.extend([{"role": "human", "content": prompt}, {"role": "ai", "content": answer}])
