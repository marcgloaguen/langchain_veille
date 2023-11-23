import time

import streamlit as st
import pandas as pd

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = open("OPENAI_API_KEY", 'r').read()

st.title('Analyse your CSV')

if "messages" not in st.session_state:
    st.session_state.messages = []

if csv := st.file_uploader('Upload your CSV', type='csv'):
    data = pd.read_csv(csv, sep=',')
    chayGPTAgent = create_pandas_dataframe_agent(
        ChatOpenAI(model_name='gpt-4'),
        df=data,
        verbose=True
    )
    st.dataframe(data)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):

        with st.chat_message("human"):
            st.markdown(prompt)
        with st.chat_message("ai"):
            with st.spinner("Waiting...") :
                answer = chayGPTAgent.run(prompt)
                st.markdown(answer)
        st.session_state.messages.append({"role": "human", "content": prompt})
        st.session_state.messages.append({"role": "ai", "content": answer})

