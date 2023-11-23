import streamlit as st
import pandas as pd

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = open("OPENAI_API_KEY", 'r').read()

st.title('Analyse your CSV')

csv = st.file_uploader('Upload your CSV', type='csv')
if csv:
    data = pd.read_csv(csv, sep=',')
    chayGPTAgent = create_pandas_dataframe_agent(
        ChatOpenAI(model_name='gpt-4'),
        df=data,
        verbose=True
    )
    st.dataframe(data)
    prompt = st.text_input("Prompt : ")
    if prompt:
        with st.spinner('Wait for it...'):
            answer = chayGPTAgent.run(prompt)
            st.markdown(answer)
