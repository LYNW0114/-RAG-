import streamlit as st
from rag import RagService
import config_data as config

st.title("智能客服")
st.divider()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "你好有什么可以帮助你？"}]
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.spinner("思考中..."):
        res = st.session_state["rag"].chain.invoke({"input": prompt}, config=config.session_config)
    st.chat_message("assistant").write(res)
    st.session_state["messages"].append({"role": "assistant", "content": res})
