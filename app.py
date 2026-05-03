import streamlit as st
import ask as ask_module

title = "RAG Search Sample"

st.title("RAG Search Sample")
query = st.text_input("Enter your query here", key="query")
if st.button("Search"):
    if st.session_state.query:
        reply = ask_module.ask(st.session_state.query)
        st.write("**Answer:**")
        st.write(reply["answer"])
        st.write("**Sources:**")
        for source in reply["sources"]:
            st.write(f"- [{source['title']}]({source['url']}) (Relevance: {source['relevance_score']})")
    else:
        st.warning("Please enter a query.")