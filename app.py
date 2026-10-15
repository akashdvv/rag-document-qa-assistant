"""
Streamlit chat UI for the RAG assistant.

    streamlit run app.py
"""
import streamlit as st

from rag.pipeline import RAGPipeline

st.set_page_config(page_title="PEP Q&A Assistant", page_icon="📘", layout="wide")


@st.cache_resource(show_spinner="Indexing documents...")
def load_pipeline() -> RAGPipeline:
    return RAGPipeline()


rag = load_pipeline()

with st.sidebar:
    st.header("Settings")
    options = ["bm25", "semantic", "hybrid"]
    retriever = st.radio("Retriever", options, index=options.index(rag.default_retriever),
                         format_func=lambda k: rag.retrievers()[k].name)
    rag.top_k = st.slider("Passages to retrieve", 3, 10, 5)
    st.caption(f"{len(rag.docs)} PEPs · {len(rag.chunks):,} chunks · answers by **{rag.generator.name}**")
    if rag.generator.name == "Extractive":
        st.info("Set ANTHROPIC_API_KEY to get generated answers from Claude. "
                "Without it, the app quotes the most relevant sentences.")

st.title("📘 Ask the Python Enhancement Proposals")
st.caption("Answers are grounded in the indexed PEPs and cite the passages they come from.")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"].answer)

if question := st.chat_input("e.g. Why was the walrus operator added to Python?"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            ans = rag.ask(question, retriever)
        st.write(ans.answer)
        with st.expander(f"Sources ({ans.retriever})"):
            for h in ans.sources:
                st.markdown(f"**[{h.rank}] {h.chunk.citation}** · score {h.score:.3f}")
                st.caption(h.chunk.text[:600] + ("..." if len(h.chunk.text) > 600 else ""))
    st.session_state.history.append({"q": question, "a": ans})
