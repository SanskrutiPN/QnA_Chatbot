import streamlit as st
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="VIT Admission Assistant",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CLEAN MODERN CSS ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 900px;
    margin: auto;
}
.main-title {
    text-align: center;
    font-size: 44px;
    font-weight: 700;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 16px;
    margin-bottom: 30px;
}
.divider {
    border-top: 1px solid #2c2f36;
    margin: 25px 0 35px 0;
}
.chat-container { width: 100%; }
.chat-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 20px;
}
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 12px;
    font-size: 18px;
}
.user-avatar {
    background-color: #ff4b4b;
    color: white;
}
.bot-avatar {
    background-color: #f59e0b;
    color: black;
}
.message {
    background-color: #1f2937;
    padding: 16px 20px;
    border-radius: 14px;
    font-size: 15px;
    line-height: 1.6;
    width: 100%;
}
.input-wrapper {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
}
.input-box { width: 900px; }
.stTextInput > div > div > input {
    height: 55px;
    border-radius: 14px;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎓 VIT Admission Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Official Admission & Fee Information Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------- LOAD RAG ----------------
@st.cache_resource
def load_rag():


    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

   
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        st.error("❌ GROQ_API_KEY not found. Please add it in Streamlit Cloud Secrets.")
        st.stop()


    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0,
        max_tokens=512
    )

    node_parser = SentenceSplitter(
        chunk_size=600,
        chunk_overlap=80
    )

    documents = SimpleDirectoryReader("vit").load_data()

    index = VectorStoreIndex.from_documents(
        documents,
        node_parser=node_parser
    )

    return index.as_query_engine(similarity_top_k=3)

query_engine = load_rag()

# ---------------- SESSION ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ---------------- SEND FUNCTION ----------------
def send_message():
    user_query = st.session_state.input_text.strip()

    if user_query == "":
        return

    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )

    with st.spinner("Searching documents..."):
        response = query_engine.query(user_query)
        answer = response.response

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    st.session_state.input_text = ""

# ---------------- CHAT DISPLAY ----------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="chat-row">
            <div class="avatar user-avatar">🙂</div>
            <div class="message">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-row">
            <div class="avatar bot-avatar">🤖</div>
            <div class="message">{msg['content']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- INPUT ----------------
st.markdown("<div class='input-wrapper'>", unsafe_allow_html=True)
st.markdown("<div class='input-box'>", unsafe_allow_html=True)

st.text_input(
    "",
    placeholder="Ask your question...",
    key="input_text",
    label_visibility="collapsed",
    on_change=send_message
)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
