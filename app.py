import streamlit as st

import config
from rag import (
    MAX_CHAT_HISTORY_MESSAGES,
    generate_answer,
    retrieve_relevant_chunks,
    trim_chat_history,
)


st.set_page_config(page_title="Simple RAG Legal Chatbot", page_icon="⚖️", layout="wide")


def format_history(history):
    lines = []

    for role, content in history:
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    return history, "\n".join(lines).strip()


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "debug" not in st.session_state:
        st.session_state.debug = False

    if "top_k" not in st.session_state:
        st.session_state.top_k = 4


def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        st.caption("Streamlit UI for the existing CLI RAG pipeline.")

        st.session_state.debug = st.checkbox("DEBUG mode", value=st.session_state.debug)
        st.session_state.top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=st.session_state.top_k)

        st.divider()
        st.caption(f"Conversation memory: last {MAX_CHAT_HISTORY_MESSAGES} messages")
        st.caption(f"Retrieval collection: {config.COLLECTION_NAME}")


def render_chat_history():
    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)


def show_debug_info(original_question, rewritten_query, chunks, distances):
    st.subheader("Debug")

    with st.expander("Original question", expanded=False):
        st.write(original_question)

    with st.expander("Rewritten retrieval query", expanded=False):
        st.write(rewritten_query)

    with st.expander("Retrieved chunks + distances", expanded=True):
        for index, (chunk_text, distance) in enumerate(zip(chunks, distances), start=1):
            st.markdown(f"**Chunk {index}**  ")
            st.caption(f"Distance: {distance:.4f}")
            st.write(chunk_text)
            if index != len(chunks):
                st.divider()


def main():
    ensure_session_state()
    render_sidebar()

    st.title("Simple RAG Legal Chatbot")
    st.write("Ask questions about the document and follow up naturally. The app keeps only recent messages in memory for context.")

    render_chat_history()

    prompt = st.chat_input("Ask a question about the document")
    if not prompt:
        return

    user_message = prompt.strip()
    if not user_message:
        return

    recent_history = trim_chat_history(st.session_state.messages)

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and drafting an answer..."):
            chunks, distances, search_queries, rewritten_query = retrieve_relevant_chunks(
                user_message,
                recent_history,
                top_k=st.session_state.top_k,
            )

            answer = generate_answer(user_message, recent_history, chunks)

            st.markdown(answer)

            if st.session_state.debug:
                show_debug_info(user_message, rewritten_query, chunks, distances)
            else:
                with st.expander("Retrieved search queries", expanded=False):
                    for index, search_query in enumerate(search_queries, start=1):
                        st.write(f"{index}. {search_query}")

    st.session_state.messages.append(("user", user_message))
    st.session_state.messages.append(("assistant", answer))
    st.session_state.messages = trim_chat_history(st.session_state.messages)


if __name__ == "__main__":
    main()