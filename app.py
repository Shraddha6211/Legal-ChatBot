import streamlit as st

import config
import document
from rag import (
    MAX_CHAT_HISTORY_MESSAGES,
    generate_answer,
    retrieve_relevant_chunks,
    summarize_document,
    trim_chat_history,
)


st.set_page_config(page_title="Simple RAG Legal Chatbot", page_icon="⚖️", layout="wide")


def ensure_session_state():
    if "debug" not in st.session_state:
        st.session_state.debug = False

    if "top_k" not in st.session_state:
        st.session_state.top_k = 4

    if "mode" not in st.session_state:
        st.session_state.mode = "Ask a Legal Query"

    if "kb_messages" not in st.session_state:
        st.session_state.kb_messages = []

    if "doc_messages" not in st.session_state:
        st.session_state.doc_messages = []

    if "active_document" not in st.session_state:
        st.session_state.active_document = None

    if "summary_cache" not in st.session_state:
        st.session_state.summary_cache = None


def render_sidebar():
    with st.sidebar:
        st.title("Settings")
        st.caption("Streamlit UI for the existing CLI RAG pipeline.")

        st.session_state.debug = st.checkbox("DEBUG mode", value=st.session_state.debug)
        st.session_state.top_k = st.slider("Retrieved chunks", min_value=2, max_value=8, value=st.session_state.top_k)

        st.divider()
        st.caption(f"Conversation memory: last {MAX_CHAT_HISTORY_MESSAGES} messages")
        st.caption(f"Retrieval collection: {config.COLLECTION_NAME}")


def get_current_history():
    if st.session_state.mode == "Upload & Query a Document":
        return st.session_state.doc_messages
    return st.session_state.kb_messages


def set_current_history(history):
    if st.session_state.mode == "Upload & Query a Document":
        st.session_state.doc_messages = history
    else:
        st.session_state.kb_messages = history


def render_chat_history():
    history = get_current_history()
    for role, content in history:
        with st.chat_message(role):
            st.markdown(content)


def clear_active_document():
    if st.session_state.active_document:
        document.delete_document(st.session_state.active_document["document_id"])

    st.session_state.active_document = None
    st.session_state.doc_messages = []
    st.session_state.summary_cache = None


def render_document_panel():
    active_document = st.session_state.active_document

    if active_document is not None:
        st.info(f"Current document: {active_document['filename']}")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("Generate Summary"):
                st.session_state.summary_cache = "__needs_summary__"

        with col2:
            if st.button("Ask Questions"):
                pass

        with col3:
            if st.button("Remove Document"):
                clear_active_document()
                return

        if st.session_state.summary_cache == "__needs_summary__":
            with st.spinner("Summarizing the uploaded document..."):
                st.session_state.summary_cache = summarize_document(
                    st.session_state.active_document["document_id"],
                    st.session_state.active_document["filename"],
                )

        if st.session_state.summary_cache and st.session_state.summary_cache != "__needs_summary__":
            st.subheader("Summary")
            st.markdown(st.session_state.summary_cache)

        st.write("---")
        st.write("Upload a new PDF to replace the currently active document.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="uploaded_pdf")

    if uploaded_file is not None:
        if st.session_state.active_document is not None:
            clear_active_document()

        pdf_bytes = uploaded_file.read()
        document_info = document.ingest_uploaded_document(pdf_bytes, uploaded_file.name)
        st.session_state.active_document = {
            "document_id": document_info["document_id"],
            "filename": uploaded_file.name,
            "chunk_count": document_info["chunk_count"],
        }
        st.session_state.doc_messages = []
        st.session_state.summary_cache = None
        st.success(f"Uploaded and indexed '{uploaded_file.name}' ({document_info['chunk_count']} chunks).")

    return


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


def render_legal_query_mode():
    st.title("Simple RAG Legal Chatbot")
    st.write("Ask questions about the legal knowledge base. The existing RAG pipeline is used unchanged.")

    render_chat_history()

    prompt = st.chat_input("Ask a legal question")
    if not prompt:
        return

    user_message = prompt.strip()
    if not user_message:
        return

    recent_history = trim_chat_history(get_current_history())

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching legal documents and drafting an answer..."):
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
                with st.expander("Expanded queries", expanded=False):
                    for index, search_query in enumerate(search_queries, start=1):
                        st.write(f"{index}. {search_query}")

    history = get_current_history()
    history.append(("user", user_message))
    history.append(("assistant", answer))
    set_current_history(trim_chat_history(history))


def render_document_query_mode():
    st.title("Upload & Query a Document")
    st.write("Upload a single PDF and ask questions that are answered only from that uploaded document.")

    render_document_panel()

    if st.session_state.active_document is None:
        return

    render_chat_history()

    prompt = st.chat_input("Ask a question about the uploaded document", key="doc_query")
    if not prompt:
        return

    user_message = prompt.strip()
    if not user_message:
        return

    recent_history = trim_chat_history(get_current_history())

    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching the uploaded document and drafting an answer..."):
            document_id = st.session_state.active_document["document_id"]
            chunks, distances, search_queries, rewritten_query = retrieve_relevant_chunks(
                user_message,
                recent_history,
                top_k=st.session_state.top_k,
                metadata_filter={"document_id": document_id},
            )

            answer = generate_answer(user_message, recent_history, chunks)

            st.markdown(answer)

            if st.session_state.debug:
                show_debug_info(user_message, rewritten_query, chunks, distances)
            else:
                with st.expander("Expanded search queries", expanded=False):
                    for index, search_query in enumerate(search_queries, start=1):
                        st.write(f"{index}. {search_query}")

    history = get_current_history()
    history.append(("user", user_message))
    history.append(("assistant", answer))
    set_current_history(trim_chat_history(history))


def main():
    # ensure_session_state()
    # render_sidebar()

    # st.radio(
    #     "Choose an experience",
    #     ["Ask a Legal Query", "Upload & Query a Document"],
    #     index=0 if st.session_state.mode == "Ask a Legal Query" else 1,
    #     key="mode",
    # )

    # if st.session_state.mode == "Ask a Legal Query":
    #     render_legal_query_mode()
    # else:
    #     render_document_query_mode()

    ensure_session_state()
    render_sidebar()

    st.markdown(
        """
        <h2 style="text-align: center; margin-bottom: 25px;">
            Choose an experience
        </h2>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # CUSTOM BUTTON STYLING
    # ---------------------------------------------------------

    st.markdown(
        """
        <style>

        div[data-testid="stHorizontalBlock"] {
            justify-content: center;
            gap: 25px;
        }

        div.stButton > button {
            height: 110px;
            width: 100%;
            border-radius: 16px;
            border: 1px solid #14213d;
            background-color: #14213d;
            font-size: 20px;
            font-weight: 600;
            color: #ffffff;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            border-color: #00ffff;
            background-color: #001e39;
            color: #00ffff;
            transform: translateY(-4px);
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


    # ---------------------------------------------------------
    # EXPERIENCE BUTTONS
    # ---------------------------------------------------------

    col1, col2= st.columns(2)

    with col1:
        if st.button(
            "⚖️  Ask a Legal Query",
            use_container_width=True,
        ):
            st.session_state.mode = "Ask a Legal Query"
            st.rerun()

    with col2:
        if st.button(
            "📄  Upload & Query a Document",
            use_container_width=True,
        ):
            st.session_state.mode = "Upload & Query a Document"
            st.rerun()

    # with col3:
    #     if st.button(
    #         "Draft a legal document",
    #         use_container_width=True,
    #     ):
    #         st.session_state.mode = "Draft a legal Document"
    #         st.rerun()

    # with col4:
    #         if st.button(
    #             "Book an appointment for legal consultation",
    #             use_container_width=True,
    #         ):
    #             st.session_state.mode = "Book an appointment for legal consultation"
    #             st.rerun()


    # ---------------------------------------------------------
    # CURRENT MODE
    # ---------------------------------------------------------

    if st.session_state.mode == "Ask a Legal Query":
        render_legal_query_mode()
    else:
        render_document_query_mode()




if __name__ == "__main__":
    main()
