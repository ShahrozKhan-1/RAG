from datetime import datetime
from pathlib import Path

import streamlit as st

from main import NoRelevantContextError, RAGResponse, get_rag_result


st.set_page_config(
    page_title="Financial Documents Assistant",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def load_styles() -> None:
    stylesheet = Path(__file__).with_name("styles.css")
    st.markdown(
        f"<style>{stylesheet.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


def current_timestamp() -> str:
    return datetime.now().astimezone().strftime("%H:%M")


def render_assistant_header() -> None:
    st.markdown(
        '<div class="assistant-name">AI Assistant</div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]) -> None:
    if not sources:
        return

    with st.expander("Sources", expanded=False):
        for source in sources:
            st.markdown(f"- {source}")


def render_message(message: dict) -> None:
    role = message["role"]
    avatar = "🤖" if role == "assistant" else "👤"

    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            render_assistant_header()
            if message.get("is_error"):
                st.error(message["content"])
                with st.expander("Technical details"):
                    st.code(message["error_details"])
            else:
                st.markdown(message["content"])
                render_sources(message.get("sources", []))
        else:
            st.markdown(message["content"])

        st.caption(message["timestamp"])


def render_welcome() -> None:
    st.markdown(
        """
        <section class="welcome-screen">
            <div class="welcome-eyebrow">Financial document intelligence</div>
            <h1 class="welcome-title">Ask your documents.</h1>
            <p class="welcome-copy">
                Explore figures, tables, and reports with answers grounded in
                your indexed documents.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_typing_indicator() -> None:
    st.markdown(
        """
        <div class="typing-indicator" role="status" aria-label="Assistant is thinking">
            <span></span><span></span><span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def conversation_history() -> list[dict[str, str]]:
    return [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages
        if message["role"] in {"user", "assistant"}
        and not message.get("is_error")
    ]


def render_response_form() -> tuple[str, bool]:
    with st.form("chat_form", clear_on_submit=True, border=False):
        icon_column, input_column, send_column = st.columns(
            [0.07, 0.81, 0.12],
            gap="small",
        )
        with icon_column:
            st.markdown(
                '<div class="input-symbol" aria-hidden="true">✧</div>',
                unsafe_allow_html=True,
            )
        with input_column:
            user_query = st.text_input(
                "Ask a question",
                placeholder="Ask me anything...",
                label_visibility="collapsed",
                key="chat_prompt",
            )
        with send_column:
            submitted = st.form_submit_button(
                "↗",
                help="Send message",
                use_container_width=True,
            )

    return user_query, submitted


def handle_submission(user_query: str, response_placeholder) -> None:
    if st.session_state.is_processing:
        return

    normalized_query = user_query.strip()
    if not normalized_query:
        return

    st.session_state.is_processing = True
    st.session_state.messages.append(
        {
            "role": "user",
            "content": normalized_query,
            "timestamp": current_timestamp(),
        }
    )
    history = conversation_history()[:-1]

    with response_placeholder.container():
        render_message(st.session_state.messages[-1])
        with st.chat_message("assistant", avatar="🤖"):
            render_assistant_header()
            render_typing_indicator()
            try:
                with st.spinner("Searching your documents..."):
                    response: RAGResponse = get_rag_result(
                        normalized_query,
                        chat_history=history,
                    )
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response.answer,
                        "sources": response.sources,
                        "timestamp": current_timestamp(),
                    }
                )
            except NoRelevantContextError as error:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": str(error),
                        "sources": [],
                        "timestamp": current_timestamp(),
                    }
                )
            except Exception as error:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": (
                            "I couldn't complete that request. Check that the "
                            "Gemini credentials, embedding service, and indexed "
                            "ChromaDB collection are configured."
                        ),
                        "sources": [],
                        "timestamp": current_timestamp(),
                        "is_error": True,
                        "error_details": f"{type(error).__name__}: {error}",
                    }
                )
            finally:
                st.session_state.is_processing = False

    st.rerun()


def main() -> None:
    load_styles()
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("is_processing", False)

    header_column, action_column = st.columns([0.78, 0.22])
    with header_column:
        st.markdown(
            """
            <div class="app-topbar">
                <div class="brand-mark" aria-hidden="true">✦</div>
                <div>
                    <div>DOCUMENT ASSISTANT</div>
                    <div class="topbar-subtitle">Grounded answers from your library</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if st.session_state.messages:
        with st.container(key="chat-transcript"):
            for message in st.session_state.messages:
                render_message(message)
    else:
        render_welcome()

    response_placeholder = st.empty()
    user_query, submitted = render_response_form()
    if submitted:
        handle_submission(user_query, response_placeholder)


if __name__ == "__main__":
    main()
