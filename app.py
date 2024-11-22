import streamlit as st
from chat_functions import ChatHandler, answer_query_titan, answer_query_llama

# Initialize chat handler in session state if not already initialized
if 'chat_handler' not in st.session_state:
    st.session_state.chat_handler = ChatHandler()

def main():
    # Sidebar for model selection and focus area
    with st.sidebar:
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.image("AnitaMDorr.jpg", width=100)
        with col2:
            st.title("Hello! I'm ANITA")

        enafocus = st.radio(
            "ENA Focus",
            ("Position Statements", "Education"),
            index=0,
            help="Select the ENA focus area"
        )

        llm_model = st.radio(
            "LLM Model",
            ("Llama", "Titan"),
            index=1,
            help="Select the LLM model"
        )

        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()

    # Main chat interface
    st.title("ANITA - ENA AI Assistant")

    # Chat input
    if enafocus == "Position Statements":
        chat_input_prompt = "Ask me anything about ENA's position statements!"
    else:
        chat_input_prompt = "Ask me anything about ENA's education programs!"

    # Display chat history
    for message in st.session_state.chat_handler.get_messages():
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input(chat_input_prompt):
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Display assistant response
        with st.chat_message("assistant"):
            response_function = answer_query_llama if llm_model == "Llama" else answer_query_titan
            response = response_function(prompt, st.session_state.chat_handler)
            st.write(response)

if __name__ == "__main__":
    main()
