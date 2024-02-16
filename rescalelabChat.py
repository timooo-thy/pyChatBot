import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
import json
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os


def generate_embeddings():
    """Function for generating embeddings from the knowledge base"""
    # Load training data from CSV file
    loader = CSVLoader(
        "data/training_data.csv")
    documents = loader.load()

    # Create a FAISS vector store from the documents
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)

    # Save the vector store to the local directory
    db.save_local("faiss_index")
    return db


def retrieve_context(db, prompt_input):
    """Function for retrieving context from the knowledge base"""
    similar_response = db.similarity_search(prompt_input, k=3)
    content_arr = [doc.page_content for doc in similar_response]
    return content_arr


def sign_in(email, password):
    """Function for signing in to Hugging Face"""
    sign = Login(email, password)
    # Path to save cookies
    cookie_path_dir = "./cookies_snapshot"
    # Save cookies to the local directory
    sign.saveCookiesToDir(cookie_path_dir)
    return Login(email, password)


def add_custom_css():
    """Function for adding custom CSS"""
    st.markdown("""
    <style>
    .wrapper {
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)
#


def generate_response(conversation, prompt_input, cookies, value, context=None,):
    """Function for generating LLM response"""
    add_custom_css()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    chatbot.switch_llm(1)
    # Create a placeholder
    placeholder = st.empty()

    # Initialize an empty string to collect responses
    collected_responses = ''

    # Create a prompt chaining the conversation and the new message from the user
    prompt = (f"This is a fun conversation between an Joe(AI) and a human. Your role is to engage users with informative, very entertaining responses about Rescale Lab in a human-like tone."
              "Focus on providing accurate and relevant information about the company's services and initiatives."
              "If a query falls outside Rescale Lab's scope, politely steer the conversation back to topics directly related to Rescale Lab."
              "Use the existing knowledge base and infer as needed to maintain a helpful and relevant dialogue, ensuring each interaction is aligned with Rescale Lab's themes and values."
              "Keep responses short and concise, and avoid using technical jargon. Follow best practices from the knowledge base strictly."
              f"Knowledge base={context} The last {value} conversation is {conversation[:value]}." + f"The new message from human: {prompt_input}. AI:")

    # Enable streaming responses from the chatbot
    for resp in chatbot.query(prompt, stream=True):
        # Append the token to the collected_responses string
        if type(resp) == dict:
            collected_responses += resp['token']
            # Update the placeholder with the new string, applying the wrapper class
            placeholder.markdown(
                f"<div class='wrapper'>{collected_responses}</div>", unsafe_allow_html=True)

    return collected_responses


def get_cookies(sign):
    """Function for getting cookies from the local directory"""
    try:
        cookie_path_dir = "./cookies_snapshot"
        cookies = sign.loadCookiesFromDir(cookie_path_dir)
    except:
        cookies = sign.login()

    return cookies


def save_conversations(conversations, email):
    """Function for saving conversations to JSON file"""

    if not os.path.exists(f'conversations/{email}'):
        os.makedirs(f'conversations/{email}')

    conversations_file = f'conversations/{email}/conversations_history.json'

    with open(conversations_file, 'w') as f:
        json.dump(conversations, f, indent=4)


def load_conversations(email):
    """Function for loading conversations from JSON file"""
    conversations_file = f'conversations/{email}/conversations_history.json'
    if os.path.exists(conversations_file):
        with open(conversations_file, 'r') as f:
            return json.load(f)
    return {"Conversation 1": [
            {"role": "AI", "content": "Hello, I am Joe from RescaleLab. How may I help you today?"}]}


def change_user():
    st.session_state.messages = load_conversations(st.session_state.email)
    st.session_state.current_conversation = next(
        iter(st.session_state.messages))
    st.session_state.conversation_count = len(
        st.session_state.messages)


def main():

    # Load environment variables
    load_dotenv()

    # App title
    st.set_page_config(
        page_title="Rescale Lab AI Chatbot by Timothy", page_icon=":robot_face:")

    # Sidebar for Hugging Face credentials, chat history download and buffer memory adjustment
    with st.sidebar:
        st.title('Rescale Lab AI Chatbot')

        email = st.text_input(
            'Enter E-mail:', type='default', on_change=change_user)
        password = st.text_input(
            'Enter password:', type='password')
        if not (email and password):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success(
                'Proceed to entering your prompt message!', icon='👉')
            credentials = sign_in(email,
                                  password)

        # Initialise session state for storing current conversation
        if 'email' not in st.session_state:
            st.session_state.email = ""

        if "messages" not in st.session_state or st.session_state.email != email:
            st.session_state.email = email
            st.session_state.messages = load_conversations(email)

        # Initialise session state for storing current conversation
        if 'current_conversation' not in st.session_state:
            st.session_state.current_conversation = next(
                iter(st.session_state.messages))

        # Initialise session state for storing conversation count
        if 'conversation_count' not in st.session_state:
            st.session_state.conversation_count = len(
                st.session_state.messages)

        # Determine if the chat input should be enabled based on the availability of credentials
        is_input_enabled = email and password

        # Sidebar for creating a new conversation
        if st.sidebar.button("Create a new conversation", disabled=not is_input_enabled):
            st.session_state.messages[f"Conversation {st.session_state.conversation_count + 1}"] = [
                {"role": "AI", "content": "Hello, I am Joe from RescaleLab. How may I help you today?"}]
            st.session_state.conversation_count += 1
            st.session_state.current_conversation = f"Conversation {
                st.session_state.conversation_count}"

        # Download chat history button
        chat_history_json = json.dumps(st.session_state.messages, indent=4)
        if st.download_button(
            label="Download current chat history",
            data=chat_history_json,
            file_name=f"{st.session_state.current_conversation}_history.json",
            mime="application/json",
            disabled=not is_input_enabled
        ):
            st.toast("Chat history downloaded!", icon="📥")

        if st.button("Delete chat history", disabled=not is_input_enabled):
            st.toast("Chat history deleted!", icon="🗑️")

            # Delete current conversation
            del st.session_state.messages[st.session_state.current_conversation]

            if len(st.session_state.messages) == 0:
                st.session_state.messages = {"Conversation 1": [
                    {"role": "AI", "content": "Hello, I am Joe from RescaleLab. How may I help you today?"}]}
                st.session_state.current_conversation = "Conversation 1"
                st.session_state.conversation_count = 1

            else:
                # Update to another existing conversation
                st.session_state.current_conversation = next(
                    iter(st.session_state.messages))

        # Sidebar for selecting conversation
        st.session_state.current_conversation = st.sidebar.selectbox(
            "Select a conversation", list(st.session_state.messages.keys()), index=list(st.session_state.messages.keys()).index(st.session_state.current_conversation))

        # Buffer memory adjustment slider
        st.title("Adjust the Buffer Memory")
        value = st.slider("Select a value", min_value=0, max_value=20, value=5)

    # Display existing chat messages
    for message in st.session_state.messages[st.session_state.current_conversation]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Retrieve embeddings from the knowledge base
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)

    # Chat input for user prompts
    prompt = st.chat_input(disabled=not is_input_enabled)

    if prompt:
        st.session_state.messages[st.session_state.current_conversation].append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from AI
    if st.session_state.messages[st.session_state.current_conversation][-1]["role"] != "AI":
        with st.chat_message("AI"):
            cookies = get_cookies(credentials)
            with st.spinner("Generating response..."):
                similar_context = retrieve_context(db, prompt)
                response = generate_response(
                    st.session_state.messages[st.session_state.current_conversation], prompt, cookies, value, similar_context)
        message = {"role": "AI", "content": response}
        st.session_state.messages[st.session_state.current_conversation].append(
            message)

        save_conversations(st.session_state.messages, email
                           )


if __name__ == "__main__":
    main()
