import streamlit as st
from hugchat import hugchat
from hugchat.login import Login
import json
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Function for generating embeddings from the knowledge base


def generate_embeddings():
    # Load training data from CSV file
    loader = CSVLoader("training_data.csv")
    documents = loader.load()

    # Create a FAISS vector store from the documents
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)
    return db

# Function for retrieving context from the knowledge base


def retrieve_context(db, prompt_input):
    similar_response = db.similarity_search(prompt_input, k=3)
    content_arr = [doc.page_content for doc in similar_response]
    print(content_arr)
    return content_arr


# Function for signing in to Hugging Face


def sign_in(email, password):
    # Hugging Face Login
    sign = Login(email, password)
    # Path to save cookies
    cookie_path_dir = "./cookies_snapshot"
    # Save cookies to the local directory
    sign.saveCookiesToDir(cookie_path_dir)
    return Login(email, password)

# Function for adding custom CSS


def add_custom_css():
    st.markdown("""
    <style>
    .wrapper {
        word-wrap: break-word;
    }
    </style>
    """, unsafe_allow_html=True)

# Function for generating LLM response


def generate_response(conversation, prompt_input, cookies, value, context=None,):
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

# Function for getting cookies


def get_cookies(sign):
    try:
        cookie_path_dir = "./cookies_snapshot"
        cookies = sign.loadCookiesFromDir(cookie_path_dir)
    except:
        cookies = sign.login()

    return cookies


def main():

    # App title
    st.set_page_config(
        page_title="Rescale Lab AI Chatbot by Timothy", page_icon=":robot_face:")

    # Initialise session state for storing messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "AI", "content": "Hello, I am Joe from RescaleLab. How may I help you today?"}]

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Sidebar for Hugging Face credentials, chat history download and buffer memory adjustment
    with st.sidebar:
        st.title('Rescale Lab AI Chatbot')

        if ('EMAIL' in st.secrets) and ('PASS' in st.secrets):
            st.success(
                'HuggingFace Login credentials already provided!', icon='✅')
            hf_email = st.secrets['EMAIL']
            hf_pass = st.secrets['PASS']
        else:
            hf_email = st.text_input('Enter E-mail:', type='default')
            hf_pass = st.text_input('Enter password:', type='password')
            if not (hf_email and hf_pass):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success(
                    'Proceed to entering your prompt message!', icon='👉')
                credentials = sign_in(hf_email, hf_pass)

        # Determine if the chat input should be enabled based on the availability of credentials
        is_input_enabled = hf_email and hf_pass

        # Buffer memory adjustment slider
        st.title("Adjust the Buffer Memory")
        value = st.slider("Select a value", min_value=0, max_value=20, value=0)

        # Download chat history button
        chat_history_json = json.dumps(st.session_state.messages, indent=4)
        st.download_button(
            label="Download chat history",
            data=chat_history_json,
            file_name="chat_history.json",
            mime="application/json",
            disabled=not is_input_enabled
        )

    # Generate embeddings from the knowledge base
    db = generate_embeddings()

    # Chat input for user prompts
    prompt = st.chat_input(disabled=not is_input_enabled)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from AI
    if st.session_state.messages[-1]["role"] != "AI":
        with st.chat_message("AI"):
            cookies = get_cookies(credentials)
            with st.spinner("Generating response..."):
                similar_context = retrieve_context(db, prompt)
                response = generate_response(
                    st.session_state.messages, prompt, cookies, value, similar_context)
        message = {"role": "AI", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
