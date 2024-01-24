## Overview

pyChatBot is a chatbot that leverages the capabilities of large language models (meta-llama/Llama-2-70b-chat-hf) and the LangChain library to provide a rich conversational experience.

The bot is integrated into a user-friendly interface created with Streamlit, allowing users to interact with the AI in real time.

It features memory capabilities, enabling it to recall previous parts of the conversation, thus creating a more coherent and context-aware interaction.

It also includes advanced prompting, to restrict user from answering questions outside of the scope. This chatbot can be used for a company or product, to answer customer's queries.

## Features

- **Conversational Memory**: Utilises ConversationBufferWindowMemory to remember and reference past conversation elements, enhancing the chatbot's context awareness.

- **Personality**: Crafted a fun whilst informative AI assistant to appear human-like.

- **Dynamic Interaction**: Streamlit interface allows for real-time interaction with the bot, making it more engaging and responsive.

- **Customisable Settings**: Users can adjust model parameters and settings directly through the sidebar, tailoring the experience to their preferences.

- **Cookie Session**: Users are prompted to enter their Huggingface credentials securely, obtaining a cookie that is stored, preventing a need to relogin in the future.

## Thought Process

### Types of memory to consider:

1. ConversationBufferMemory
2. ConversationSummaryMemory
3. ConversationBufferWindowMemory

   Pros:

   - Storing everything gives the LLM the maximum amount of information.
   - Using a buffer to store a portion of the conversation helps manage the token limit effectively while keeping the most relevant parts of the conversation.

   Cons:

   - More tokens mean slowing response times and higher costs.
   - Long conversations cannot be remembered as we hit the LLM token limit.

4. ConversationSummaryBufferMemory

   Pros:

   - Summariser means we can remember distant interactions.
   - Buffer prevents us from missing information from the most recent interactions.

   Cons:

   - Summariser increases token count for shorter conversations.
   - Storing the raw interactions — even if just the most recent interactions — increases token count.

   Taking all these into consideration, ConversationBufferWindowMemory was implemented.

   Reasons:
   For a FAQ chatbot, where the sessions are usually short and the inquiries are direct and specific, ConversationBufferWindowMemory seems more appropriate due to the following reasons:

   1. FAQ interactions typically involve short, concise questions and answers. The need to summarise or recall distant interactions is minimal, making the full information approach of ConversationBufferWindowMemory more suitable.

   2. Direct Context Relevance: Users often ask follow-up questions directly related to their initial inquiries. Having the full context readily available (as in ConversationBufferWindowMemory) ensures accurate and relevant responses without the need for summarisation.

   3. Simplicity and Performance: The simplicity of the buffer window approach aligns well with the straightforward nature of FAQ interactions. It avoids the additional complexity and token usage of summarisation, which is an essential consideration given the brevity of FAQ sessions.

### Types of model used

1. [PEFT Model](https://huggingface.co/matrixavenger/rescalelab) and [Base Model](https://huggingface.co/tiiuae/falcon-7b-instruct)

   Input Prompt Example 1:

   ```bash
   prompt = """ <user>: explain to me what can rescalelab do that benefits students? <ai>: """
   ```

   Output 1:

   ```bash
   <user>: Explain to me what can rescalelab do that benefits students?
   <ai>: RescaleLab offers experiential learning opportunities that enhance academic, professional, and personal development, responding to the demand for human-centered learning experiences. We focus on 'lab' experiences to train future innovators, encouraging hands-on learning opportunities that complement traditional academic programs.
   <link>: Learn more at rescalelab.com
   ```

   Input Prompt Example 2:

   ```bash
   prompt = """ <user>: What are some pain points of trainers that that rescalelab can solve? <ai>: """
   ```

   Output 2:

   ```bash
   <user>: What are some pain points of trainers that that rescalelab can solve?
   <ai>: Some pain points include limited training opportunities, lack of professional development, and challenges with retention rates. Our platform addresses these issues through innovative solutions and services for trainers.
   <lab>: We focus on product development, training services, and platform development, with a focus on delivering quality and results for trainers.
   ```

   However, due to GPU limitations and lack of cloud infrastructure, I was unable to test it together with Langchain to build a chatbot. Hence, I decided to go with prompt engineering by Chain of Thought.

2. Using Chain of Thought and Retrieval Augmented Generation

   I created embeddings of private data with OPENAI embeddings and store them in vector database in order to carry out RAG.

   Given user prompts, the vector database will find similar FAQs and provide the top 3 FAQs for context awareness.

   By giving context about the conversation and a set of instructions, we can restrict the chatbot to act as an assistant for a company or product. This acheived similar results compared to finetuning a model.

   Input Prompt Example 1:

   ```bash
   prompt = """ <user>: explain to me what can rescalelab do that benefits students? <ai>: """
   ```

   Output 1:

   ```bash
   <user>: Explain to me what can rescalelab do that benefits students?
   <ai>: Absolutely! At Rescale Lab, we offer a variety of services that can greatly benefit students in their academic pursuits.
   Our cloud-based simulation platform allows students to access high-performance computing resources without needing expensive hardware or software.
   This means that students can run complex simulations for engineering, science, and mathematics projects at a much faster pace than they could on their personal computers.
   It also enables them to collaborate more easily with classmates and instructors by sharing project data and results through our secure platform.

   Additionally, Rescale Lab offers educational discounts for students, making these powerful computational tools accessible to those who might not otherwise be able to afford them.
   By leveraging our technology, students can gain valuable experience working with cutting-edge tools used by professionals in industry and research settings.
   Overall, Rescale Lab helps equip students with the skills and expertise necessary to excel in today's rapidly evolving technological landscape.
   ```

   Input Prompt Example 2:

   ```bash
   prompt = """ <user>: What are some pain points of trainers that that rescalelab can solve? <ai>: """
   ```

   Output 2:

   ```bash
   <user>: What are some pain points of trainers that that rescalelab can solve?
   <ai>: Certainly! Trainers often face challenges when it comes to providing hands-on experience with advanced technologies due to limited resources or outdated equipment.
   Rescale Lab addresses these issues by offering a state-of-the-art cloud-based simulation platform that requires no upfront investment in costly hardware or software.
   With Rescale Lab, trainers can provide easy access to high-performance computing resources for their learners, enabling them to perform complex calculations and simulations seamlessly.
   Rescalable's flexible licensing options make it easier for trainers to manage costs while still delivering top-notch education.
   Additionally, our platform simplifies collaboration among trainees and instructors, facilitating better communication and feedback loops during projects.
   In summary, Rescale Lab solves several pain points for trainers by offering affordable, adaptable, and efficient simulation solutions tailored to modern teaching requirements.
   ```

## Installation

Try it [here](https://pychatbot.streamlit.app)!

If you want to try it out and modify it:

1. Clone the Repository

```bash
git clone https://github.com/timooo-thy/pyChatBot.git
```

2. Install Dependencies
   Ensure you have Python installed, and then run:

```bash
pip install -r requirements.txt
```

3. Launch the App
   Navigate to the app's directory and run:

```bash
streamlit run rescalelabChat.py
```

## Usage

**Login**: Enter your Huggingface credentials at the sidebar.

**Start Chatting**: Simply type your query or message in the input field and press Enter.

**Manage Sessions**: Start new sessions or review past conversations for reference.

**Download Conversations**: Download the transcript of your current or past conversations for record-keeping.

## Acknowledgements

- Huggingface for the LLM model (meta-llama/Llama-2-70b-chat-hf & tiiuae/falcon-7b-instruct)

- Transformers and PEFT to train model.

- Streamlit for providing an intuitive app-building framework.

- Langchain for retrieval augmented generation.

- OpenAI for text embeddings.

- FAISS for vector store and similarity search.
