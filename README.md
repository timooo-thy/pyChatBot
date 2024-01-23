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

- **Session Management**: Ability to start new chat sessions and review past conversations, offering a seamless user experience.

- **Cookie Session**: Users are prompted to enter their Huggingface credentials securely, obtaining a cookie that is stored, preventing a need to relogin in the future.

## Thought Process
### Types of memory in Langchain:
1. ConversationBufferMemory
2. ConversationSummaryMemory
3. ConversationBufferWindowMemory
4. ConversationSummaryBufferMemory
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
   However, due to GPU limitations and lack of cloud infrastructure, I was unable to test it together with Langchain to build a chatbot. Hence, I decided to go with advanced prompting.
   
3. Using advanced prompting
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
Login: Enter your Huggingface credentials at the sidebar.
Start Chatting: Simply type your query or message in the input field and press Enter.
Manage Sessions: Start new sessions or review past conversations for reference.
Download Conversations: Download the transcript of your current or past conversations for record-keeping.

## Acknowledgements
Huggingface for the LLM model (meta-llama/Llama-2-70b-chat-hf)
LangChain library for conversational AI functionalities.
Transformers and PEFT to train model
Streamlit for providing an intuitive app-building framework.
