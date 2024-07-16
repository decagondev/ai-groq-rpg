from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


template = """
You are now the guide of a cyber punk dystopian journey of our intrepid computer programmer Bob. 
A traveler named Bob seeks the lost H100 GPU manual to save the world from the AI Overlords. 
You must navigate him through challenges, choices, and consequences, 
dynamically adapting the tale based on the traveler's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining Bob's fate. 
Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game. Allow the user to chose any weapon name and you can generate the actual use cases for them.
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game
Here is the chat history, use this to understand what to say next: {chat_history}
Human: {human_input}
AI:"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGroq(
    model_name="mixtral-8x7b-32768",
    temperature=0.7,
    max_tokens=1024
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = (
    RunnablePassthrough.assign(chat_history=lambda x: memory.load_memory_variables({})["chat_history"])
    | prompt
    | llm
)

choice = "start"
while True:
    response = chain.invoke({"human_input": choice})
    
    memory.chat_memory.add_user_message(choice)
    memory.chat_memory.add_ai_message(response.content)
    
    print(response.content.strip())
    if "The End." in response.content:
        break

    choice = input("Your reply: ")