from openai import OpenAI
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings


f = open('.demo_key.txt')

open_ai_key = f.read()

embeddings_model = OpenAIEmbeddings(openai_api_key=open_ai_key)


db_connection = Chroma(persist_directory="./chroma", embedding_function=embeddings_model)



st.title('Ask me anything about the paper, \'Leave No Context Behind:Efficient Infinite Context Transformers with Infini-attention\'')
#st.subheader('Enter your code here....')

user_input = st.text_input('Enter your question here....')
out = db_connection.similarity_search(user_input)

if st.button('Generate')==True:
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", api_key=open_ai_key, temperature=0.1)

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. Give them in clear bullet points if required, for clarity.
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(chat_model, question_answering_prompt)
    st.write(document_chain.invoke(
    {
        "context": out,
        "messages": [
            HumanMessage(content=user_input)
        ],
    }))
    