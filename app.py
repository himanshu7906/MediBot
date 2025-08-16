

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file , text_split , download_hugging_face_embeddings
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import OpenAI
from flask import Flask , render_template , jsonify , request
from dotenv import load_dotenv
import os
from src.prompt import *

load_dotenv()

app = Flask(__name__)



PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MediBot_API_KEY = os.getenv("MediBot_API_KEY")


embeddings = download_hugging_face_embeddings()

index_name = "medi-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",  
    search_kwargs={"k": 3}
)

llm = OpenAI(
    openai_api_key=MediBot_API_KEY,
    temperature=0,
     max_tokens = 500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , system_prompt),
        ("human" , "{input}") ,

    ]
)

question_answers_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answers_chain)

@app.route("/")
def index ():
    return render_template("chatbot.html")


@app.route("/get" , methods=["GET" , "POST"])
def chatting():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input" : msg})
    print("Response :" , response["answer"] )
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host = "0.0.0.0" , port = 5000 , debug = True)