from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
MediBot_API_KEY = os.getenv("MediBot_API_KEY")
os.environ["MediBot_API_KEY"] = MediBot_API_KEY
extracted_data = load_pdf_file(data="data/")
text_chunks = text_split(extracted_data)  # Should return List[Document]
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medi-bot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)


docsearch = PineconeVectorStore.from_documents(
    text_chunks,  # <-- passed as positional
    embedding=embeddings,
    index_name=index_name
)


# docsearch = PineconeVectorStore.from_existing_index(
#     index_name=index_name,
#     embedding=embeddings
# )

# retriever = docsearch.as_retriever(
#     search_type="similarity",  
#     search_kwargs={"k": 3}
# )

# system_prompt = (
#     "You  are an assistant for question answering.  tasks"
#     "Use the following pieces of relevent context to answer"
#     "the  question , If you don't know the answer , just say that you don't know "
#     "Use three sentences maximum and  keep the answer as concise as possible."\
#     "\n\n"
#     "{context}"
#     )

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system" , system_prompt),
#         ("human" , "{input}") ,

#     ]
# )

