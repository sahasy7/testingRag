from fastapi import FastAPI, HTTPException
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import Pinecone as pa
from langchain.chains import RetrievalQA
import os

app = FastAPI()

# Initialize Pinecone client
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
indices = pc.Index("gsm-demo1")

# Initialize OpenAIEmbeddings
os.environ['OPENAI_API_KEY'] = os.environ['openai']
embed = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize Pinecone as vector store
text_field = "text"
vectorstore = pa(indices, embed, text_field)

# Initialize ChatOpenAI
llm = ChatOpenAI(openai_api_key=os.environ['openai'],
                 model_name='gpt-3.5-turbo',
                 temperature=0.0)

# Initialize RetrievalQA
qa = RetrievalQA.from_chain_type(llm=llm,
                                 chain_type="stuff",
                                 retriever=vectorstore.as_retriever())


@app.get("/answer")
async def get_answer(query: str):
  try:
    # Invoke the query and return the answer
    ans = qa.invoke(query)
    return {"answer": ans["result"]}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
  return {"status": "ok"}
