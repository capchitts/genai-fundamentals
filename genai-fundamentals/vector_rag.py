import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from sentence_transformers import SentenceTransformer
from neo4j_graphrag.llm.base import LLMInterface
from langchain_groq import ChatGroq
import os


class LLMResponse:
    def __init__(self, content: str):
        self.content = content

class GroqLangChainLLM(LLMInterface):

    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
        )

    def invoke(self, input: str, **kwargs):
        result = self.llm.invoke(input)
        return LLMResponse(result.content)

    async def ainvoke(self, input: str, **kwargs):
        return self.invoke(input, **kwargs)

class SentenceTransformerEmbedder:
    def __init__(self, model_name="Orange/orange-nomic-v1.5-1536"):
        self.model = SentenceTransformer(model_name,trust_remote_code=True)

    def embed_query(self, text):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts):
        return [self.model.encode(text).tolist() for text in texts]


# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = SentenceTransformerEmbedder()


# Create retriever
retriever = VectorRetriever(
    driver,
    neo4j_database=os.getenv("NEO4J_DATABASE"),
    index_name="moviePlots",
    embedder=embedder,
    return_properties=["title", "plot"],
)


# Create the LLM
llm = GroqLangChainLLM()


# Create GraphRAG pipeline
# tag::graphrag[]
# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)
# end::graphrag[]

# # tag::search[]
# # Search
# query_text = "Find me movies about toys coming alive"

# response = rag.search(
#     query_text=query_text, 
#     retriever_config={"top_k": 5}
# )

# print(response.answer)
# # end::search[]

# tag::search_return_context[]
# Search
query_text = "Find me movies about toys coming alive"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)
# end::search_return_context[]

# Close the database connection
driver.close()