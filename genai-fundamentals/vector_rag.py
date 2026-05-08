import os
from dotenv import load_dotenv
from utils import *
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG

load_dotenv()



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