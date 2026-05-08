import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from neo4j_graphrag.llm.base import LLMInterface
from langchain_groq import ChatGroq


load_dotenv()


class LLMResponse:
    def __init__(self, content: str):
        self.content = content

class GroqLangChainLLM(LLMInterface):

    def __init__(self):
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile",
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