import openai

import os
from dotenv import load_dotenv

# The .env file is actually not committed to the repo because of the gitignore
# just make a .env file and put this in it: OPENAI_API_KEY=your_api_key_here

# Load the .env file
load_dotenv()

# Now you can access the OPENAI_API_KEY as an environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

from typing import Any, List
from InstructorEmbedding import INSTRUCTOR

from llama_index.bridge.pydantic import PrivateAttr
from llama_index.embeddings.base import BaseEmbedding


class InstructorEmbeddings(BaseEmbedding):
    _model: INSTRUCTOR = PrivateAttr()
    _instruction: str = PrivateAttr()

    def __init__(
        self,
        instructor_model_name: str = "hkunlp/instructor-large",
        instruction: str = "Represent a document for semantic search:",
        **kwargs: Any,
    ) -> None:
        self._model = INSTRUCTOR(instructor_model_name)
        self._instruction = instruction
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "instructor"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, query]])
        return embeddings[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        embeddings = self._model.encode([[self._instruction, text]])
        return embeddings[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode([[self._instruction, text] for text in texts])
        return embeddings

from llama_index import ServiceContext, SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("data/").load_data()

service_context = ServiceContext.from_defaults(
    embed_model=InstructorEmbeddings(embed_batch_size=2), chunk_size=512
)

instructor_embeddings = InstructorEmbeddings(embed_batch_size=1)

embed = instructor_embeddings.get_text_embedding("How do I create a vector index?")
print(len(embed))
print(embed[:10])

# if running for the first time, will download model weights first!
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
response = index.as_query_engine().query("What did the author do growing up?")
print(response)