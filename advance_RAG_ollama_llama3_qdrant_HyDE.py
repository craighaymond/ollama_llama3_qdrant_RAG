# initial source:  https://blog.gopenai.com/improved-rag-with-llama3-and-ollama-c17dc01f66f6

# high-level overview of what the script does:
#
# Initializes the logging and Qdrant client.
# Loads the documents from a local directory and splits the text into chunks.
# Initializes the Ollama embedding model and global settings.
# Creates TextNodes for each text chunk and generates their embeddings.
# Stores the nodes and their embeddings in a VectorStoreIndex.
# Initializes a VectorIndexRetriever and a RetrieverQueryEngine.
# Transforms the query using HyDE (Hybrid Dense-sparse Embeddings) and queries the engine.
# Prints the response to the query.

# required installs
# pip install llama-index llama-index-vector-stores-qdrant
# pip install llama-index-embeddings-ollama
# pip install llama-index-llms-ollama
# ollama pull mxbai-embed-large

# required servers running prior to running this script
# start these in separate command prompts (or background both in same terminal)
# qdrant:  docker run -p 6333:6333 -v ./qdrant/storage qdrant/qdrant
# ollama:  ollama serve

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    get_response_synthesizer)
from llama_index.core.query_engine import RetrieverQueryEngine, TransformQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
import qdrant_client
import logging
import socket

# Initializations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load the local data directory and chunk the data for further processing
docs = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"]).load_data(show_progress=True)
text_parser = SentenceSplitter(chunk_size=512, chunk_overlap=100)

text_chunks = []
doc_ids = []
nodes = []

# Output: localhost
logger.info(socket.gethostbyname('localhost'))

# Create a local Qdrant vectore store to push embeddings
logger.info("initializing the vector store related objects")
client = qdrant_client.QdrantClient(host="localhost", port=6333)
logger.info("QdrantVectorStore")
vector_store = QdrantVectorStore(client=client, collection_name="research_papers")
logger.info("Done QdrantVectorStore")

# local vector embeddings and LLM model
logger.info("initializing the OllamaEmbedding")
embed_model = OllamaEmbedding(model_name='mxbai-embed-large', base_url='http://localhost:11434')
logger.info("initializing the global settings")
Settings.embed_model = embed_model
Settings.llm = Ollama(model="llama3", request_timeout=300.0, base_url='http://localhost:11434')
Settings.transformations = [text_parser]

# Creating the nodes, vector store, HyDE transformer and finally querying
logger.info("enumerating docs")
for doc_idx, doc in enumerate(docs):
    curr_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(curr_text_chunks)
    doc_ids.extend([doc_idx] * len(curr_text_chunks))

logger.info("enumerating text_chunks")
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(text=text_chunk)
    src_doc = docs[doc_ids[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

logger.info("enumerating nodes")
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode=MetadataMode.ALL)
    )
    node.embedding = node_embedding

logger.info("initializing the storage context")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
logger.info("indexing the nodes in VectorStoreIndex")
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    transformations=Settings.transformations,
)

logger.info("initializing the VectorIndexRetriever with top_k as 5")
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
response_synthesizer = get_response_synthesizer()
logger.info("creating the RetrieverQueryEngine instance")
vector_query_engine = RetrieverQueryEngine(
    retriever=vector_retriever,
    response_synthesizer=response_synthesizer,
)
logger.info("creating the HyDEQueryTransform instance")
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(vector_query_engine, hyde)

logger.info("About to retrieve the response to the query")
try:
    response = hyde_query_engine.query(
        str_or_query_bundle="summarize what Israel did.")
    logger.info("Successfully retrieved the response to the query")
except Exception as e:
    logger.error(f"Failed to retrieve the response to the query. Error: {e}")
print(response)

client.close()