"""RAG pipeline using LlamaIndex, Weaviate, and Azure Files.

Ingests documents from an Azure file share, converts files to LlamaIndex
Documents, indexes into Weaviate, and provides an interactive Q&A loop.
"""

import logging
import os
import tempfile
from pathlib import Path


logging.disable(logging.CRITICAL)

import weaviate
from llama_index.core import Document, PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import CSVReader, DocxReader, PDFReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from weaviate.classes.init import Auth

from azure_files import DownloadedFile, connect_to_share, download_files, list_share_files
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv()

# Azure Storage
STORAGE_ACCOUNT_NAME = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
SHARE_NAME = os.environ["AZURE_STORAGE_SHARE_NAME"]

# Azure OpenAI
OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
OPENAI_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
OPENAI_CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# RAG tuning
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

# Azure authentication
CREDENTIAL = DefaultAzureCredential()
TOKEN_PROVIDER = get_bearer_token_provider(
    CREDENTIAL, "https://cognitiveservices.azure.com/.default"
)

# REST endpoint from the Weaviate Cloud console (not the gRPC endpoint).
# Example: https://abc123.c0.us-east1.gcp.weaviate.cloud
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "AzureFilesRAG")

# File types that need specialized LlamaIndex readers.
# Everything not listed here falls back to plain-text reading.
LOADER_MAP: dict[str, object] = {
    ".pdf": PDFReader(),
    ".docx": DocxReader(),
    ".csv": CSVReader(),
}


def parse_downloaded_files(
    downloaded_files: list[DownloadedFile],
) -> list[Document]:
    """Parse downloaded files from an Azure file share into LlamaIndex Documents.

    Args:
        downloaded_files: A list of DownloadedFile objects containing the path
            and metadata for each file.

    Returns:
        A list of LlamaIndex Documents.
    """
    documents = []

    for info in downloaded_files:
        ext = os.path.splitext(info.file_name.lower())[1]
        metadata = {
            "azure_file_path": info.relative_path,
            "file_name": info.file_name,
        }

        loader = LOADER_MAP.get(ext)
        try:
            if loader:
                docs = loader.load_data(Path(info.local_path), extra_info=metadata)
            else:
                text = Path(info.local_path).read_text(encoding="utf-8")
                docs = [Document(text=text, metadata=metadata)]
        except Exception:
            print(f"Failed to parse {info.relative_path}, skipping...")
            continue

        for doc in docs:
            doc.metadata.update(metadata)

        documents.extend(docs)

    return documents


def chunk_documents(documents: list[Document]) -> list[TextNode]:
    """Split documents into overlapping text nodes for embedding.

    Args:
        documents: The Documents to split.

    Returns:
        A list of TextNode chunks with preserved metadata.
    """
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.get_nodes_from_documents(documents)


def embed_and_index(
    nodes: list[TextNode],
) -> tuple[VectorStoreIndex, weaviate.WeaviateClient]:
    """Embed text nodes via Azure OpenAI and index into Weaviate.

    Args:
        nodes: TextNode chunks to embed and index.

    Returns:
        A tuple of (VectorStoreIndex, Weaviate client). The caller is
        responsible for closing the Weaviate client.
    """
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

    # Clear existing collection if RESET_INDEX is set
    if os.getenv("RESET_INDEX") == "true" and client.collections.exists(
        WEAVIATE_COLLECTION_NAME
    ):
        client.collections.delete(WEAVIATE_COLLECTION_NAME)

    vector_store = WeaviateVectorStore(
        weaviate_client=client,
        index_name=WEAVIATE_COLLECTION_NAME,
    )

    embed_model = AzureOpenAIEmbedding(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        use_azure_ad=True,
        model="text-embedding-3-small",
        dimensions=EMBEDDING_DIMENSIONS,
        api_version="2024-06-01",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    return index, client


def build_query_engine(index: VectorStoreIndex):
    """Create a LlamaIndex query engine over a Weaviate-backed index.

    Args:
        index: The VectorStoreIndex to query against.

    Returns:
        A LlamaIndex query engine.
    """
    llm = AzureOpenAI(
        engine=OPENAI_CHAT_DEPLOYMENT,
        model="gpt-4o-mini",
        azure_endpoint=OPENAI_ENDPOINT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        use_azure_ad=True,
        api_version="2024-06-01",
    )
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        text_qa_template=PromptTemplate(
            "Answer the question based on the context below. "
            "Be specific and cite the source file name in brackets for each fact.\n\n"
            "Context:\n{context_str}\n\n"
            "Question: {query_str}\n\nAnswer:"
        ),
    )


def main():
    """Main execution flow."""
    share = connect_to_share(STORAGE_ACCOUNT_NAME, SHARE_NAME, CREDENTIAL)

    # 1. List files from the share
    print("Scanning file share...")
    file_references = list_share_files(share)
    if not file_references:
        print("No files found.")
        return
    print(f"Found {len(file_references)} files.\n")

    # 2. Download files (shared Azure Files logic)
    print("Downloading files onto temporary local directory...")
    with tempfile.TemporaryDirectory() as temp_directory:
        downloaded = download_files(file_references, temp_directory)
        if not downloaded:
            print("No files downloaded.")
            return
        print()

        # 3. Parse into LlamaIndex Documents
        print("Parsing files...")
        documents = parse_downloaded_files(downloaded)

    if not documents:
        print("No documents parsed.")
        return
    print(f"{len(documents)} documents.\n")

    # 4. Chunk
    print("Splitting into chunks...")
    nodes = chunk_documents(documents)
    print(f"{len(documents)} docs → {len(nodes)} chunks.\n")

    # 5. Embed and index
    print("Indexing into Weaviate...")
    index, weaviate_client = embed_and_index(nodes)
    print(f"{len(nodes)} chunks indexed.\n")

    query_engine = build_query_engine(index)
    print("Ready. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            response = query_engine.query(question)
            print(f"\nAnswer: {response}\n")
    except KeyboardInterrupt:
        pass
    finally:
        weaviate_client.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
