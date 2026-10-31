"""Create the Chroma collections from the committed source data when needed."""
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import CHROMA_DIR
from ingest_faq import COLLECTION as FAQ_COLLECTION
from ingest_faq import CSV_PATH, load_faq_documents
from ingest_pdf import COLLECTION as GUIDE_COLLECTION
from ingest_pdf import PDF_PATH
from ingest_pdf import CHUNK_OVERLAP, CHUNK_SIZE
from ingest_pdf import RecursiveCharacterTextSplitter, PyPDFLoader
from ingest_tickets import COLLECTION as TICKET_COLLECTION
from ingest_tickets import DB_PATH, load_ticket_documents

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
REQUIRED_COLLECTIONS = {FAQ_COLLECTION, TICKET_COLLECTION, GUIDE_COLLECTION}


def _collection_names() -> set[str]:
    if not CHROMA_DIR.exists():
        return set()
    client = PersistentClient(path=str(CHROMA_DIR))
    return {collection.name for collection in client.list_collections()}


def _load_guide_documents():
    pages = PyPDFLoader(PDF_PATH).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(pages)
    for index, chunk in enumerate(chunks):
        chunk.metadata["source"] = "guide"
        chunk.metadata["chunk_index"] = index
    return chunks


def ensure_vector_store() -> None:
    missing = REQUIRED_COLLECTIONS - _collection_names()
    if not missing:
        return

    print(f"Building missing Chroma collections: {', '.join(sorted(missing))}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    documents_by_collection = {
        FAQ_COLLECTION: load_faq_documents(CSV_PATH),
        TICKET_COLLECTION: load_ticket_documents(DB_PATH),
        GUIDE_COLLECTION: _load_guide_documents(),
    }

    for collection_name in sorted(missing):
        Chroma.from_documents(
            documents=documents_by_collection[collection_name],
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DIR),
        )
    print("Chroma vector store is ready.")
