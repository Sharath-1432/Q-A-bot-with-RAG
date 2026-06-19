from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import CHUNK_SIZE, CHUNK_OVERLAP, DB_DIR


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_documents():
    documents = []

    print(f"\nLooking for files in: {DATA_DIR}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    txt_files = list(DATA_DIR.glob("*.txt"))

    if len(txt_files) == 0:
        raise ValueError(f"No .txt files found inside {DATA_DIR}")

    print(f"Found {len(txt_files)} txt files")

    for file in txt_files:
        print(f"Loading: {file.name}")

        loader = TextLoader(
            str(file),
            encoding="utf-8"
        )

        documents.extend(loader.load())

    return documents


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)


def build_vector_db():
    print("Loading documents...")

    documents = load_documents()

    print(f"Loaded {len(documents)} documents")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks")

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Chroma database...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()

    print("\nVector DB created successfully!")
    print(f"Saved at: {DB_DIR}")


if __name__ == "__main__":
    build_vector_db()