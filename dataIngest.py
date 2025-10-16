import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "./web_pages"
CHROMA_DIR = "./chroma_store"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Collect all text files
documents = []

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunk_count = 0

print(f"Loading documents from {DATA_DIR}...")

for file_name in os.listdir(DATA_DIR):
    if file_name.endswith(".txt"):
        file_path = os.path.join(DATA_DIR, file_name)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                # Split the content into chunks
                chunks = text_splitter.split_text(content)

                # Create Document objects with metadata
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": file_name,
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
                    chunk_count += 1

                print(f"{file_name}: {len(chunks)} chunks")

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

# Create or load Chroma index
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR
)

vectorstore.persist()
print(f"✅ Chroma vector store created and persisted at: {CHROMA_DIR}")