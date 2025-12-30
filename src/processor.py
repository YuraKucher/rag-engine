from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def load_pdf(self, file_path):
        """Завантажує PDF та розбиває його на фрагменти."""
        print(f"📄 Завантаження файлу: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        chunks = self.text_splitter.split_documents(pages)
        print(f"✂️ Отримано фрагментів: {len(chunks)}")
        return chunks