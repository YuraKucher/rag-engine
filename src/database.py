import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm  # Бібліотека для прогрес-бару


class VectorDatabase:
    def __init__(self, model_name="phi3:latest", folder_path="vector_store"):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.folder_path = folder_path

    def create_from_documents(self, documents):
        """Створює базу FAISS з візуалізацією прогресу."""
        print(f"🧠 Починаємо векторазацію ({len(documents)} фрагментів)...")

        # Створюємо базу з першого шматочка, щоб ініціалізувати індекс
        vector_db = FAISS.from_documents([documents[0]], self.embeddings)

        # Додаємо решту документів через цикл із tqdm
        for i in tqdm(range(1, len(documents)), desc="Індексація", unit="chunk"):
            vector_db.add_documents([documents[i]])

        vector_db.save_local(self.folder_path)
        print(f"✅ Готово! Збережено в {self.folder_path}")
        return vector_db

    def load_local(self):
        if os.path.exists(self.folder_path):
            return FAISS.load_local(
                self.folder_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return None