from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain


class AIBrain:
    def __init__(self, model_name="phi3:latest"):
        self.llm = ChatOllama(model=model_name, temperature=0, streaming=True)

    def rerank_documents(self, query, documents, model, top_n=3):
        # Перевірка: чи є документи взагалі
        if not documents:
            return []

        # Створюємо пари для ML-моделі
        pairs = [[query, doc.page_content] for doc in documents]

        # Використовуємо передану модель для передбачення
        scores = model.predict(pairs)

        # Додаємо оцінки до метаданих
        for i, doc in enumerate(documents):
            doc.metadata['relevance_score'] = float(scores[i])

        # Сортуємо
        reranked = sorted(documents, key=lambda x: x.metadata['relevance_score'], reverse=True)
        return reranked[:top_n]

    def get_qa_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a professional assistant. Use ONLY provided context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("system", "Context: {context}"),
            ("human", "{input}"),
        ])
        return create_stuff_documents_chain(self.llm, prompt)