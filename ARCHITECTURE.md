# Архітектурний документ RAG Engine

## 1. Призначення документа

Цей документ фіксує **формальну архітектуру локальної RAG-системи (Retrieval-Augmented Generation)**, яка розробляється як **повноцінний програмний продукт**, а не демонстраційний прототип.

Документ є **канонічною точкою відліку**:

* до нього можна і потрібно повертатися
* будь-який код у репозиторії має відповідати цій архітектурі
* всі майбутні розширення (evaluation, learning, UI, installer) мають логічно вкладатися у цей фундамент

---

## 2. Цілі системи

### Функціональні цілі

* Локальна робота (offline-first)
* Робота з приватними документами
* Пояснювані, обґрунтовані відповіді
* Підтримка різних джерел знань
* Можливість оцінювання та навчання

### Нефункціональні цілі

* Стійкість до помилок та шуму
* Масштабованість
* Контроль якості відповідей
* Відтворюваність результатів
* Модульність і замінність компонентів

---

## 3. Загальна архітектура (High-Level)

```
Ingestion
   ↓
Knowledge Storage  ← source of truth
   ↓
Indexing
   ↓
Retrieval
   ↓
Reasoning (RAG / Agent)
   ↓
Generation
   ↓
Evaluation
   ↓
Learning Loop
```

Evaluation та Learning є **обовʼязковими частинами ядра**, а не додатковими модулями.

---

## 4. Компоненти системи

### 4.1 Ingestion Layer

**Призначення:**

* Перетворити будь-яке джерело (PDF, DOCX, Web, Notes) у уніфікований формат документа

**Вхід:**

* сирі джерела даних

**Вихід:**

* NormalizedDocument

**Контракт (логічний):**

```json
{
  "source_id": "uuid",
  "source_type": "pdf | docx | web | note",
  "source_path": "...",
  "checksum": "...",
  "metadata": {
    "title": "...",
    "author": "...",
    "created_at": "..."
  }
}
```

LangChain або інші бібліотеки **не є частиною контракту**, лише інструментами реалізації.

---

### 4.2 Knowledge Storage Layer (Source of Truth)

**Призначення:**

* Єдине достовірне сховище знань

**Принцип:**

> FAISS — це індекс, а не база знань

Knowledge Storage зберігає:

* документи
* чанки
* метадані
* звʼязки
* версії

**Chunk — базова одиниця знань**

**Контракт чанку:**

```json
{
  "chunk_id": "uuid",
  "source_id": "uuid",
  "text": "...",
  "position": {
    "page": 12,
    "offset": 340
  },
  "metadata": {
    "section": "2.3",
    "language": "uk"
  },
  "quality": {
    "length": 487,
    "structure_score": 0.82
  }
}
```

Реалізація: JSON / SQLite / DuckDB (визначається пізніше).

---

### 4.3 Indexing Layer

**Призначення:**

* Створення похідних представлень знань (embeddings, індекси)

**Вхід:**

* chunks із Knowledge Storage

**Вихід:**

* векторні індекси (FAISS або аналоги)

**Index Metadata (обовʼязково):**

```json
{
  "index_id": "uuid",
  "embedding_model": "nomic-embed-text",
  "chunking_policy": "v1_recursive_500_50",
  "created_at": "...",
  "dimension": 768
}
```

Цей шар дозволяє:

* переіндексацію
* A/B testing
* зміну embedding моделей

---

### 4.4 Retrieval Layer

**Призначення:**

* Пошук релевантних знань для питання

**Ключова ідея:**

> Retriever — це політика, а не виклик similarity_search

**Контракт:**

```
retrieve(query, context) → List[Chunk]
```

**Внутрішня логіка Retriever:**

* переписування запиту
* вибір індексу
* адаптивний вибір k
* reranking
* пороги релевантності

Retriever є **самостійним компонентом**, незалежним від UI та LLM.

---

### 4.5 Reasoning / RAG Layer

**Призначення:**

* Прийняття рішення, *як відповідати*

Функції:

* перевірка достатності контексту
* multi-step retrieval
* уточнюючі запити
* вибір стратегії генерації

Це **agent**, а не простий chain.

---

### 4.6 Generation Layer

**Призначення:**

* Генерація відповіді на основі питання та контексту

**Контракт:**

```
Question + Context → Answer
```

Вимоги до відповіді:

* обґрунтованість (grounded)
* простежуваність (traceable)
* мінімум галюцинацій

Generation layer **не знає** нічого про FAISS, індекси чи retriever.

---

### 4.7 Evaluation Layer

**Призначення:**

* Оцінювання якості кожної відповіді

**Мінімальні метрики:**

```json
{
  "relevance": 0.0,
  "groundedness": 0.0,
  "answerability": true
}
```

Evaluation є обовʼязковою умовою для навчання та покращення системи.

---

### 4.8 Learning Layer

**Призначення:**

* Покращення системи на основі оцінювання та фідбеку

Навчання відбувається **не шляхом тренування LLM**, а через:

* оновлення retriever policy
* зміну ваг чанків
* адаптацію rerank порогів
* query rewriting

Learning Layer замикає feedback loop.

---

## 5. Принципи розвитку системи

* Дані важливіші за код
* Контракти важливіші за реалізацію
* Оцінювання — не опція
* Кожен компонент має бути замінним
* LangChain — інструмент, а не архітектура

---

**Цей документ є живим і може доповнюватися, але базові блоки вважаються зафіксованими.**

## Схема файлів репозиторію
  rag-engine/
│
├─ README.md
├─ ARCHITECTURE.md
├─ pyproject.toml / requirements.txt
├─ .gitignore
│
├─ config/
│   ├─ system.yaml
│   ├─ models.yaml
│   ├─ thresholds.yaml
│   └─ paths.yaml
│
├─ schemas/
│   ├─ document.schema.json
│   ├─ chunk.schema.json
│   ├─ index.schema.json
│   ├─ evaluation.schema.json
│   └─ feedback.schema.json
│
├─ core/
│   │
│   ├─ cache/
│   │   ├─ __init__.py
│   │   ├─ manager.py
│   │   └─ semantic_cache.py
│   │
│   ├─ ingestion/
│   │   ├─ __init__.py
│   │   ├─ base_loader.py
│   │   ├─ pdf_loader.py
│   │   └─ registry.py
│   │
│   ├─ knowledge/
│   │   ├─ __init__.py
│   │   ├─ document_store.py
│   │   ├─ chunk_store.py
│   │   ├─ metadata.py
│   │   └─ chunker.py
│   │  
│   │
│   ├─ indexing/
│   │   ├─ __init__.py
│   │   ├─ embedder.py
│   │   ├─ index_manager.py
│   │   └─ faiss_index.py
│   │
│   ├─ retrieval/
│   │   ├─ __init__.py
│   │   ├─ retriever.py
│   │   ├─ query_rewriter.py
│   │   ├─ reranker.py
│   │   └─ policies.py
│   │
│   ├─ reasoning/
│   │   ├─ __init__.py
│   │   ├─ agent.py
│   │   ├─ context_builder.py
│   │   └─ strategies.py
│   │
│   ├─ generation/
│   │   ├─ __init__.py
│   │   ├─ llm_client.py
│   │   └─ prompts.py
│   │
│   ├─ evaluation/
│   │   ├─ __init__.py
│   │   ├─ evaluator.py
│   │   ├─ relevance.py
│   │   ├─ groundedness.py
│   │   └─ answerability.py
│   │
│   ├─ learning/
│   │   ├─ __init__.py
│   │   ├─ feedback_store.py
│   │   ├─ trainer.py
│   │   └─ policies_update.py
│   │
│   └─ logging/
│       ├─ __init__.py
│       ├─ event_logger.py
│       └─ trace.py
│
├─ services/
│   ├─ rag_service.py
│   ├─ indexing_service.py
│   └─ evaluation_service.py
│
├─ ui/
│   ├─ streamlit_app.py
│   └─ components/
│
├─ data/
│   ├─ documents/
│   ├─ chunks/
│   ├─ indexes/
│   ├─ evaluations/
│   └─ feedback/
│
└─ tests/
    ├─ test_retrieval.py
    ├─ test_evaluation.py
    └─ test_learning.py