# RAG-система на текстах из Википедии

Полнофункциональная система RAG (Retrieval-Augmented Generation) для русскоязычных текстов из Википедии с использованием современных технологий машинного обучения.

## Быстрый старт

### Предварительные требования
- Docker и Docker Compose
- Python 3.11+ (для разработки)
- OpenAI API ключ

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd rag-system

# Создайте .env файл с вашим OpenAI API ключом
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
echo "SEARCH_DB=chroma" >> .env
```

### 2. Запуск системы

```bash
# Запуск всех сервисов
docker compose up -d

# Инициализация базы данных (если не выполнена)
python scripts/create_chroma_db.py
```

### 3. Проверка готовности

```bash
# Проверка статуса сервисов
docker compose ps

# Тест API
curl -X POST "http://localhost:8081/invoke" \
  -H "Content-Type: application/json" \
  -d '{"message": "Расскажи о истории России"}'
```

### 4. Доступ к интерфейсам

- **Streamlit Web App**: http://localhost:8502
- **Agent API**: http://localhost:8081
- **Phoenix Monitoring**: http://localhost:8881
- **PostgreSQL**: localhost:5440

## Архитектура системы

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   PostgreSQL    │
│   Frontend      │◄──►│   Agent Service │◄──►│   Database      │
│   Port: 8502    │    │   Port: 8081    │    │   Port: 5440    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   ChromaDB      │    │   Phoenix       │
                    │   Vector Store  │    │   Monitoring    │
                    │                 │    │   Port: 8881    │
                    └─────────────────┘    └─────────────────┘
```

## RAG Implementation Details

### Компоненты RAG системы

#### 1. **Enhanced Retriever** (`src/agents/enhanced_retriever.py`)
Продвинутая система поиска с несколькими уровнями обработки:

- **Vector Search**: Семантический поиск по векторным представлениям
- **Relevance Filtering**: Фильтрация по порогу релевантности (>0.5)
- **Deduplication**: Удаление дублирующихся фрагментов

```python
# Основные параметры
chunk_size = 800           # Размер текстовых фрагментов
chunk_overlap = 400        # Перекрытие между фрагментами
base_k = 10               # Количество кандидатов для поиска
min_score_threshold = 0.5  # Минимальный порог релевантности
```

#### 2. **Vector Store** (ChromaDB)
- **Database**: ChromaDB с SQLite backend
- **Embeddings Model**: `deepvk/USER-base` (оптимизирована для русского языка)
- **Collection**: `ru_rag_collection`
- **Document Count**: 49,348 фрагментов
- **Data Source**: Русскоязычные статьи Wikipedia

#### 3. **RAG Agent** (`src/agents/rag_assistant.py`)
LangGraph-основанный агент с возможностями:

- **Multi-turn Conversations**: Поддержка многоходовых диалогов
- **Tool Integration**: Интеграция с инструментом поиска в базе знаний
- **Safety Checks**: Проверка безопасности с LlamaGuard
- **Memory Management**: Управление историей разговора
- **Streaming Support**: Потоковая генерация ответов

#### 4. **Database Search Tool** (`src/agents/tools.py`)
Специализированный инструмент для поиска в базе знаний:

```python
def database_search_func(query: str) -> str:
    """Поиск информации в базе знаний на русском языке с улучшенным ранжированием."""
    retriever = load_enhanced_retriever()
    documents = retriever.retrieve(query, k=5)
    return format_contexts(documents)
```

### Алгоритм работы RAG

1. **Прием запроса**: Пользователь отправляет вопрос на русском языке
2. **Vector Search**: Поиск релевантных документов в ChromaDB по векторному сходству
3. **Relevance Filtering**: Фильтрация результатов по порогу релевантности
4. **Context Assembly**: Сборка контекста из найденных фрагментов
5. **LLM Generation**: Генерация ответа с использованием GPT-4
6. **Response Streaming**: Потоковая передача ответа пользователю

### Особенности реализации

- **Russian Language Optimization**: Использование специализированных эмбеддингов для русского языка (deepvk/USER-base)
- **Intelligent Chunking**: Разбиение текста с учетом структуры документа
- **Relevance Scoring**: Ранжирование результатов поиска по векторному сходству

## Dataset Information

### Russian RAG Test Dataset
- **Source**: Статьи из русскоязычной Wikipedia
- **Format**: Pickle файл с метаданными
- **Test Entries**: 923 записи для оценки качества
- **Text Files**: Обработанные статьи в формате .txt
- **Total Chunks**: 49,348 индексированных фрагментов

### Data Processing Pipeline
```python
# Обработка текстов
RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", "]
)

# Генерация эмбеддингов
HuggingFaceEmbeddings(
    model_name="deepvk/USER-base",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

## API Endpoints

### Agent Service (Port 8081)

#### POST `/invoke`
Однократный запрос к RAG системе:
```json
{
    "message": "Расскажи о истории России",
    "model": "gpt-4o-2024-08-06",
    "thread_id": "optional-thread-id"
}
```

#### POST `/stream`
Потоковый запрос с получением ответа в реальном времени:
```json
{
    "message": "Что такое машинное обучение?",
    "stream_tokens": true
}
```

#### GET `/info`
Метаданные сервиса:
```json
{
    "agents": ["rag_assistant"],
    "models": ["gpt-4o-2024-08-06", "gpt-3.5-turbo"],
    "default_agent": "rag_assistant",
    "default_model": "gpt-4o-2024-08-06"
}
```

#### POST `/history`
Получение истории разговора:
```json
{
    "thread_id": "conversation-id"
}
```

#### GET `/health`
Проверка состояния сервиса:
```json
{
    "status": "ok"
}
```

## Configuration

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_openai_api_key_here
SEARCH_DB=chroma
MODE=development
DEV_MODE=true
POSTGRES_HOST=bread-postgres
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=agent_service
PHOENIX_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://bread-phoenix:4317
OTEL_SERVICE_NAME=bread-agent-service
```

### Service Configuration
- **LLM Models**: Поддерживаются различные модели OpenAI
- **Retrieval Parameters**: Настраиваемые параметры поиска
- **Safety Filters**: Опциональные фильтры безопасности
- **Monitoring**: Интеграция с Phoenix для мониторинга

## Development Setup

### Local Development
```bash
# Установка зависимостей
pip install -r requirements.txt

# Локальный запуск сервиса
python src/run_service.py

# Локальный запуск Streamlit
streamlit run src/streamlit_app.py
```

### Testing
```bash
# Тестирование RAG функциональности
python -c "
from src.agents.enhanced_retriever import EnhancedRetriever
retriever = EnhancedRetriever('./data/chroma_db', 'ru_rag_collection')
results = retriever.retrieve('история России', k=3)
print(f'Found {len(results)} results')
"
```

## Monitoring

### Phoenix Integration
- **Tracing**: Отслеживание запросов и ответов
- **Metrics**: Метрики производительности
- **Debugging**: Инструменты отладки RAG pipeline
- **Feedback**: Система обратной связи

### Logs and Debugging
```bash
# Просмотр логов сервисов
docker compose logs -f bread-service
docker compose logs -f bread-streamlit

# Мониторинг базы данных
docker compose logs -f bread-postgres
```

## Performance Optimization

### Retrieval Optimization
- **Indexing**: Предварительная индексация всех документов в ChromaDB

### Response Generation
- **Streaming**: Потоковая генерация для лучшего UX
- **Context Limiting**: Ограничение контекста для оптимизации токенов
- **Model Selection**: Выбор оптимальной модели для задачи

## Acknowledgments

- **deepvk/USER-base**: Русскоязычная модель эмбеддингов
- **ChromaDB**: Векторная база данных
- **LangChain**: Фреймворк для LLM приложений
- **LangGraph**: Граф-основанные агенты
- **Streamlit**: Web интерфейс
- **Phoenix**: Мониторинг и отладка LLM
