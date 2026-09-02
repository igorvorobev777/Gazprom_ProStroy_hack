🏆 TOP-3 ЛУЧШИХ РЕШЕНИЙ ХАКАТОНА ГАЗПРОМА «КЕЙС-ЧЕМПИОНАТ PRO СТРОЙ 4.0»

Unified Adaptive RAG Service for Corporate Knowledge Base
Production-oriented RAG-сервис для интеллектуального поиска и генерации ответов по корпоративной базе знаний.

Python | RAG | Qwen3 | llama.cpp | gRPC | scikit-learn

Этот репозиторий содержит RAG/ML-сервис, разработанный как часть командного решения для хакатона Газпрома. Командное решение вошло в Top-3 лучших решений чемпионата, доказав свою эффективность в условиях строгих требований к latency, качеству ответов и локальной работе без внешних API.

О проекте
Проект представляет собой локальный Retrieval-Augmented Generation (RAG) сервис для работы с корпоративной базой знаний. Система получает документы из HiHub, строит локальный индекс, находит релевантные фрагменты по пользовательскому запросу и формирует ответ с привязкой к источникам.

Основной акцент был сделан не только на качестве retrieval, но и на практической эксплуатации модели:
- hybrid retrieval вместо одного способа поиска;
- reranking и дополнительные lexical-сигналы;
- адаптивное формирование контекста для LLM;
- контроль hallucinations через quality gate и grounding checks;
- fallback без генерации, если LLM недоступна или ответ нельзя надёжно подтвердить;
- ограничение latency и динамический token budget;
- безопасное обновление индекса без потери рабочей версии;
- gRPC-интерфейс для интеграции с backend-сервисом;
- диагностика latency каждого этапа RAG pipeline.

Таким образом, проект ближе к production ML/NLP service, чем к классическому notebook-прототипу.

Задача
В корпоративных базах знаний информация обычно распределена по множеству инструкций, регламентов и внутренних документов. Обычный keyword search требует от пользователя знать точную формулировку и самостоятельно просматривать найденные документы.

Цель RAG-сервиса — предоставить единый интерфейс, который:
1. Принимает вопрос на естественном языке;
2. Определяет поисковые формулировки;
3. Находит наиболее релевантные части документов;
4. Отбрасывает слабый или противоречивый контекст;
5. Передаёт компактный набор фактов в LLM;
6. Генерирует ответ только на основании найденных источников;
7. Возвращает пользователю ответ и ссылки на документы.

Пример пользовательского запроса: "Какой срок действия наряда-допуска?"
Вместо поиска по десяткам документов пользователь получает краткий ответ, сформированный по найденным фрагментам базы знаний, вместе с источниками.

Архитектура
Поток запроса:
Query
  -> query planning / normalization
  -> hybrid retrieval
       -> vector search (Hashing / SentenceTransformers)
       -> sparse TF-IDF search
  -> Reciprocal Rank Fusion (RRF)
  -> lexical reranking
  -> neighbor chunk expansion
  -> context selection
  -> quality gate
  -> Qwen3 generation (via llama.cpp)
  -> grounding / relevance validation
  -> sources resolution
  -> gRPC stream response

Что реализовано
1. Hybrid Retrieval
Retrieval объединяет несколько независимых сигналов поиска.
Vector channel: по умолчанию используется быстрый CPU-friendly HashingVectorizer (word n-grams 1-2, character n-grams 3-5). Также поддерживается полноценный embedding backend на базе sentence-transformers.
Sparse channel: для отдельного sparse-индекса используется TF-IDF через FeatureUnion (word TF-IDF 1-2 grams, character TF-IDF 3-5 grams). Это помогает учитывать точные термины, русскую морфологию, опечатки и аббревиатуры.

2. Reciprocal Rank Fusion и reranking
Результаты dense/vector и sparse retrieval объединяются через Reciprocal Rank Fusion (RRF). Дополнительно учитываются: lexical coverage запроса, совпадения с заголовком, близость query-term друг к другу внутри chunk, document-level affinity и штрафы за нерелевантные блоки.

3. Neighbor Expansion
Если найден релевантный chunk, retrieval может добавить соседние chunks того же документа, уменьшая вероятность потери важной информации на границах разбиения.

4. Adaptive Context Selection
Перед отправкой текста в LLM найденные фрагменты обрабатываются: удаляются дубли, выбираются наиболее информативные passages, ограничивается общий размер контекста. Предусмотрены профили запросов: fact (короткий контекст), default (средний), procedure (больше контекста).

5. Quality Gate
До генерации система оценивает качество найденных evidence (strong / borderline / weak). Если evidence недостаточно надёжный, система не заставляет LLM «додумывать» ответ, а использует уточняющий вопрос или extractive fallback.

6. Grounding Guard
После генерации ответ проверяется на соответствие retrieved context: coverage утверждений источниками, отсутствие выдуманных чисел/сущностей, наличие корректных source markers.

7. Answer Relevance Guard
Отдельная проверка контролирует, действительно ли сгенерированный ответ отвечает на исходный вопрос, а не просто пересказывает найденный документ.

8. Corrective Retrieval
Если первый набор документов недостаточен, pipeline поддерживает переформулирование поисковых запросов и повторный поиск.

9. Extractive Fallback
Если LLM не успела ответить в SLA, недоступна или не прошла валидацию, сервис возвращает extractive answer непосредственно из retrieved passages.

10. Latency-aware inference
Система рассчитана на локальный CPU inference: hard request deadline (23.5s), отдельный budget для LLM, адаптивный max_tokens, prompt cache, EMA-оценка реальной скорости llama.cpp.

11. Safe Knowledge Base Sync
Индекс строится во временной директории и только после успешной сборки атомарно заменяет предыдущую версию. Если пересборка завершится ошибкой, рабочий индекс остаётся доступным.

ML / NLP Stack
- Language: Python 3.10-3.13
- RAG orchestration: custom Python pipeline
- LLM: Qwen3-4B-RAG
- Local inference: llama.cpp
- LLM API: OpenAI-compatible HTTP API
- Dense/vector representation: HashingVectorizer / SentenceTransformers
- Sparse retrieval: TF-IDF, word + char n-grams (scikit-learn)
- Fusion: Reciprocal Rank Fusion
- Reranking: lexical reranking / optional CrossEncoder
- Validation: Pydantic, pydantic-settings
- Service communication: gRPC + Protocol Buffers
- Knowledge source: HiHub

Структура проекта
Gazprom_ProStroy_hack/
├── src/
│   └── rag_app/                 # Инфраструктурный слой RAG-сервиса
│       ├── grpc_main.py         # Entry point gRPC-сервера
│       ├── grpc_service.py      # gRPC API, health check и sync
│       ├── composition.py       # Сборка зависимостей приложения
│       └── ...                  # Модули retrieval, embeddings, chunking
├── proto/                       # gRPC Protobuf спецификации (ml.proto)
├── data/                        # Runtime index хранится в data/index/
├── scripts/                     # Вспомогательные утилиты (check_hihub, build_index)
├── .env.example                 # Пример runtime-конфигурации
├── pyproject.toml               # Конфигурация Python package (unified-adaptive-rag)
├── requirements.txt             # Основные зависимости
├── requirements-models.txt      # Опциональные embedding/reranking модели
├── setup.ps1                    # Создание venv и установка зависимостей
├── run_rag.ps1                  # Запуск RAG gRPC service
├── run_llama_cpu_stable.ps1     # Стабильный CPU запуск llama.cpp + Qwen3
├── autotune_llama_cpu.ps1       # Подбор CPU-параметров llama.cpp
├── diagnose_llama.ps1           # Диагностика локальной LLM
├── test_query.py                # Набор end-to-end тестовых gRPC запросов
└── README.md

gRPC API
Сервис реализует контракт ml.v1.MLGatewayService.
Доступные методы:
- Query: Получить RAG-ответ на пользовательский вопрос (stream)
- SyncKnowledgeBase: Запустить обновление knowledge base
- GetSyncStatus: Проверить состояние задачи синхронизации
- HealthCheck: Проверить состояние LLM и локального индекса

Query Request:
message QueryRequest {
  string trace_id = 1;
  string query = 2;
  int32 top_k = 3;
  int32 section_id = 4;
}
Ответ возвращается stream-сообщениями (токены) и завершается FinalResult (answer, sources, route_type, latency_ms).

Быстрый запуск
Требования:
- Windows 10/11
- Python 3.11+
- llama.cpp с доступной командой llama-server
- Локальная GGUF-модель (например, Qwen3-4B-Q4_K_M.gguf)
- Доступ к HiHub API

1. Клонирование проекта:
git clone https://github.com/igorvorobev777/Gazprom_ProStroy_hack.git
cd Gazprom_ProStroy_hack

2. Установка Python-зависимостей:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
(Скрипт создаст .venv, установит зависимости и проект в editable mode)

3. Настройка окружения:
Copy-Item .env.example .env
Заполните HiHub credentials в .env:
HIHUB_BASE_URL=https://hihub.ru
HIHUB_EMAIL=<email>
HIHUB_PASSWORD=<password>
# или HIHUB_TOKEN_NAME=<token>

Проверьте настройки LLM:
LLM_BASE_URL=http://127.0.0.1:1234/v1
LLM_MODEL=qwen3-4b-rag
GRPC_HOST=0.0.0.0
GRPC_PORT=50052

4. Проверка HiHub и построение индекса:
.\.venv\Scripts\Activate.ps1
python -m scripts.check_hihub
python -m scripts.build_index --force

5. Запуск Qwen3 через llama.cpp:
.\run_llama_cpu_stable.ps1
(или .\autotune_llama_cpu.ps1 для авто-тюнинга под ваше железо)

6. Запуск RAG-сервиса (во втором окне PowerShell):
.\run_rag.ps1
Сервис будет доступен на 0.0.0.0:50052

Проверка работы
End-to-end gRPC запросы:
В репозитории есть готовый тестовый клиент с реалистичными корпоративными запросами (Устав, Отпуск, IT-поддержка, Пожарная безопасность):
python test_query.py
Он отправляет вопросы в RAG и выводит generated answer, source title и source URL.

Другие утилиты:
- python -m scripts.quality_probe "Ваш вопрос" (проверка retrieval без полной генерации)
- python -m scripts.llm_profile (диагностика производительности LLM)

Конфигурация retrieval
Основные параметры вынесены в .env. Пример:
CHUNK_SIZE_CHARS=1400
CHUNK_OVERLAP_CHARS=180
EMBEDDING_BACKEND=hashing
DENSE_TOP_K=60
SPARSE_TOP_K=80
RERANK_CANDIDATE_K=80
RERANKER_BACKEND=rrf
LEXICAL_RERANK_ENABLED=true
NEIGHBOR_EXPANSION_ENABLED=true
FOCUSED_PASSAGES_ENABLED=true
MAX_CONTEXT_CHARS=1500

Для semantic embeddings можно установить дополнительные зависимости:
pip install -r requirements-models.txt
и переключить backend в .env:
EMBEDDING_BACKEND=sentence_transformers

Quality & Hallucination Control
Одна из ключевых частей проекта — отказ от стратегии «LLM всегда должна что-то ответить».
retrieval -> quality gate
  -> strong: generation
  -> borderline: restricted generation / fallback
  -> weak: clarification / insufficient context

После generation дополнительно выполняются: format validation -> grounding guard -> relevance guard -> source resolution. Если ответ не проходит проверки, pipeline переходит к extractive fallback.

Наблюдаемость и диагностика
Каждый запрос получает trace_id. FinalResult содержит детальную диагностику: latency_ms, route_type, retrieval_attempts, rewritten_queries и тайминги отдельных стадий. Это позволяет анализировать bottleneck-и inference pipeline.

Результат хакатона
Для проекта было важным практическим ограничением: система должна была не только демонстрировать идею RAG, но и интегрироваться с backend, работать локально, отвечать в заданном latency budget и устойчиво обрабатывать ситуацию, когда в базе знаний нет достаточного ответа. Именно этот production-ready подход позволил решению занять призовое место.

Ключевые идеи проекта:
Hybrid Retrieval + Adaptive Context + Local Qwen3 + Quality Gate + Grounding Validation + gRPC Integration = Production-oriented RAG Service
