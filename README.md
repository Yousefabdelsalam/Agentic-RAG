# Agentic RAG

Question answering over your own PDFs. Ingest documents, then ask about them.
Each question runs through a LangGraph agent that analyses it, plans how to
gather evidence, uses tools and retrieval only where they are needed, drafts an
answer, and has a critic check that answer against its sources before returning
it.

Answers carry citations. Every response reports which nodes ran and what the
critic concluded, so a confident answer can be told apart from one that was
accepted reluctantly.

## Quick start

```bash
cp .env.example .env          # then set OPENAI__API_KEY
uv sync
uv run uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`.

```bash
# Ingest a document
curl -F "file=@handbook.pdf" http://localhost:8000/api/v1/upload

# Ask about it
curl -X POST http://localhost:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"query": "How many days of leave do staff get?"}'

# Stream the run instead
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H 'content-type: application/json' \
  -d '{"query": "How much leave?", "stream": true}'
```

Ingestion also runs offline, without the API:

```bash
uv run python -m app.ingestion handbook.pdf policies.pdf
```

## Docker

```bash
cp .env.example .env          # then set OPENAI__API_KEY
docker compose up --build
```

This starts the API with Chroma as a separate service, so rebuilding the API
does not lose the index. For a single container, drop the `chroma` service and
unset `CHROMA__HOST`; the API then persists to its own volume.

```bash
docker compose ps             # health status of both services
docker compose logs -f api
docker compose down           # add -v to discard the index too
```

**The API runs one worker on purpose.** Conversation memory is in-process, so a
second worker would answer follow-up questions from a session it has never seen.
Replace `InMemoryStore` with a shared implementation before scaling out.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | Ask a question. `stream: true` returns Server-Sent Events. |
| `POST` | `/api/v1/upload` | Ingest a PDF (multipart, 50 MB limit). |
| `POST` | `/api/v1/reset-memory` | Forget one conversation. |
| `GET` | `/api/v1/cache/stats` | Cache counters and the versions in force. |
| `GET` | `/api/v1/health` | Liveness. Cheap, no dependency calls. |
| `GET` | `/api/v1/ready` | Readiness, per registered component. |

Errors always return the same envelope — `code`, `message`, `request_id`,
`details` — and echo `X-Request-ID`, which also appears on every log line for
that request.

### Streaming

With `stream: true` the response is `text/event-stream`, one JSON object per
event:

- `stage` — a graph node finished, so progress is visible during a slow run;
- `answer` — a draft was written (twice, if the critic sent the run back);
- `result` — terminal, carrying the same body as the non-streaming response.

A failure after streaming has begun arrives as a terminal `error` event: the
HTTP status is already committed by then.

## How a question is answered

```
START → memory_loader → query_analyzer → planner ─┬─ tools ─┬─ retriever ─┐
                                                  │         └─────────────┤
                                                  ├─ retriever ───────────┤
                                                  └───────────────────────┴→ generator
                                      generator → critic ─┬─ planner (insufficient context)
                                                          └─ memory_writer → END
```

The planner decides independently whether retrieval and tools are needed, and
the graph routes past whichever is not: a question needing neither never enters
those nodes. The critic judges two things separately — whether the context was
sufficient, and whether the answer stayed within it. Only insufficient context
sends the run back to the planner; an answer that strayed from context it
already had is not fixed by retrieving again. Revisions are capped by
`AGENT__MAX_REVISIONS`.

Tools are a calculator, a clock, and a web search interface. No search backend
ships: the default one fails with a stated reason, so a plan that reaches for it
produces a legible gap rather than a silent one.

## Answer caching

A question that has already been answered does not run the graph again.

```
request -> exact cache -> semantic cache -> the graph
```

The exact cache keys on the question after normalisation — case, whitespace, and
typographic punctuation are erased, and nothing else is. The semantic cache runs
only when the exact key missed: it embeds the question and compares it against
the cached ones, serving the nearest above `RAG_CACHE_SIMILARITY_THRESHOLD`.
Every response says which happened:

```json
{ "cache_hit": true, "cache_type": "semantic", "cache_age": 41.2, "answer": "..." }
```

**A hit runs no nodes and calls no model.** It still appends the turn to
conversation memory, because that costs a dictionary write rather than a model
call.

**Only answers the critic stood behind are cached.** An answer that was not
grounded, ran out of context, failed a tool call, or landed below
`AGENT__MIN_CONFIDENCE` is finished but not correct, and caching it would turn
one bad answer into every future answer to that question.

**Follow-up questions are neither served nor stored.** "And for part-time
staff?" means something different in every conversation it appears in, so
sessions that already have a turn on record bypass the cache in both directions.
What remains cached is the set of questions that stand on their own.

### Invalidation

An entry is only valid under the three conditions that produced it, all three of
which are part of its key:

| Version | Derived from | Moves when |
| --- | --- | --- |
| `knowledge_base_version` | a counter advanced by ingestion | a document is indexed, by API or CLI |
| `model_version` | `OPENAI__CHAT_MODEL` and `AGENT__TEMPERATURE` | either is changed |
| `prompt_version` | a digest of `app/prompts/templates/` | any template is edited |

Nothing has to be remembered or bumped by hand: editing a prompt or swapping a
model invalidates the cache by changing what the entries are keyed under. TTL
(`RAG_CACHE_TTL_SECONDS`) is the backstop for everything these three do not
cover.

### Backend

In-process by default. Set `RAG_CACHE_REDIS_URL` to share the cache across
replicas and restarts — which is also what makes the knowledge base version
shared, so a replica that did not perform an ingestion still stops serving
answers from the corpus that preceded it.

```bash
uv sync --extra redis
RAG_CACHE_REDIS_URL=redis://localhost:6379/0
```

Redis is verified with one round trip at boot. If the package is missing or the
server unreachable, the process logs `cache.redis_unavailable` and runs on the
in-memory backend; a Redis that fails later degrades to a cache that misses.
Neither ever fails a request — the whole layer is an optimisation, and an
optimisation that can break the thing it optimises is not one.

### Observability

`GET /api/v1/cache/stats` reports `cache_hits_total`, `cache_misses_total`,
`exact_cache_hits`, `semantic_cache_hits`, `cache_hit_rate`, and
`estimated_llm_calls_saved`, alongside the three versions currently in force.
Counters are per process. Lookups and stores are traced as `cache.lookup` and
`cache.store` spans carrying `cache_hit`, `cache_type`, `cache_key`, and
`cache_age`, so a hit is visible in LangSmith as a run that answered without
calling anything.

## Configuration

Every setting is an environment variable; nested ones use a double underscore
(`OPENAI__API_KEY`). `.env.example` lists all of them with defaults. The ones
that matter most:

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENAI__API_KEY` | — | Required for `/chat` and `/upload`. |
| `OPENAI__CHAT_MODEL` | `gpt-4o` | |
| `OPENAI__EMBEDDING_MODEL` | `text-embedding-3-small` | Changing this invalidates the index. |
| `CHROMA__HOST` / `CHROMA__PORT` | unset | Set to use a Chroma server; unset persists locally. |
| `AGENT__MAX_REVISIONS` | `2` | Critic → planner loops allowed. |
| `RAG_CACHE_ENABLED` | `true` | Answer cache. Off means every question runs the graph. |
| `RAG_CACHE_TTL_SECONDS` | `3600` | Backstop for what versioning does not catch. |
| `RAG_CACHE_SIMILARITY_THRESHOLD` | `0.92` | Below this, a question is not the same question. |
| `RAG_CACHE_REDIS_URL` | unset | Set to share the cache; unset keeps it in-process. |
| `MEMORY__SHORT_TERM_TURNS` | `6` | Turns kept verbatim; older ones survive as summary. |
| `RESILIENCE__REQUEST_TIMEOUT_SECONDS` | `120` | Time to first byte, not total. |
| `LANGSMITH__ENABLED` | `false` | Set with `LANGSMITH__API_KEY` to trace runs. |

**Starting without an API key is allowed.** The process boots, logs
`bootstrap.degraded`, and serves `/health` and `/reset-memory`; `/chat` and
`/upload` answer `502` explaining what is missing. A container that refuses to
start makes `/health` unreachable exactly when someone is diagnosing why.

## Production behaviour

**Health checks.** `/health` is liveness and touches nothing. `/ready` reports
each registered component. Both are cheap by design — the readiness probe does
not call OpenAI, because a probe polled every few seconds must not bill per
poll. `ChatService.probe()` and `EmbeddingService.probe()` make real round trips
when that is actually what you want.

**Retries.** Calls to Chroma retry transient failures with exponential backoff
and jitter. Rejections are not retried: a bad filter fails identically the
second time, so retrying only makes the error slower and the dependency busier.
OpenAI calls retry inside the client (`OPENAI__MAX_RETRIES`).

**Timeouts.** Layered: `RESILIENCE__REQUEST_TIMEOUT_SECONDS` for HTTP requests,
`AGENT__REQUEST_TIMEOUT_SECONDS` for chat calls, `OPENAI__TIMEOUT_SECONDS` for
embeddings, `TOOLS__TIMEOUT_SECONDS` per tool call. The HTTP budget covers
time-to-first-byte only, so streaming runs are not cut off mid-answer.

**Caching.** Query embeddings are cached with a TTL, keyed by model and text.
Concurrent misses on the same key share one load rather than each starting their
own. Document embeddings are not cached — each chunk is embedded once.

**Observability.** With LangSmith enabled, every node, model call, and retriever
is traced with latency, tokens, cost, retrieved documents, the final prompt, the
final answer, and errors. Costs are also estimated in-process, so they appear in
logs whether or not tracing is on. Logs are structured JSON
(`LOGGING__JSON_FORMAT=false` for readable local output).

## Tests

```bash
uv run pytest                      # everything
uv run pytest -m "not integration" # fast unit suite
uv run pytest -m integration       # real stack, no network
uv run ruff check app tests
uv run mypy app
```

Integration tests wire the production container and stub only the two paid
network calls. The real PDF loader, splitter, Chroma collection, retrievers,
graph, and HTTP app all run. No test reaches the network.

## Layout

| Path | Responsibility |
| --- | --- |
| `app/config` | Typed settings loaded from the environment |
| `app/core` | Logging, tracing, errors, middleware, DI container, retry, cache |
| `app/api` | HTTP transport: routers and dependency providers |
| `app/agents` | The LangGraph agent: state, nodes, graph, facade |
| `app/retrieval` | Vector store, retrievers, filters, compression |
| `app/ingestion` | PDF loading, chunking, indexing, pipeline |
| `app/memory` | Conversation summary, short-term window, user context |
| `app/tools` | Calculator, clock, web search interface, registry |
| `app/services` | Chat and embedding clients |
| `app/prompts` | Prompt templates and renderer |
| `app/models` | Pydantic base schemas and API payloads |

## Conventions

- Dependencies flow inward: `api → services → domain contracts`. Nothing in
  `services` or below imports FastAPI.
- Concrete implementations are chosen in one place, `app/core/bootstrap.py`, and
  resolved by type from the `Container`.
- Contracts are Protocols; collaborators arrive via constructor injection.
- Graph nodes communicate only through state. No node imports another.
- Every error reaching the transport becomes an `ErrorResponse` carrying the
  `X-Request-ID` correlation id.

## Known limits

- **Memory and the embedding cache are in-process.** They do not survive a
  restart or span replicas. Both are seams, not assumptions.
- **No sparse retrieval backend.** Hybrid search fuses dense and sparse
  rankings, but with nothing registered on the sparse side it degrades to dense
  and says so in the result trace.
- **Metadata values are strings.** Ingestion writes `page="3"`, so `eq` and `in`
  filters behave, while `gt`/`lt` compare lexically.
- **Semantic cache lookup is a linear scan.** Cached vectors are compared in
  process, bounded by `RAG_CACHE_SEMANTIC_MAX_ENTRIES`. A vector-native index
  (RediSearch, or the Chroma collection itself) is the seam for a larger cache.
- **Without Redis, offline ingestion cannot invalidate the API's cache.** The
  knowledge base version lives in the cache backend, and an in-process one dies
  with the CLI. Either set `RAG_CACHE_REDIS_URL` or ingest through `/upload`.
- **Cost figures are estimates** from a local price table, and embedding token
  counts are approximated from text length because the endpoint reports none.
