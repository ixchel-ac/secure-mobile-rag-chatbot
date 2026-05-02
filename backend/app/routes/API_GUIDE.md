# API Guide: REST APIs, Structure, Best Practices, and This Backend

---

## What Is an API?

An API (Application Programming Interface) is a contract between two pieces of software. One side (the **client**) sends a request, the other side (the **server**) processes it and sends back a response. Think of it like a restaurant menu — the menu defines what you can order, in what format, and what you'll get back.

In web development, the most common type is a **REST API** — an API that uses HTTP methods (GET, POST, PUT, DELETE) to operate on resources identified by URLs.

---

## Anatomy of an HTTP Request

Every API call has these parts:

```
POST /query HTTP/1.1              ← Method + Path
Host: localhost:8000              ← Where to send it
Content-Type: application/json    ← Format of the body
Authorization: Bearer <token>     ← Authentication (if required)

{                                 ← Body (the data you're sending)
  "query": "What medications is the patient taking?",
  "top_k": 5
}
```

| Part | Purpose |
|------|---------|
| **Method** | What action to perform (GET = read, POST = create/submit, PUT = update, DELETE = remove) |
| **Path** | Which resource (`/query`, `/health`, `/ingest`) |
| **Headers** | Metadata — content type, auth tokens, etc. |
| **Body** | The actual data (only for POST/PUT) |

## Anatomy of an HTTP Response

```
HTTP/1.1 200 OK                   ← Status code
Content-Type: application/json

{                                  ← Body (what the server returns)
  "response": "The patient is taking Metformin 500mg...",
  "redacted_entities": ["SSN", "NAME"],
  "sources": ["patient_record_001.txt"],
  "fw_l2_passed": true
}
```

| Status Code | Meaning |
|-------------|---------|
| **200** | OK — request succeeded |
| **201** | Created — resource was created |
| **400** | Bad Request — client sent invalid data |
| **401** | Unauthorized — authentication required |
| **404** | Not Found — resource doesn't exist |
| **422** | Unprocessable Entity — validation failed (FastAPI default for bad request bodies) |
| **500** | Internal Server Error — something broke on the server |

---

## REST API Best Practices

### 1. Use nouns for paths, not verbs
```
Good:  POST /query          ← "create a query"
Bad:   POST /run-query      ← verb in the path
```

### 2. Use HTTP methods to express actions
```
GET    /patients            ← list patients
GET    /patients/123        ← get one patient
POST   /patients            ← create a patient
PUT    /patients/123        ← update a patient
DELETE /patients/123        ← delete a patient
```

### 3. Return consistent response shapes
Every endpoint should return a predictable structure. Use Pydantic models (in FastAPI) to enforce this — the client always knows what fields to expect.

### 4. Use proper status codes
Don't return 200 for everything. A failed validation should be 422, a missing resource should be 404, an auth failure should be 401.

### 5. Validate inputs at the boundary
Never trust client data. Validate types, ranges, and required fields before processing. FastAPI does this automatically via Pydantic models.

### 6. Version your API (when needed)
If you expect breaking changes, prefix paths: `/v1/query`, `/v2/query`. For internal/single-client APIs (like this one), versioning is optional.

### 7. Keep endpoints focused
Each endpoint should do one thing. Don't combine "ingest data" and "query data" into a single endpoint.

### 8. Document your API
FastAPI generates interactive docs automatically at `/docs` (Swagger UI) and `/redoc` (ReDoc). Both are available out of the box.

---

## How This Backend Applies These Practices

### Framework: FastAPI

FastAPI is a Python web framework built on top of:
- **Starlette** — async HTTP handling
- **Pydantic** — request/response validation via type hints
- **OpenAPI** — auto-generated API documentation

Defined in `backend/app/main.py`:

```python
app = FastAPI(
    title="Mobile RAG Firewall Backend",
    description="Cloud backend for the two-level firewall RAG system",
    version="0.1.0",
)
```

### Our Three Endpoints

#### `GET /health` — Service health check

**File:** `health.py`

```python
@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "faiss_loaded": False,
        "ollama": "unchecked",
    }
```

**Purpose:** Lets the mobile app (or Docker health check) verify the backend is running before sending queries. No request body needed — it's a simple GET.

**Best practices applied:**
- GET method (read-only, no side effects)
- Returns service dependency status (FAISS, Ollama)

---

#### `POST /query` — Main RAG endpoint

**File:** `query.py`

```python
@router.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    ...
```

**Request schema** (defined in `models/schemas.py`):

```python
class QueryRequest(BaseModel):
    query: str       # The natural language medical question
    top_k: int = 5   # Number of chunks to retrieve (default: 5)
```

**Response schema:**

```python
class QueryResponse(BaseModel):
    response: str              # The anonymized LLM answer
    redacted_entities: list[str]  # PHI types found and redacted (e.g., ["SSN", "NAME"])
    sources: list[str]         # Source files used for context
    fw_l2_passed: bool         # Whether FW-L2 validation passed
```

**What happens internally (when fully implemented):**

```
Client sends: {"query": "What medications is the patient on?", "top_k": 5}
                │
                ▼
    1. Embed the query → 384-dim vector (all-MiniLM-L6-v2)
    2. Search FAISS index → top-5 most relevant chunks
    3. Build prompt: system prompt + retrieved chunks + user query
    4. Send to LLM (Ollama/Groq/W&B) → raw answer
    5. FW-L2 scans raw answer:
       - Regex: catches SSN, phone, email, DOB, MRN
       - NER: catches names, addresses
       - Injection: catches system prompt leaks
    6. Redact all detected PHI → sanitized answer
                │
                ▼
Client receives: {"response": "The patient takes Metformin 500mg...",
                  "redacted_entities": ["SSN", "NAME"],
                  "sources": ["patient_001.txt"],
                  "fw_l2_passed": true}
```

**Best practices applied:**
- POST method (submitting data for processing)
- Pydantic validation on input (`query` is required, `top_k` has a default)
- `response_model=QueryResponse` ensures the output shape is enforced
- Separation of concerns: route handler delegates to `RAGPipeline`

---

#### `POST /ingest` — Data ingestion trigger

**File:** `ingest.py`

```python
@router.post("/ingest", response_model=IngestResponse)
async def trigger_ingestion(request: IngestRequest):
    ...
```

**Request schema:**

```python
class IngestRequest(BaseModel):
    data_path: str = "./data"       # Path to Synthea data directory
    clear_existing: bool = True     # Whether to rebuild index from scratch
```

**Response schema:**

```python
class IngestResponse(BaseModel):
    chunks_ingested: int   # Number of chunks added to FAISS
```

**Purpose:** Triggers the full ingestion pipeline: load Synthea CSVs + text files → clean → chunk → embed → build FAISS index.

**Best practices applied:**
- POST method (triggers a side effect — rebuilds the index)
- `clear_existing` flag gives the client control over destructive behavior
- Returns a count so the client can verify success

---

### Request Validation with Pydantic

FastAPI uses Pydantic models to validate every request automatically. If a client sends bad data:

```bash
# Missing required field "query"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"top_k": 3}'
```

FastAPI returns a 422 with details:

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

No manual validation code needed — the schema definition IS the validation.

---

### CORS (Cross-Origin Resource Sharing)

Defined in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

This allows any client (including the Android emulator at `http://10.0.2.2:8000`) to call the API. For production, you'd restrict `allow_origins` to specific domains.

---

### Router Organization

Each route file defines a `router` that gets included in the main app:

```python
# main.py
app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingest.router)
```

This keeps concerns separated — health checks, queries, and ingestion are each in their own file with their own logic.

---

### Auto-Generated Documentation

FastAPI generates interactive API docs automatically:

- **Swagger UI:** `http://localhost:8000/docs` — interactive testing interface
- **ReDoc:** `http://localhost:8000/redoc` — clean read-only documentation

Both are generated from the Pydantic models and route decorators — no manual documentation needed. The `description` fields in `Field(...)` appear directly in the docs.

---

## Running the Backend

```bash
# Option 1: Docker (recommended)
docker compose up backend -d
# API available at http://localhost:8000

# Option 2: Direct
cd backend
uv sync && uv pip install -e .
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

# Test it
curl http://localhost:8000/health
curl http://localhost:8000/docs   # Open in browser for Swagger UI
```

---

## Architecture Summary

```
Mobile App (Android)
    │
    │  POST /query  {"query": "...", "top_k": 5}
    ▼
┌─────────────────────────────────────────────┐
│  FastAPI Backend (port 8000)                │
│                                             │
│  main.py  ──→  routes/query.py              │
│                    │                        │
│                    ▼                        │
│  rag/pipeline.py                            │
│    ├── retriever.py  (FAISS search)         │
│    ├── generator.py  (LLM call)             │
│    └── fw_l2.py      (PHI redaction)        │
│                    │                        │
│                    ▼                        │
│  QueryResponse  ──→  client                 │
└─────────────────────────────────────────────┘
```