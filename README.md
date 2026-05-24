# Brand Guardian — AI Compliance Auditing Pipeline

Brand Guardian is an AI-powered backend pipeline that automatically audits video advertisements for brand compliance violations. It downloads a YouTube video, extracts speech and on-screen text using AWS, and runs a RAG-based audit against a regulatory knowledge base using Mistral AI and Pinecone — returning a structured report of every violation found.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Backend Deep Dive](#backend-deep-dive)
  - [API Layer](#api-layer)
  - [LangGraph Workflow](#langgraph-workflow)
  - [Node 1 — Video Indexer](#node-1--video-indexer)
  - [Node 2 — Compliance Auditor](#node-2--compliance-auditor)
  - [State Schema](#state-schema)
  - [Telemetry & Observability](#telemetry--observability)
  - [Knowledge Base Indexing](#knowledge-base-indexing)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Setup & Running](#setup--running)
  - [Docker (recommended)](#docker-recommended)
  - [Local Development](#local-development)
- [API Reference](#api-reference)
- [Frontend](#frontend)

---

## Architecture Overview

```
YouTube URL
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   FastAPI Server                    │
│                   POST /audit                       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Workflow                     │
│                                                     │
│   START ──► [ indexer node ] ──► [ auditor node ] ──► END
└─────────────────────────────────────────────────────┘
                        │                    │
                        ▼                    ▼
             ┌──────────────────┐   ┌─────────────────────┐
             │   AWS Services   │   │   Mistral AI + RAG  │
             │                  │   │                     │
             │  yt-dlp download │   │  MistralAIEmbeddings│
             │  S3 upload       │   │  Pinecone similarity│
             │  Transcribe STT  │   │  ChatMistralAI LLM  │
             │  Rekognition OCR │   │  Compliance report  │
             └──────────────────┘   └─────────────────────┘
                        │
                        ▼
             ┌──────────────────┐
             │  AWS CloudWatch  │
             │  Metrics & Logs  │
             └──────────────────┘
```

---

## Backend Deep Dive

### API Layer

**File:** `backend/src/api/server.py`

Built with **FastAPI**. Exposes two endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/health` | Health probe used by Docker and load balancers |
| `POST` | `/audit`  | Main entry point — accepts a YouTube URL, runs the full pipeline |

The `/audit` endpoint generates a `session_id` (UUID) and a short `video_id` (first 8 chars of the UUID), then invokes the LangGraph workflow synchronously. It returns an `AuditResponse` with the final status, report, and list of violations.

**Request body:**
```json
{ "video_url": "https://youtu.be/dT7S75eYhcQ" }
```

**Response body:**
```json
{
  "session_id": "b3f1a2c4-...",
  "video_id": "b3f1a2c4",
  "status": "FAIL",
  "final_report": "The video contains unsubstantiated health claims...",
  "compliance_results": [
    {
      "category": "Claim Validation",
      "severity": "CRITICAL",
      "description": "Product claims '100% cures' without regulatory approval"
    }
  ]
}
```

---

### LangGraph Workflow

**File:** `backend/src/graph/workflow.py`

The pipeline is built as a **LangGraph `StateGraph`** — a directed acyclic graph where each node receives the full state, performs work, and returns a partial state update.

```
START ──► indexer ──► auditor ──► END
```

The graph is compiled once at module load time and reused across all requests:

```python
workflow = create_graph()   # compiled once
graph.invoke(initial_inputs)  # called per request
```

---

### Node 1 — Video Indexer

**File:** `backend/src/graph/nodes.py` → `index_video_node()`
**Service:** `backend/src/services/video_indexer.py` → `VideoIndexerService`

This node handles all video acquisition and media intelligence. It performs the following steps in sequence:

**1. Download video**
Uses `yt-dlp` to download the YouTube video as an MP4 to a local temp file. Targets `best[ext=mp4]` quality with Android/Web client spoofing to bypass bot detection.

**2. Upload to S3**
Uploads the MP4 to the configured S3 bucket under `videos/{video_id}.mp4`. Auto-creates the bucket if it does not exist. Verifies the upload with `head_object` before proceeding (retries up to 5 times).

**3. Start AWS jobs in parallel**
Fires off three AWS async jobs simultaneously:
- **AWS Transcribe** — speech-to-text on the video audio, returns a full transcript string
- **AWS Rekognition (label detection)** — detects objects, scenes, and activities with ≥70% confidence
- **AWS Rekognition (text detection)** — extracts all on-screen text (OCR), filtered to `LINE` type only

**4. Poll until complete**
Each job is polled on a 10-second interval with a 600-second timeout. On completion, the transcript URL is fetched from Transcribe's output JSON, and Rekognition responses are parsed directly.

**5. Cleanup**
The local MP4 temp file is deleted after a successful S3 upload.

**State output:**
```python
{
    "transcript": "..full speech text..",
    "ocr_text": ["Brand Name®", "Limited Offer", ...],
    "video_metadata": { "detected_labels": ["Person", "Product", ...] },
    "error": []
}
```

---

### Node 2 — Compliance Auditor

**File:** `backend/src/graph/nodes.py` → `audio_content_node()`

This node performs the RAG-based compliance audit using the transcript and OCR text extracted by the indexer node.

**1. Guard clause**
If no transcript is available (indexer failed), the node returns immediately with `FAIL` status and skips the LLM call entirely.

**2. Pinecone vector search**
Combines the transcript and OCR text into a single query string and runs a `similarity_search(k=4)` against the Pinecone index. The index holds chunked compliance rules (brand guidelines, regulatory policies, FTC/FDA standards) that were pre-loaded by the indexing script. The 4 most relevant rule chunks are retrieved as context.

**3. LLM audit via Mistral**
Sends the retrieved rules + video content to `mistral-large-latest` via LangChain. The system prompt instructs the model to act as a senior brand compliance auditor and return **only** strict JSON — no preamble, no markdown. The expected JSON schema is:

```json
{
  "compliance_results": [
    {
      "category": "Claim Validation",
      "severity": "CRITICAL",
      "description": "..."
    }
  ],
  "status": "FAIL",
  "final_report": "..."
}
```

**4. JSON parsing**
The response is stripped of any accidental markdown code fences (` ```json `) before being parsed. If parsing fails, the raw response is logged and a system error is returned.

**Severity levels:** `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`

**State output:**
```python
{
    "compliance_result": [...violations...],
    "final_result": "FAIL",
    "final_report": "Summary of findings..."
}
```

---

### State Schema

**File:** `backend/src/graph/state.py`

The entire pipeline shares a single typed state object — `VideoAuditState`. LangGraph passes this between nodes, with list fields using `operator.add` to safely accumulate results across nodes.

```python
class VideoAuditState(TypedDict):
    # Input
    video_url:        str
    video_id:         str

    # Populated by indexer node
    local_file_path:  Optional[str]
    video_metadata:   Dict[str, Any]      # detected_labels from Rekognition
    transcript:       Optional[str]        # AWS Transcribe output
    ocr_text:         List[str]            # AWS Rekognition text lines

    # Populated by auditor node
    compliance_result: Annotated[List[ComplianceIssue], operator.add]
    final_result:     str                  # "PASS" or "FAIL"
    final_report:     str                  # LLM narrative summary

    # Accumulated across all nodes
    error:            Annotated[List[str], operator.add]
```

---

### Telemetry & Observability

**File:** `backend/src/api/telemetry.py`

Every significant operation publishes metrics and structured logs to **AWS CloudWatch** under the namespace `BrandGuardian/CompliancePipeline`.

**Metrics tracked:**

| Metric | Unit | Description |
|--------|------|-------------|
| `VideoDownloadDuration` | Milliseconds | Time to download the YouTube video |
| `S3UploadDuration` | Milliseconds | Time to upload to S3 |
| `TranscriptionDuration` | Milliseconds | Time for Transcribe job to complete |
| `RekognitionDuration` | Milliseconds | Time for both Rekognition jobs |
| `PineconeQueryDuration` | Milliseconds | Vector similarity search latency |
| `LLMInterfaceDuration` | Milliseconds | Mistral LLM call latency |
| `VideoUploaded` | Count | Successful S3 uploads |
| `AuditCompleted` | Count | Completed audit runs |
| `AuditStatus` | Count | Pass/fail dimension |
| `ViolationsDetected` | Count | Number of violations per audit |
| `AuditDurationMs` | Milliseconds | Total end-to-end audit time |
| `AuditSkipped` | Count | Audits skipped due to missing transcript |

`MetricTimer` is a context manager wrapping any code block to measure its duration and push to CloudWatch automatically:

```python
with MetricTimer("TranscriptionDuration", {"VideoId": video_id}):
    transcript = vi_service.wait_for_transcription(job_name)
```

LangSmith tracing is also enabled via `LANGSMITH_TRACING=true`, which traces all LangChain/LangGraph calls for debugging and evaluation.

---

### Knowledge Base Indexing

**File:** `backend/scripts/index_documents.py`

Before the system can audit anything, compliance rules must be indexed into Pinecone. This is a one-time (or on-update) script that:

1. Loads all PDF files from `backend/data/`
2. Splits them into chunks of 400 tokens with 100-token overlap using `RecursiveCharacterTextSplitter`
3. Generates embeddings using `MistralAIEmbeddings` (`mistral-embed`, 1024 dimensions)
4. Upserts all chunks into the Pinecone index (`compliance-rules` by default)

**To run:**
```bash
# Place your compliance PDF rulebooks in backend/data/
python -m backend.scripts.index_documents
```

This only needs to be re-run when your compliance rulebooks change. The Pinecone index is persistent and reused across all audit requests.

---

## Project Structure

```
ComplianceQAPipeline/
├── Dockerfile                        # Backend image (Python 3.11, ffmpeg, uvicorn)
├── Dockerfile.frontend               # Frontend image (nginx + static files)
├── docker-compose.yml                # Orchestrates api + frontend
├── nginx.conf                        # nginx reverse proxy config
├── main.py                           # CLI runner for local testing
├── pyproject.toml                    # Python dependencies (uv)
│
├── backend/
│   ├── data/                         # Place compliance PDF rulebooks here
│   ├── scripts/
│   │   └── index_documents.py        # One-time Pinecone indexing script
│   └── src/
│       ├── api/
│       │   ├── server.py             # FastAPI app, /health and /audit endpoints
│       │   └── telemetry.py          # CloudWatch metrics + MetricTimer
│       ├── graph/
│       │   ├── workflow.py           # LangGraph StateGraph definition
│       │   ├── nodes.py              # indexer node + auditor node
│       │   └── state.py             # VideoAuditState TypedDict
│       └── services/
│           └── video_indexer.py      # VideoIndexerService (yt-dlp, S3, Transcribe, Rekognition)
│
└── frontend/
    ├── index.html
    ├── app.js
    └── style.css
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Mistral AI
MISTRAL_API_KEY=your_mistral_api_key

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=compliance-rules

# AWS (S3, Transcribe, Rekognition, CloudWatch)
AWS_REGION=us-east-1
AWS_BUCKET_NAME=your-s3-bucket-name

# LangSmith (optional but recommended for tracing)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=brand_guardian
```

AWS credentials are expected to be available via the standard boto3 credential chain — either environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`), an IAM role, or `~/.aws/credentials`.

**Required AWS IAM permissions:**
- `s3:CreateBucket`, `s3:PutObject`, `s3:HeadObject`
- `transcribe:StartTranscriptionJob`, `transcribe:GetTranscriptionJob`
- `rekognition:StartLabelDetection`, `rekognition:StartTextDetection`, `rekognition:GetLabelDetection`, `rekognition:GetTextDetection`
- `cloudwatch:PutMetricData`
- `logs:CreateLogGroup`, `logs:CreateLogStream`, `logs:PutLogEvents`

---

## Setup & Running

### Docker (recommended)

```bash
# 1. Clone and enter the project
cd D:\Brand_Guardline\ComplianceQAPipeline

# 2. Add your compliance PDFs
#    Copy rulebook PDFs into backend/data/

# 3. Index the knowledge base (run once)
docker compose run --rm api python -m backend.scripts.index_documents

# 4. Start the full stack
docker compose up --build
```

- API available at: `http://localhost:8000`
- Frontend available at: `http://localhost`

### Local Development

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync

# Run the API
uvicorn backend.src.api.server:app --host 0.0.0.0 --port 8000 --reload

# Or run a single audit via CLI
python main.py
```

---

## API Reference

### `POST /health`

Returns `200 OK` if the service is running. Used by Docker healthcheck.

```json
{ "status": "healthy", "service": "Brand Guardian" }
```

### `POST /audit`

Runs the full compliance audit pipeline on a YouTube video.

**Request:**
```json
{ "video_url": "https://youtu.be/dT7S75eYhcQ" }
```

**Response `200 OK`:**
```json
{
  "session_id": "b3f1a2c4-8e9d-4f1a-a2c4-8e9d4f1ab3f1",
  "video_id": "b3f1a2c4",
  "status": "FAIL",
  "final_report": "The video contains two critical violations...",
  "compliance_results": [
    {
      "category": "Claim Validation",
      "severity": "CRITICAL",
      "description": "Unsubstantiated superlative claim: 'the only product proven to...'"
    },
    {
      "category": "Trademark Usage",
      "severity": "HIGH",
      "description": "Registered trademark symbol missing on third on-screen mention"
    }
  ]
}
```

**Response `500`:**
```json
{ "detail": "Workflow Execution Failed: <reason>" }
```

---

## Frontend

A lightweight static UI (`frontend/index.html`) served by nginx. It submits YouTube URLs to the `/audit` endpoint and renders the compliance report. No build step required — nginx serves the files directly from the Docker image.

See `Dockerfile.frontend` and `nginx.conf` for serving configuration. The nginx config also handles the `/api/` → `http://api:8000/` proxy with a 300-second timeout to accommodate long-running audit jobs.
