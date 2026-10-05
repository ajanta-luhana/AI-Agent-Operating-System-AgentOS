# AgentOS

> An autonomous AI agent operating system for research, data analysis, verification, and report generation.

AgentOS transforms a natural-language goal into a structured, verified, evidence-backed report using a multi-agent workflow.

**Example:**

> "Analyze my sales data, detect anomalies, research competitors, and prepare a report."

The system plans the task, analyzes data, researches the web, verifies results, and generates the final report.

---

## Architecture

```text
User Goal
    │
    ▼
 Planner
    │
    ▼
Human Approval
    │
 ┌──┴──────────────┐
 ▼                 ▼
Analyst        Researcher
 │                 │
 └───────┬─────────┘
         ▼
      Verifier
         │
    ┌────┴────┐
    ▼         ▼
Reflector   Reporter
    │         │
    └──►Planner
              │
              ▼
        Final Report
Key Features
🤖 Multi-Agent Orchestration — Planner, Analyst, Researcher, Verifier, Reflector & Reporter
🔎 RAG Research — Hybrid vector and full-text retrieval with pgvector
📊 Data Analysis — CSV profiling, anomaly detection and chart generation
🐳 Sandboxed Execution — LLM-generated Python runs inside isolated Docker containers
✅ Self-Verification — Independently validates analytical results and research claims
👤 Human-in-the-Loop — Review and approve plans before execution
🧠 Long-Term Memory — PostgreSQL + pgvector
⚡ Live Observability — Redis + Server-Sent Events
🔒 Local AI — Ollama-based LLMs keep sensitive data local
Tech Stack

Python · FastAPI · LangGraph · PostgreSQL · pgvector · Redis · Docker · Ollama

API
Method	Endpoint	Description
POST	/goals	Create an agent run
GET	/runs/{id}	Get run status and results
POST	/runs/{id}/approve	Approve or reject the plan
GET	/runs/{id}/stream	Stream live agent events
GET	/runs/{id}/report	Get the final report
GET	/health	Health check
Project Structure
app/
├── agents/          # Agent implementations
├── graph/           # LangGraph orchestration
├── tools/           # Web & sandbox tools
├── memory/          # RAG & long-term memory
├── observability/   # Audit logging
├── api/             # FastAPI routes
└── worker.py        # Background worker

evals/               # Evaluation pipeline
sandbox/             # Isolated code execution
tests/               # Automated tests
docker-compose.yml
Security
Isolated Docker sandbox
No network access during code execution
CPU, memory and process limits
SSRF protection
Non-root execution
Web content treated as untrusted data
Evaluation

AgentOS includes a reproducible evaluation pipeline for:

Anomaly detection precision & recall
Research faithfulness
Report completeness
Verification accuracy
Roadmap
 Multi-agent orchestration
 Web research & RAG
 Autonomous data analysis
 Verification & reflection
 Human-in-the-loop
 Sandboxed code execution
 Authentication & authorization
 PDF report generation
 Advanced observability
 Dependency-aware task DAGs
Author

Ajanta Luhana
AI Engineer focused on Generative AI, LLMs, RAG, AI Agents, Computer Vision, and AI Automation.

AgentOS — Plan. Execute. Verify. Report.