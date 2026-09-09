# Agentic AI Learning

Learning repo covering **Python fundamentals → advanced Python → data libraries → LangChain / RAG → LangChain v1 agents → LangGraph**.

📓 **[NOTES.md](NOTES.md)** — the full write-up of everything covered so far, with code snippets.

---

## Structure

```
Agentic_AI_Learning/
├── README.md                  you are here
├── NOTES.md                   full notes, basics -> advanced
├── .env                       API keys (git-ignored)
├── .env.example               template - copy to .env and fill in
├── .gitignore
│
├── 01-python/                     ── Part 1: Python
│   ├── requirements.txt
│   ├── 01-basics/                 syntax, variables, types, control flow,
│   │                              lists, tuples, sets, dicts, functions
│   ├── 02-oop/                    classes, inheritance, polymorphism, ABCs,
│   │                              encapsulation, magic methods, operator overloading
│   ├── 03-advanced-python/        iterators, generators, decorators
│   ├── 04-exception-handling/     try/except/else/finally, custom exceptions
│   ├── 05-modules-and-packages/   imports, packages, standard library
│   │   └── package/               sample package (__init__.py + maths.py)
│   ├── 06-file-handling/          text & binary files, os.path
│   ├── 07-logging/                levels, formats, multiple loggers, handlers
│   │   └── logging_demo.py        runnable script version
│   ├── 08-data-analysis/          numpy, pandas, reading sources, manipulation
│   └── data/                      data.csv, example.csv, test.txt, test.bin
│
├── 02-langchain/                  ── Part 2: GenAI / LangChain (LangChain 0.x)
│   ├── requirements.txt
│   ├── 01-getting-started/        prompts, models, output parsers, LCEL,
│   │                              and a full RAG pipeline
│   ├── 02-data-ingestion/         Text / PDF / Web / Arxiv document loaders
│   ├── 03-text-splitting/         recursive character, HTML header, JSON splitters
│   ├── 04-embeddings/             HuggingFace, Ollama
│   ├── 05-vector-stores/          Chroma, FAISS
│   │   └── faiss_index/           saved FAISS index (regenerable)
│   ├── 06-pydantic/               models, optional fields, nesting, Field constraints
│   ├── 07-LCEL/                   Groq chat model, message objects, prompt | model | parser
│   ├── apps/
│   │   ├── streamlit_ollama_app.py   Streamlit chat UI over local Ollama
│   │   └── langserver.py             LangServe / FastAPI API over the same chain
│   ├── chat-bot/                  multi-turn chat: message history, session ids,
│   │                              MessagesPlaceholder, trimming with trim_messages
│   ├── data/                      speech.txt, sample.pdf, attention.pdf
│   └── vectorrectriver/           Document objects, Chroma + HuggingFace embeddings,
│                                  retrievers, a RAG chain built from raw runnables
│
├── Langchainupdated/               ── Part 3: LangChain v1 (separate uv project)
│   ├── pyproject.toml              dependencies; uv.lock pins them
│   ├── .python-version             3.11
│   ├── .env                        this project's own keys (git-ignored)
│   └── updatedlangchain/
│       ├── langchain.ipynb         create_agent - the model picks the tool
│       ├── modelintegration.ipynb  init_chat_model, provider strings, stream, batch
│       ├── mesaages.ipynb          message types, metadata, text vs message prompts
│       ├── tools.ipynb             @tool, bind_tools, the tool-execution loop by hand
│       ├── structuredOutput.ipynb  with_structured_output: Pydantic / TypedDict / dataclass
│       └── middleware.ipynb        summarization + human-in-the-loop, checkpointers, threads
│
└── AgenticAIWorkSpace/             ── Part 4: LangGraph (separate venv)
    ├── requirements.txt            langchain, langgraph, langchain-core, langchain-community
    ├── .env                        this project's own keys (git-ignored)
    └── Langgraph-basics/
        └── simplegraph.ipynb       State, nodes, conditional edges, START/END, compile, invoke
```

Notebooks in Parts 1 and 2 are numbered in the order they should be read. Part 3's are not — read
them as `langchain` → `modelintegration` → `mesaages` → `tools` → `structuredOutput` →
`middleware`.

---

## Setup

Parts 1 and 2 share one `pip` environment. Parts 3 and 4 each have their own, because
LangChain 1.x and 0.x cannot sit in the same environment.

```bash
# 1. keys
cp .env.example .env        # then fill in your keys

# 2. Python track
pip install -r 01-python/requirements.txt

# 3. LangChain track (0.x)
pip install -r 02-langchain/requirements.txt

# 4. LangChain v1 track - uv builds .venv from pyproject.toml + uv.lock
cd Langchainupdated
uv sync

# 5. LangGraph track
cd AgenticAIWorkSpace
python -m venv venv
venv\Scripts\activate           # Git Bash: source venv/Scripts/activate
pip install -r requirements.txt
```

### API keys

Each part reads the nearest `.env`: the root one for Parts 1 and 2, `Langchainupdated/.env` for
Part 3, `AgenticAIWorkSpace/.env` for Part 4. They hold the same keys — the projects are
self-contained, not differently configured. What each key unlocks:

| Key | Used by |
| --- | --- |
| `GROQ_API_KEY` | `07-LCEL/`, `chat-bot/`, `vectorrectriver/`, `apps/langserver.py`, `Langchainupdated/` — hosted `ChatGroq` models |
| `GEMINI_API_KEY` | `01-getting-started/`, `Langchainupdated/` — Gemini chat models |
| `HF_TOKEN` | `04-embeddings/`, `vectorrectriver/` — HuggingFace models |
| `LANGSMITH_*` / `LANGCHAIN_API_KEY` | tracing; optional, everything runs without it |

Ollama needs no key — it runs locally.

### Running the Streamlit app

Needs [Ollama](https://ollama.com) running locally:

```bash
ollama pull gemma:2b
ollama pull nomic-embed-text        # the embedding model the notebooks use

cd 02-langchain/apps
streamlit run streamlit_ollama_app.py
```

### Running the LangServe API

Serves the §23 translation chain over HTTP. Needs `GROQ_API_KEY`:

```bash
cd 02-langchain/apps
python langserver.py                # http://127.0.0.1:8000
```

- `POST /chain/invoke` — `{"input": {"language": "Japanese", "text": "Hello"}}`
- `/chain/playground/` — built-in UI
- `/docs` — FastAPI's Swagger UI

---

## Conventions

- **Paths** — notebooks reach shared files as `../data/...`, so run them with the notebook's own
  folder as the working directory (which is what VS Code and Jupyter do by default).
- **Secrets** — one `.env` per part: the root for Parts 1-2, `Langchainupdated/.env` for Part 3,
  `AgenticAIWorkSpace/.env` for Part 4. `load_dotenv()` walks up the directory tree, so a notebook
  at any depth finds the nearest one (`load_dotenv(find_dotenv())` makes that explicit). Every
  `.env` is git-ignored; `.env.example` is not.
- **Naming** — folders and files are `kebab-case` and numbered by reading order.

---

## Notes on dependencies

- `PyMuPDF` provides the `fitz` module. **Do not `pip install fitz`** — that PyPI project is
  unrelated and will break the import.
- Embeddings need a *dedicated embedding model*. `gemma:2b` is a chat model and has no embedding
  head — `ollama pull nomic-embed-text` instead.
- `faiss_index/` and any Chroma directories are git-ignored; re-run the notebook to rebuild them.
- `OllamaEmbeddings` has **no default model** — `OllamaEmbeddings()` raises a pydantic
  `ValidationError`. Pass `model="nomic-embed-text"` explicitly.
- If Ollama calls fail with `ConnectionError`, either the server is not running, or `localhost`
  resolved to IPv6 `::1` while Ollama listens on IPv4 only. `base_url="http://127.0.0.1:11434"`
  settles the second case.
- `langserve` pulls in `fastapi` + `uvicorn`; `langchain_groq` needs `GROQ_API_KEY` in `.env`.
- `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` needs no key or server — it downloads the
  model once and runs locally. It is the quickest path when Ollama isn't running.
- `RunnableWithMessageHistory` warns that it is deprecated in favour of LangGraph persistence. It
  still works, and the notebooks use it to show the mechanics.
- **Do not install LangChain 1.x into the Part 2 environment.** The import paths moved
  (`langchain.messages`, `langchain.tools`, `langchain.chat_models`) and the classic chain helpers
  moved to `langchain-classic`. That split is why Part 3 is its own `uv` project.
- `init_chat_model("groq:...")` still needs the provider package installed — the string picks a
  class, it does not vendor the SDK.
- If both `GOOGLE_API_KEY` and `GEMINI_API_KEY` are set, `langchain-google-genai` uses
  `GOOGLE_API_KEY` and says so in a warning.
- Agent middleware needs `checkpointer=InMemorySaver()` **and** a `thread_id` in the invoke config.
  Without both, there is no state to summarise or to pause and resume.
- `draw_mermaid_png()` renders through the remote Mermaid.INK API, so graph pictures need network
  access. `get_graph().draw_ascii()` works offline.
