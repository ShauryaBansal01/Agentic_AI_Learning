# Agentic AI Learning

Learning repo covering **Python fundamentals → advanced Python → data libraries → LangChain / RAG**.

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
└── 02-langchain/                  ── Part 2: GenAI / LangChain
    ├── requirements.txt
    ├── 01-getting-started/        prompts, models, output parsers, LCEL,
    │                              and a full RAG pipeline
    ├── 02-data-ingestion/         Text / PDF / Web / Arxiv document loaders
    ├── 03-text-splitting/         recursive character, HTML header, JSON splitters
    ├── 04-embeddings/             HuggingFace, Ollama
    ├── 05-vector-stores/          Chroma, FAISS
    │   └── faiss_index/           saved FAISS index (regenerable)
    ├── 06-pydantic/               models, optional fields, nesting, Field constraints
    ├── 07-LCEL/                   Groq chat model, message objects, prompt | model | parser
    ├── apps/
    │   ├── streamlit_ollama_app.py   Streamlit chat UI over local Ollama
    │   └── langserver.py             LangServe / FastAPI API over the same chain
    ├── chat-bot/                  multi-turn chat: message history, session ids,
    │                              MessagesPlaceholder, trimming with trim_messages
    ├── data/                      speech.txt, sample.pdf, attention.pdf
    └── vectorrectriver/           Document objects, Chroma + HuggingFace embeddings,
                                   retrievers, a RAG chain built from raw runnables
```

Notebooks are numbered in the order they should be read within each folder.

---

## Setup

```bash
# 1. keys
cp .env.example .env        # then fill in your keys

# 2. Python track
pip install -r 01-python/requirements.txt

# 3. LangChain track
pip install -r 02-langchain/requirements.txt
```

### API keys

`.env` at the repo root holds every key. What each one unlocks:

| Key | Used by |
| --- | --- |
| `GROQ_API_KEY` | `07-LCEL/`, `chat-bot/`, `vectorrectriver/`, `apps/langserver.py` — hosted `ChatGroq` models |
| `GEMINI_API_KEY` | `01-getting-started/` — `ChatGoogleGenerativeAI` |
| `HF_TOKEN` | `04-embeddings/`, `vectorrectriver/` — HuggingFace models |
| `LANGSMITH_*` | tracing; optional, everything runs without it |

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
- **Secrets** — everything lives in the root `.env`. `load_dotenv()` walks up the directory tree,
  so any notebook at any depth picks it up. `.env` is git-ignored; `.env.example` is not.
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
