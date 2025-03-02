Below is a **detailed** `task_execution.md` (or simply `task.md`) that you can give to your coding agent in Cursor. It breaks the project into **two** distinct tools:

1. **`create_database_schema`** — A CLI tool that generates (or updates) an ArangoDB schema for the `verifaix` database and saves it to a `database_schema` collection.  
2. **`agent_memory`** — A CLI tool for retrieving conversation logs (or knowledge docs) from `verifaix`, using BM25 + vector embeddings.

Both tools reside under `src/agent_tools/` in sibling directories, similar to how `method_validator` is structured.

The plan enforces the constraints and preferences you described:

- Use **PyProject.toml** instead of `requirements.txt`.  
- Use **Loguru** for logging.  
- Keep token limit ~4000, with a best-practice approach to context management (particularly for the memory retrieval).  
- The “agent_memory” tool can be invoked by an LLM agent to query ArangoDB, returning results or error corrections.  
- The “create_database_schema” tool excludes system collections (leading underscore) and user-specified excluded collections.  
- The user can debug each tool’s main function individually.  
- We do not over-engineer or create excessive features.  

-----

## `task_execution.md` (Step-by-Step Plan)

Below is your actionable plan for the coding agent.

---

### 1. Create Basic Project Layout

**Directory Structure** under `src/agent_tools/`:

```
src/
  └─ agent_tools/
      ├─ create_database_schema/
      │   ├─ analyzer.py
      │   ├─ cli.py
      │   ├─ tool_main.py
      │   ├─ README.md
      │   └─ pyproject.toml
      └─ agent_memory/
          ├─ analyzer.py
          ├─ cli.py
          ├─ tool_main.py
          ├─ README.md
          └─ pyproject.toml
```

**Task**: The coding agent should create these folders and files (where they do not already exist).  
- `pyproject.toml` in each folder will list dependencies (including `sentence-transformers`, `loguru`, `arango`, etc.).  
- The user’s root-level `config.py` can remain in `src/search_tool/settings/config.py` or be adapted as needed.

**Notes**:

- Because we’re using `uv pip install -e .`, each sub-tool can have its own `pyproject.toml`. Or you can unify them into a single top-level. For now, the plan is to let each tool be installable.  
- The coding agent can also place a minimal `__init__.py` in each folder if needed.

---

### 2. Implement `create_database_schema` Tool

**Goal**: A command-line tool that, upon running, connects to ArangoDB (`verifaix` database), inspects collections/views, excludes system collections and user-specified ones, and **writes** the discovered schema into a `database_schema` collection.

#### 2.1 `pyproject.toml`

- Must include:
  - `arango`
  - `loguru`
  - `nltk` (if needed for text processing, like stopwords)
  - Possibly `deepmerge`, `pydantic`, and `sentence-transformers` if used
- The coding agent sets up versions as pinned or approximate. (E.g. `sentence-transformers = "^2.2.2"`).

#### 2.2 `analyzer.py`

- Houses the main logic to:
  - Connect to Arango (reading config from `config.py` or environment).
  - Discover relationships, sample rows (like your snippet).
  - Build a `SchemaDescription` data structure (like your snippet uses).
  - Exclude `_system` or other user-specified collections.  
  - Summarize everything into a dictionary or pydantic model.

**Key Methods** (example names, at agent’s discretion):

1. `load_config()` – loads or merges the user’s `config.py`.  
2. `connect_arango(arango_config)` – returns a `db` handle.  
3. `build_schema(db, config) -> dict` – enumerates collections, views, analyzers.  
4. `store_schema(db, schema_dict) -> bool` – inserts into `database_schema` collection.  

#### 2.3 `cli.py`

- Uses `argparse` to define subcommands or flags, e.g.:
  - `--exclude-collections` to override or append to config?
  - `--dry-run` to not write results to DB?
  - `--json` to print the final schema to stdout instead of storing?  
- Minimal usage example:
  ```
  usage: create-database-schema [--json] [--exclude-collections col1,col2]

  optional arguments:
    --json         print schema to stdout
    --exclude-collections ...
  ```

#### 2.4 `tool_main.py`

- A single `main()` function. 
- Gathers config, calls `build_schema(...)`, prints or saves the result.  
- Also logs progress using **loguru**.  
- This is the function you can run directly (e.g. `python tool_main.py --exclude-collections foo,bar`).  

#### 2.5 `README.md`

- Explains how to install (e.g. `uv pip install -e .` in the `create_database_schema` folder).
- Usage examples:
  - `create-database-schema --json` => prints the discovered schema.  
  - `create-database-schema` => writes results into the `database_schema` collection.

**Note**: Reuse as much from your existing `generate_schema_for_llm.py` snippet as is practical, but keep it simpler if you like (especially skipping advanced relationships if not needed). Make sure the **excluded** system collections and user-specified ones are indeed skipped.  

---

### 3. Implement `agent_memory` Tool

**Goal**: Provide a CLI that the LLM can call to query ArangoDB for conversation logs or knowledge docs, using BM25 + vector embeddings. The result is printed to stdout as JSON or textual lines. The LLM or user can parse the output.

#### 3.1 `pyproject.toml`

- Must include:
  - `arango`
  - `loguru`
  - `sentence-transformers` (for embedding the user’s query).
  - Possibly `regex` or `pydantic` if you do extra text processing.  
- If you want to store embeddings for each doc, ensure you have a field in the doc (like `doc.embedding = [floats]`). If you do that offline, the tool can just query.  
- Or the tool might embed the user’s query on the fly, then do an AQL `COSINE_SIMILARITY` check.

#### 3.2 `analyzer.py`

Core logic for memory retrieval:

- **`load_config()`** to read from `config.py` or environment.  
- **`connect_arango(arango_config)`** => returns `db`.  
- **`embed_query(query_str, model)`** => returns vector.  
- **`search_memory(db, query_str, top_n, thresholds) -> list[dict]`**:
  1. Possibly embed the query.  
  2. Construct an AQL that merges BM25 & vector similarity.  
  3. Return the top results (like `[{"_key": "...", "content": "...", "score": 0.87}, ...]`).  
- **`get_conversation_history(db, session_id, limit=... ) -> list[dict]`**: if you plan to fetch logs by session.  
- Possibly define a method to handle **token budget** ~4000. But for now, you can store that logic in `cli.py` or `tool_main.py`.

#### 3.3 `cli.py`

- Provide arguments:
  - `--search "text"`: let the user or LLM do a combined BM25+vector search.  
  - `--session-id "xyz"` + `--list-logs`: fetch conversation logs for a session.  
  - `--json`: output in JSON.  
  - `--limit` / `--top-n`: how many results.  

**Usage**:

```
agent-memory --search "Where can I find info about Outlook issues?" --json
agent-memory --session-id "abc123" --list-logs
```

- The agent can parse the output. If empty or errors, the agent might do a retry.

#### 3.4 `tool_main.py`

- Has `main()` that:
  1. Loads config,
  2. Connects to Arango,
  3. Reads CLI args (search or list-logs),
  4. Calls the relevant analyzer function,
  5. Prints or logs the results.  

**Error Correction Approach**:
- If the AQL query returns empty or an error, you can do minimal logic to log an error or print something like:  
  ```json
  {
    "error": "No results found",
    "suggestion": "Try adjusting the query threshold or rechecking your search text"
  }
  ```
  The LLM can interpret this and possibly retry with a different query.

#### 3.5 `README.md`

- Explains usage.  
- Provide example calls, e.g.:
  ```
  agent-memory --search "why is Visual Studio slow" --json
  ```
  Returns a list of doc references.

---

### 4. Common or Shared Code

If you have repeated logic (like `connect_arango` or `load_config`), consider placing it in a shared folder, e.g. `src/search_tool/shared_utils/`. That’s where your `get_project_root`, etc., might live. The coding agent should reuse it rather than rewriting from scratch.

---

### 5. Token Limit & Context

Implement a best-practice approach:

- **Memory retrieval**: If you’re returning many results, chunk them or limit top N.  
- The agent can then parse the top results into its own context.  
- For conversation logs, keep them separate from large “fact knowledge.” The agent can do “agent-memory --list-logs --session-id=xxx” to get the last ~N messages, or chunk it if more than 4000 tokens worth.  

You do not need an elaborate concurrency or “fancy” approach yet—just keep it simple.

---

### 6. Testing / Debugging

- Each tool has a `tool_main.py` so you can do: `python tool_main.py --some-flag`.  
- Minimal error checking: if an Arango connection fails, log with `loguru.error(...)`.  
- No formal test harness is needed right now.

---

### 7. Example Minimal Commands

1. **Install** (in each tool directory):
   ```bash
   cd src/agent_tools/create_database_schema
   uv venv --python=3.10.16 .venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. **Run** (to see help):
   ```
   create-database-schema --help
   ```
3. **Agent calls** (for memory):
   ```
   agent-memory --search "Outlook crash" --json
   ```

---

### 8. Additional Developer Flow (Optional)

If you want a dev flow with `--debug`, `--cache-clear`, etc., the coding agent can add subcommands. But we keep it minimal unless needed.

---

### 9. Future Possibilities

- Merging with your advanced `generate_schema_for_llm.py` logic if you want a richer schema generator.  
- Adding partial pagination for memory results or function-calling approach for the LLM.  
- Implementing an “embedding store” if you need to embed entire documents and keep them updated.

---

## Conclusion

**This** `task.md` provides step-by-step instructions for the coding agent to:

1. Set up **two** sub-tools: `create_database_schema` and `agent_memory`.  
2. Use the style from `method_validator` (with `analyzer.py`, `cli.py`, etc.).  
3. Keep usage simple, rely on `loguru` for logging, store config in `config.py`, and reference it in each tool.  
4. Provide a minimal step for context management (~4000 token budget).  
5. Support quick debugging by having each tool’s `main()` callable.

---

