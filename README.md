# omop-mcp

MCP tool server exposing read-only access to an OMOP CDM Postgres database.

Designed to be consumed by SAS Retrieval Agent Manager (RAM) as a Container
MCP server, but speaks standard MCP over stdio so any MCP-compatible client
(Claude Desktop, custom agents, etc.) can use it.

## Tools

| Tool                  | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `run_omop_sql`        | Read-only SELECT against the OMOP CDM.                 |
| `lookup_concept`      | Fuzzy concept search across the OMOP vocabulary.       |
| `expand_concept_set`  | `concept_ancestor` descendant expansion.               |

A read-only SQL guard rejects any statement matching mutation keywords
(`INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|...`) before execution.
Default row cap of 1000 per query.

## Configuration

| Env var        | Required | Description                                       |
| -------------- | -------- | ------------------------------------------------- |
| `DATABASE_URL` | yes      | Postgres connection string (Supabase or similar). |
| `MAX_ROWS`     | no       | Hard row cap per query. Default `1000`.           |

## Build

The container is built and pushed to `ghcr.io` by GitHub Actions on every
push to `main`. See `.github/workflows/build.yml`.

After a successful build, the image is available at:

```
ghcr.io/<your-github-username>/omop-mcp:latest
```

## Local development

```bash
pip install -r requirements.txt
export DATABASE_URL="postgresql://..."
python server.py
```
