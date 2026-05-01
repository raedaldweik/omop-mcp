"""
OMOP MCP tool server.

Exposes three tools over MCP:

  - run_omop_sql       Read-only SELECT against the OMOP CDM.
  - lookup_concept     Fuzzy concept search across the OMOP vocabulary.
  - expand_concept_set concept_ancestor descendant expansion.

Connection is via DATABASE_URL (Postgres connection string). The pool is
sized small intentionally — RAM agents are short-lived and Supabase's
transaction pooler does the heavy lifting.

The read-only guard rejects any SQL containing mutation keywords before
sending it to the database.

Speaks MCP over Streamable HTTP on PORT (default 8000) at BASE_PATH
(default /mcp). RAM's Container MCP template connects to this endpoint.
"""
from __future__ import annotations
import os
import re
import sys
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from mcp.server.fastmcp import FastMCP


# ─── Config ──────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    sys.exit("FATAL: DATABASE_URL environment variable is required.")

PORT      = int(os.environ.get("PORT", "8000"))
BASE_PATH = os.environ.get("BASE_PATH", "/mcp")
MAX_ROWS  = int(os.environ.get("MAX_ROWS", "1000"))

# Read-only guard. Reject any SQL containing these tokens (case-insensitive,
# word-boundary). Coarse, but it stops the obvious things.
FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|"
    r"ATTACH|DETACH|GRANT|REVOKE|VACUUM|COMMENT|MERGE|CALL|DO)\b",
    re.IGNORECASE,
)


# ─── Connection pool ─────────────────────────────────────────────

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            DATABASE_URL,
            min_size=1,
            max_size=4,
            open=True,
            kwargs={"row_factory": dict_row},
        )
    return _pool


# ─── MCP server setup ────────────────────────────────────────────
#
# host=0.0.0.0 so the container is reachable from the cluster network.
# streamable_http_path sets the URL path the transport listens on.

mcp = FastMCP(
    "omop-cdm",
    host="0.0.0.0",
    port=PORT,
    streamable_http_path=BASE_PATH,
)


# ─── Tool 1: run_omop_sql ────────────────────────────────────────

@mcp.tool()
def run_omop_sql(sql: str, purpose: str = "") -> dict[str, Any]:
    """
    Execute a read-only SELECT against the OMOP CDM (Postgres).

    Use this for any cohort count, characterization, or exploratory
    question the other tools don't handle directly. The CDM contains
    standard OMOP v5.4 tables: person, observation_period,
    visit_occurrence, condition_occurrence, drug_exposure, measurement,
    procedure_occurrence, death, concept, concept_ancestor,
    concept_relationship, cdm_source.

    Always JOIN the `concept` table when displaying clinical results so
    the user sees concept_name, not raw concept_id. Mutating statements
    (INSERT/UPDATE/DELETE/DROP/etc.) are blocked. Default row limit is
    1000; you may set a smaller LIMIT in the query itself.

    Args:
        sql:     The SELECT statement to execute.
        purpose: One-line description of what this query computes
                 (surfaced in the activity log; helpful for auditability).

    Returns:
        {columns, rows, row_count, sql, purpose} on success,
        or {error, sql} on failure.
    """
    if FORBIDDEN_SQL.search(sql):
        return {
            "error": "Only read-only SELECT queries are allowed.",
            "sql": sql,
        }

    sql_clean = sql.strip().rstrip(";")
    if "LIMIT" not in sql_clean.upper():
        sql_clean = f"{sql_clean} LIMIT {MAX_ROWS}"

    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_clean)
                rows = cur.fetchmany(MAX_ROWS)
                cols = [d.name for d in cur.description] if cur.description else []
        # Coerce non-JSON-serializable types (dates, Decimal) to strings.
        rows = [
            {k: (v if _json_safe(v) else str(v)) for k, v in r.items()}
            for r in rows
        ]
        return {
            "sql":       sql_clean,
            "purpose":   purpose,
            "columns":   cols,
            "rows":      rows,
            "row_count": len(rows),
        }
    except psycopg.Error as e:
        return {"error": str(e), "sql": sql_clean}


def _json_safe(v: Any) -> bool:
    return v is None or isinstance(v, (bool, int, float, str))


# ─── Tool 2: lookup_concept ──────────────────────────────────────

@mcp.tool()
def lookup_concept(
    term: str,
    domain: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    """
    Search the OMOP standardized vocabulary for concepts matching a
    clinical term — e.g. 'metformin', 'type 2 diabetes', 'HbA1c'.

    Returns the closest matches (case-insensitive substring match,
    shortest names first since they tend to be the most general /
    standard concepts). Optionally filter by OMOP domain
    (Condition, Drug, Measurement, Procedure, Observation, etc.).

    Always call this BEFORE writing SQL or building a cohort that
    references a clinical entity by name. Never guess concept_ids.

    Args:
        term:   Free-text clinical term to search for.
        domain: Optional OMOP domain filter ('Condition', 'Drug',
                'Measurement', 'Procedure', etc.).
        limit:  Max concepts to return (default 10, max 50).

    Returns:
        {concepts: [{concept_id, concept_name, domain_id,
                     vocabulary_id, concept_class_id,
                     standard_concept}], count}
    """
    limit = max(1, min(int(limit or 10), 50))
    pattern = f"%{term}%"

    if domain:
        sql = (
            "SELECT concept_id, concept_name, domain_id, vocabulary_id, "
            "       concept_class_id, standard_concept "
            "FROM concept "
            "WHERE concept_name ILIKE %s AND domain_id = %s "
            "ORDER BY LENGTH(concept_name) ASC LIMIT %s"
        )
        params: tuple = (pattern, domain, limit)
    else:
        sql = (
            "SELECT concept_id, concept_name, domain_id, vocabulary_id, "
            "       concept_class_id, standard_concept "
            "FROM concept "
            "WHERE concept_name ILIKE %s "
            "ORDER BY LENGTH(concept_name) ASC LIMIT %s"
        )
        params = (pattern, limit)

    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return {"concepts": rows, "count": len(rows)}
    except psycopg.Error as e:
        return {"error": str(e), "term": term}


# ─── Tool 3: expand_concept_set ──────────────────────────────────

@mcp.tool()
def expand_concept_set(
    concept_id: int,
    include_self: bool = True,
    max_levels: int | None = None,
) -> dict[str, Any]:
    """
    Return all descendant concepts of a given concept via the OMOP
    concept_ancestor table. Use this for class-to-ingredient expansion
    (e.g., 'SGLT2 inhibitors' parent class → individual ingredients) or
    parent-condition expansion (e.g., 'diabetes mellitus' → all
    diabetes subtypes).

    Always call this BEFORE using a parent concept in SQL or a cohort
    definition — operating on a parent concept directly will miss
    everything coded at the descendant level.

    Args:
        concept_id:   The ancestor concept_id to expand from.
        include_self: If True (default), the ancestor itself is
                      included in the result.
        max_levels:   Optional cap on hierarchy depth (None = no cap).

    Returns:
        {ancestor: int, descendants: [...], count: int}
    """
    min_sep = 0 if include_self else 1
    max_clause = ""
    params: list[Any] = [concept_id, min_sep]
    if max_levels is not None:
        max_clause = " AND ca.min_levels_of_separation <= %s"
        params.append(int(max_levels))

    sql = (
        "SELECT DISTINCT c.concept_id, c.concept_name, c.domain_id, "
        "       c.vocabulary_id, c.concept_class_id, "
        "       ca.min_levels_of_separation "
        "FROM concept_ancestor ca "
        "JOIN concept c ON c.concept_id = ca.descendant_concept_id "
        "WHERE ca.ancestor_concept_id = %s "
        "  AND ca.min_levels_of_separation >= %s"
        f"{max_clause} "
        "ORDER BY ca.min_levels_of_separation ASC, c.concept_name ASC "
        "LIMIT 500"
    )

    try:
        with _get_pool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return {
            "ancestor":    concept_id,
            "descendants": rows,
            "count":       len(rows),
        }
    except psycopg.Error as e:
        return {"error": str(e), "concept_id": concept_id}


# ─── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Streamable HTTP — the modern MCP transport, served at BASE_PATH on PORT.
    # RAM's Container MCP template (Transport=HTTP, Port=8000, Base Path=/mcp)
    # connects here.
    mcp.run(transport="streamable-http")
