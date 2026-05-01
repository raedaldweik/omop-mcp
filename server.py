"""
OMOP MCP tool server.

Exposes four tools over MCP (Streamable HTTP transport):

  - run_omop_sql       Read-only SELECT against the OMOP CDM (Postgres).
  - lookup_concept     Fuzzy concept search across the OMOP vocabulary.
  - expand_concept_set concept_ancestor descendant expansion.
  - render_chart       Render a matplotlib chart from data and return as
                       inline image content for the agent to embed.

Connection is via DATABASE_URL. The pool is sized small intentionally —
RAM agents are short-lived and Supabase's transaction pooler does the
heavy lifting.

The read-only guard rejects any SQL containing mutation keywords before
sending it to the database.
"""
from __future__ import annotations
import io
import os
import re
import sys
from typing import Any, Literal

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcp.server.fastmcp import FastMCP, Image


# ─── Config ──────────────────────────────────────────────────────

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    sys.exit("FATAL: DATABASE_URL environment variable is required.")

PORT      = int(os.environ.get("PORT", "8000"))
BASE_PATH = os.environ.get("BASE_PATH", "/mcp")
MAX_ROWS  = int(os.environ.get("MAX_ROWS", "1000"))

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
        {concepts: [...], count: int}
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
    concept_ancestor table. Use for class-to-ingredient expansion (e.g.,
    'SGLT2 inhibitors' → individual ingredients) or parent-condition
    expansion (e.g., 'diabetes mellitus' → all subtypes).

    Always call this BEFORE using a parent concept in SQL — operating on
    a parent concept directly will miss everything coded at the
    descendant level.

    Args:
        concept_id:   The ancestor concept_id to expand from.
        include_self: If True (default), include the ancestor itself.
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


# ─── Tool 4: render_chart ────────────────────────────────────────

# SAS-aligned palette so charts feel native to the demo.
_PALETTE = ["#0072CE", "#A42548", "#0C6E7A", "#B8620A", "#7B61FF",
            "#DAB866", "#3CC4CF", "#1A7A45"]


@mcp.tool()
def render_chart(
    chart_type: Literal["bar", "barh", "line", "pie"],
    title: str,
    labels: list[str],
    values: list[float],
    x_label: str = "",
    y_label: str = "",
) -> Image:
    """
    Render a matplotlib chart and return it as an inline image. Use this
    when the user asks to visualize, plot, compare, or graph data — or
    when a chart would substantially clarify a result.

    Pass a parallel pair of `labels` and `values` arrays of equal length.
    For ranked categorical data (top-N conditions, drug counts, cohort
    sizes), use `barh` — horizontal bars handle long category names
    cleanly. Use `bar` only when category names are short. Use `line`
    for ordered/temporal data. Use `pie` only for a small number of
    parts-of-a-whole categories (≤6).

    Args:
        chart_type: One of "bar", "barh", "line", "pie".
        title:      Chart title (shown above the plot).
        labels:     Category labels (x-axis for bar/line, segment names for pie).
        values:     Numeric values, parallel to labels.
        x_label:    Optional x-axis label.
        y_label:    Optional y-axis label.

    Returns:
        Inline PNG image rendered for the chat UI.
    """
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    if not labels:
        raise ValueError("labels and values must not be empty")

    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)

    if chart_type == "bar":
        ax.bar(labels, values, color=_PALETTE[0], edgecolor="none")
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        ax.tick_params(axis="x", rotation=30)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    elif chart_type == "barh":
        # Reverse so largest is on top
        ax.barh(labels[::-1], values[::-1], color=_PALETTE[0], edgecolor="none")
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)

    elif chart_type == "line":
        ax.plot(labels, values, color=_PALETTE[0], linewidth=2.0, marker="o")
        if x_label: ax.set_xlabel(x_label)
        if y_label: ax.set_ylabel(y_label)
        ax.tick_params(axis="x", rotation=30)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(alpha=0.25, linewidth=0.6)

    elif chart_type == "pie":
        colors = (_PALETTE * ((len(labels) // len(_PALETTE)) + 1))[:len(labels)]
        ax.pie(values, labels=labels, colors=colors,
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1})
        ax.set_aspect("equal")

    ax.set_title(title, fontsize=12, loc="left")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return Image(data=buf.getvalue(), format="png")


# ─── Entrypoint ──────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
