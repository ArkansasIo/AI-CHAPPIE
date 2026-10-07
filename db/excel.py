"""
db.excel -- Excel (.xlsx) import/export, and SQL <-> Excel synchronization.

This module bridges the SQL database (`db.database.Database`) and Excel
workbooks, using `openpyxl`:

* `write_workbook(path, sheets)`      -- write a dict of {sheet_name: rows}
* `read_workbook(path)`               -- read an .xlsx into {sheet_name: rows}
* `export_query(db, sql, path, ...)`  -- run a SELECT and export the result
* `import_sheet(db, table, path, ...)`-- load a worksheet into a SQL table
* `db_to_excel(db, path)`             -- dump EVERY table to its own worksheet
* `excel_to_db(path, db)`             -- create/insert tables from a workbook
* `sync(db, path)`                    -- two-way convenience reference

Rows are represented as lists of dicts (same shape as `Database.query`), so the
Excel layer is a drop-in mirror of the SQL layer. Formatting (bold headers,
column widths, autofilter, freeze panes) is applied for usability.
"""

from __future__ import annotations

import os

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

__all__ = [
    "write_workbook",
    "read_workbook",
    "export_query",
    "import_sheet",
    "db_to_excel",
    "excel_to_db",
    "dictionary_to_excel",
    "sync",
]

_HEADER_FONT = Font(bold=True, color="FFFFFF")
_HEADER_FILL = PatternFill("solid", fgColor="2F5597")
_NULL = ""


def _sanitize(value):
    """Excel cannot store arbitrary nested objects; coerce to a writable scalar."""
    if value is None:
        return _NULL
    if isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def _fit_columns(ws) -> None:
    """Auto-size columns to their content (bounded)."""
    for col in ws.columns:
        length = max((len(str(c.value)) if c.value is not None else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(
            max(length + 2, 8), 48)


def _format_header(ws) -> None:
    """Style the first row as a header and add a freeze pane."""
    for cell in ws[1]:
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    if ws.max_row >= 1 and ws.max_column >= 1:
        ws.auto_filter.ref = ws.dimensions


# ---------------------------------------------------------------------------
# Low-level read / write
# ---------------------------------------------------------------------------
def write_workbook(path: str, sheets: dict) -> str:
    """Write `{sheet_name: rows}` to an .xlsx file.

    Each `rows` is a list of dicts (or a list of lists). Dict rows share a
    header taken from the union of keys, preserving key order of the first row.
    Returns the path written.
    """
    wb = Workbook()
    wb.remove(wb.active)          # drop the default empty sheet

    for name, rows in sheets.items():
        ws = wb.create_sheet(title=str(name)[:31] or "Sheet")
        rows = list(rows)
        if not rows:
            ws.append(["<empty>"])
            continue
        if isinstance(rows[0], dict):
            headers = list(rows[0].keys())
            ws.append(headers)
            for r in rows:
                ws.append([_sanitize(r.get(h)) for h in headers])
        else:                      # list of lists
            for r in rows:
                ws.append([_sanitize(v) for v in r])
        _format_header(ws)
        _fit_columns(ws)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    wb.save(path)
    return path


def read_workbook(path: str) -> dict:
    """Read an .xlsx into `{sheet_name: [ {header: value}, ... ]}`.

    The first row of each sheet is treated as the header.
    """
    wb = load_workbook(path, data_only=True)
    out = {}
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            out[ws.title] = []
            continue
        headers = [str(h) if h is not None else f"col{i}"
                   for i, h in enumerate(rows[0])]
        records = []
        for row in rows[1:]:
            if all(v is None for v in row):
                continue
            records.append({h: v for h, v in zip(headers, row)})
        out[ws.title] = records
    return out


# ---------------------------------------------------------------------------
# SQL <-> Excel
# ---------------------------------------------------------------------------
def export_query(db, sql: str, path: str, sheet_name: str = "Query",
                 params=()) -> dict:
    """Run a SELECT and write the result set to an .xlsx file.

    Returns the query rows too, for convenience.
    """
    rows = db.query(sql, params)
    write_workbook(path, {sheet_name: rows})
    return {"path": path, "sheet": sheet_name, "n_rows": len(rows)}


def import_sheet(db, table: str, path: str, sheet_name: str | None = None,
                 create_if_missing: bool = True) -> int:
    """Load a worksheet into a SQL table (INSERT OR REPLACE by row).

    If the table does not exist and `create_if_missing` is set, it is created
    with TEXT/REAL/INTEGER column types inferred from the sheet's header.

    Returns the number of rows inserted.
    """
    data = read_workbook(path)
    if not data:
        return 0
    if sheet_name is None:
        sheet_name = next(iter(data))
    rows = data[sheet_name]
    if not rows:
        return 0

    headers = list(rows[0].keys())

    existing = db.tables()
    if table not in existing:
        if not create_if_missing:
            raise ValueError(f"Table '{table}' does not exist.")
        cols = ", ".join(f'"{h}" TEXT' for h in headers)
        db.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({cols});')
    else:
        # Use the real table columns to avoid inserting unknown headers.
        info = db.query(f'PRAGMA table_info("{table}");')
        real_cols = [r["name"] for r in info]
        headers = [h for h in headers if h in real_cols]

    placeholders = ", ".join("?" for _ in headers)
    col_list = ", ".join(f'"{h}"' for h in headers)
    sql = f'INSERT OR REPLACE INTO "{table}" ({col_list}) VALUES ({placeholders});'
    db.executemany(sql, [[r.get(h) for h in headers] for r in rows])
    return len(rows)


def db_to_excel(db, path: str, tables=None, include_queries: bool = False) -> str:
    """Export every (or selected) SQL table to its own worksheet.

    With `include_queries=True`, extra worksheets with analytical reports
    (`run_summary`, `gate_accuracy_report`, `windows_ranked`) are added.
    """
    tables = tables or db.tables()
    sheets = {t: db.query(f"SELECT * FROM {t};") for t in tables}
    if include_queries:
        sheets["_run_summary"] = db.run_summary()
        sheets["_gate_accuracy"] = db.gate_accuracy_report()
        sheets["_window_ranking"] = db.windows_ranked()
    return write_workbook(path, sheets)


def excel_to_db(path: str, db, tables=None, create_if_missing: bool = True) -> dict:
    """Load every (or selected) worksheet of a workbook into matching SQL tables.

    Tables are imported in *dependency order* (referenced parents before
    children) so that foreign-key constraints are satisfied. Any sheet whose
    name starts with `_` is treated as an analytical report and skipped.

    Returns `{sheet_name: rows_loaded}`.
    """
    data = read_workbook(path)
    requested = tables or list(data.keys())

    # Dependency order: parents before the tables that reference them.
    dependency_order = [
        "gates", "datasets", "models", "hyperparameters",
        "training_runs", "metrics", "epoch_history", "gate_experiments",
    ]
    selected = [t for t in dependency_order if t in requested and t in data]
    # Append any requested tables not covered above (custom user tables).
    selected += [t for t in requested if t not in dependency_order and t in data]

    result = {}
    for name in selected:
        if name.startswith("_"):
            continue
        result[name] = import_sheet(db, name, path, sheet_name=name,
                                    create_if_missing=create_if_missing)
    return result


def sync(db, path: str, direction: str = "export", tables=None) -> str:
    """Convenience two-way sync entry point.

    direction : 'export' (SQL -> Excel), 'import' (Excel -> SQL), or 'both'.
    """
    if direction in ("export", "both"):
        db_to_excel(db, path, tables=tables, include_queries=True)
    if direction in ("import", "both"):
        excel_to_db(path, db, tables=tables)
    return path


# ---------------------------------------------------------------------------
# Dictionary-specific export
# ---------------------------------------------------------------------------
def dictionary_to_excel(dictionary, path: str, include_reports: bool = True) -> str:
    """Export an English `Dictionary` database to a formatted .xlsx workbook.

    One worksheet per table (words, senses, examples, synonyms, antonyms,
    relations, etymology, pronunciations, word_frequency, anagrams) plus
    report sheets (`_parts_of_speech`, `_length_distribution`,
    `_word_summary`) when `include_reports` is set.

    Works with the `db.dictionary.Dictionary` class (which has `.tables()`,
    `.query()`, and `.stats()`).
    """
    sheets = {t: dictionary.query(f"SELECT * FROM {t};")
              for t in dictionary.tables()}

    if include_reports:
        sheets["_parts_of_speech"] = dictionary.query(
            "SELECT * FROM v_parts_of_speech;")
        sheets["_length_distribution"] = dictionary.query(
            "SELECT * FROM v_length_distribution;")
        # Full word summary, ordered by frequency.
        sheets["_word_summary"] = dictionary.query(
            "SELECT word, length, pos_summary, first_pos, first_definition, "
            "n_senses, n_synonyms, n_examples FROM v_word_summary "
            "ORDER BY frequency_rank IS NULL, frequency_rank, word;")

    return write_workbook(path, sheets)
