"""
db.database -- a SQL (SQLite) database for the neural-network / logic-gate project.

This module stores and queries everything the library works with:

    gates            : the 16 two-input Boolean functions (truth tables)
    gate_experiments : results of training a network to realize a gate
    models           : saved model architectures + parameter counts
    training_runs    : epochs, loss, accuracy per run
    datasets         : dataset metadata (shape, source)
    hyperparameters  : key/value hyperparameters per model
    metrics          : generic metric log (name, value, step)

Design
------
* `Database` wraps a single SQLite file (default `db/nn.db`).
* Schema is created idempotently via `init_schema()` from `SCHEMA_SQL`.
* Every table has typed columns + constraints (PRIMARY KEY, FOREIGN KEY,
  UNIQUE, CHECK) so the DB itself enforces integrity.
* A `sql()` escape hatch runs arbitrary SELECTs and returns list-of-dicts.
* `to_excel()` / `from_excel()` delegate to `db.excel` for round-tripping.

Pure standard library (`sqlite3`) -- no external dependency.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

__all__ = ["Database", "SCHEMA_SQL", "DEFAULT_DB_PATH"]

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "nn.db")


# ---------------------------------------------------------------------------
# Schema (data-definition language)
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 1. Boolean gates and their 4-bit truth tables ----------------------------
CREATE TABLE IF NOT EXISTS gates (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    n_inputs      INTEGER NOT NULL DEFAULT 2 CHECK (n_inputs >= 1),
    truth_bits    TEXT    NOT NULL CHECK (length(truth_bits) = (1 << n_inputs)),
    is_universal  INTEGER NOT NULL DEFAULT 0 CHECK (is_universal IN (0, 1)),
    linearly_separable INTEGER NOT NULL DEFAULT 1
                        CHECK (linearly_separable IN (0, 1)),
    description   TEXT
);

-- 2. Datasets used for training --------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    n_samples     INTEGER NOT NULL CHECK (n_samples >= 0),
    n_features    INTEGER NOT NULL CHECK (n_features >= 1),
    n_classes     INTEGER NOT NULL CHECK (n_classes >= 1),
    source        TEXT    DEFAULT 'synthetic'
);

-- 3. Saved models ----------------------------------------------------------
CREATE TABLE IF NOT EXISTS models (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    architecture  TEXT    NOT NULL,            -- e.g. '2-8-8-1'
    n_params      INTEGER NOT NULL DEFAULT 0 CHECK (n_params >= 0),
    loss_name     TEXT,
    task          TEXT    DEFAULT 'classification'
                        CHECK (task IN ('classification', 'regression', 'logic')),
    path          TEXT,                        -- optional .pkl path
    created_at    TEXT    DEFAULT (datetime('now'))
);

-- 4. Training runs (one row per fit() call) --------------------------------
CREATE TABLE IF NOT EXISTS training_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    dataset_id    INTEGER          REFERENCES datasets(id) ON DELETE SET NULL,
    optimizer     TEXT    NOT NULL DEFAULT 'adam',
    learning_rate REAL    NOT NULL DEFAULT 0.001 CHECK (learning_rate > 0),
    epochs        INTEGER NOT NULL DEFAULT 1 CHECK (epochs >= 1),
    batch_size    INTEGER NOT NULL DEFAULT 32 CHECK (batch_size >= 1),
    final_loss    REAL,
    final_accuracy REAL CHECK (final_accuracy IS NULL OR
                              (final_accuracy >= 0 AND final_accuracy <= 1)),
    started_at    TEXT    DEFAULT (datetime('now')),
    finished_at   TEXT
);

-- 5. Hyperparameters (key/value, per model) --------------------------------
CREATE TABLE IF NOT EXISTS hyperparameters (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id      INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    key           TEXT    NOT NULL,
    value         TEXT,
    UNIQUE (model_id, key)
);

-- 6. Logic-gate learning experiments ---------------------------------------
CREATE TABLE IF NOT EXISTS gate_experiments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    gate_name     TEXT    NOT NULL REFERENCES gates(name) ON DELETE CASCADE,
    model_id      INTEGER          REFERENCES models(id) ON DELETE SET NULL,
    hidden_units  INTEGER NOT NULL DEFAULT 0 CHECK (hidden_units >= 0),
    tau           REAL,                          -- smoothness for soft gates
    accuracy      REAL    NOT NULL CHECK (accuracy >= 0 AND accuracy <= 1),
    learned_at    TEXT    DEFAULT (datetime('now'))
);

-- 7. Generic metric log ----------------------------------------------------
CREATE TABLE IF NOT EXISTS metrics (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER          REFERENCES training_runs(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,
    value         REAL    NOT NULL,
    step          INTEGER NOT NULL DEFAULT 0
);

-- 8. Per-epoch history (fine-grained training curves) ----------------------
CREATE TABLE IF NOT EXISTS epoch_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    epoch         INTEGER NOT NULL CHECK (epoch >= 1),
    loss          REAL,
    accuracy      REAL,
    val_loss      REAL,
    val_accuracy  REAL,
    UNIQUE (run_id, epoch)
);

-- Helpful indexes for the common joins / lookups.
CREATE INDEX IF NOT EXISTS idx_runs_model   ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_dataset ON training_runs(dataset_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run  ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_epochs_run   ON epoch_history(run_id);
CREATE INDEX IF NOT EXISTS idx_gateexp_gate ON gate_experiments(gate_name);
"""


class Database:
    """A thin, safe wrapper around a SQLite database file."""

    def __init__(self, path: str = DEFAULT_DB_PATH, echo: bool = False):
        self.path = path
        self.echo = echo
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row          # dict-like rows
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------
    def init_schema(self) -> None:
        """Create all tables/indexes if they do not already exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @contextmanager
    def transaction(self):
        """Context manager: commit on success, roll back on exception."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- low-level helpers -------------------------------------------------
    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        """Run an INSERT/UPDATE/DELETE/DDL statement and commit."""
        if self.echo:
            print(f"[SQL] {sql.strip()}  {params}")
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def executemany(self, sql: str, seq_of_params) -> sqlite3.Cursor:
        """Run a parameterized statement for many rows, in one transaction."""
        if self.echo:
            print(f"[SQL] {sql.strip()}  x{len(list(seq_of_params))}")
            seq_of_params = list(seq_of_params)
        cur = self.conn.executemany(sql, seq_of_params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params=()) -> list:
        """Run a SELECT and return a list of `dict` rows."""
        if self.echo:
            print(f"[SQL] {sql.strip()}  {params}")
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def query_one(self, sql: str, params=()):
        """Run a SELECT and return the first row as a dict (or None)."""
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params=()):
        """Return a single scalar value from a SELECT."""
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else row[0]

    def sql(self, sql: str, params=()):
        """Escape hatch: detect the statement type and dispatch."""
        head = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if head in ("SELECT", "WITH", "PRAGMA", "EXPLAIN"):
            return self.query(sql, params)
        return self.execute(sql, params)

    def tables(self) -> list:
        """List user table names."""
        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        return [r["name"] for r in rows]

    def count(self, table: str) -> int:
        """Row count of a table."""
        return int(self.scalar(f"SELECT COUNT(*) FROM {table};")) if table in self.tables() else 0

    # =====================================================================
    # Domain operations
    # =====================================================================
    # -- gates -------------------------------------------------------------
    def seed_gates(self) -> int:
        """Populate the `gates` table from `nn.gates.GATE_TABLE` (idempotent)."""
        from nn.gates import GATE_TABLE

        universal = {"NAND", "NOR"}
        not_separable = {"XOR", "XNOR"}
        rows = []
        for name, bits in GATE_TABLE.items():
            if len(bits) != 4:              # only the 2-input gates live here
                continue
            bit_str = "".join(str(b) for b in bits)
            rows.append((
                name, 2, bit_str,
                int(name in universal),
                int(name not in not_separable),
                f"{name} two-input Boolean gate",
            ))
        self.executemany(
            "INSERT OR IGNORE INTO gates "
            "(name, n_inputs, truth_bits, is_universal, linearly_separable, description) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            rows,
        )
        return len(rows)

    def add_gate(self, name, truth_bits, n_inputs=2, is_universal=False,
                 linearly_separable=True, description=None) -> int:
        """Insert a gate and return its id."""
        cur = self.execute(
            "INSERT INTO gates (name, n_inputs, truth_bits, is_universal, "
            "linearly_separable, description) VALUES (?, ?, ?, ?, ?, ?);",
            (name, n_inputs, "".join(str(b) for b in truth_bits),
             int(is_universal), int(linearly_separable), description))
        return cur.lastrowid

    def get_gate(self, name: str):
        """Fetch a gate row by name."""
        return self.query_one("SELECT * FROM gates WHERE name = ?;", (name,))

    def gates_output(self, value: int = 1) -> list:
        """Gates whose truth table contains `value` (e.g. all gates with a 1)."""
        return self.query(
            "SELECT name, truth_bits FROM gates WHERE truth_bits LIKE ? "
            "ORDER BY name;", (f"%{value}%",))

    def universal_gates(self) -> list:
        """Return the functionally complete single gates."""
        return self.query(
            "SELECT name, truth_bits FROM gates WHERE is_universal = 1 "
            "ORDER BY name;")

    # -- models ------------------------------------------------------------
    def add_model(self, name, architecture, n_params=0, loss_name=None,
                  task="logic", path=None, **hyperparameters) -> int:
        """Insert (or replace) a model and optional hyperparameters."""
        self.execute(
            "INSERT OR REPLACE INTO models "
            "(name, architecture, n_params, loss_name, task, path) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (name, architecture, n_params, loss_name, task, path))
        model_id = int(self.scalar("SELECT id FROM models WHERE name = ?;", (name,)))
        for key, value in hyperparameters.items():
            self.set_hyperparameter(model_id, key, value)
        return model_id

    def set_hyperparameter(self, model_id: int, key: str, value) -> None:
        """Upsert a hyperparameter for a model."""
        self.execute(
            "INSERT INTO hyperparameters (model_id, key, value) VALUES (?, ?, ?) "
            "ON CONFLICT(model_id, key) DO UPDATE SET value = excluded.value;",
            (model_id, key, str(value)))

    def get_hyperparameters(self, model_id: int) -> dict:
        """Return `{key: value}` for a model."""
        return {r["key"]: r["value"] for r in self.query(
            "SELECT key, value FROM hyperparameters WHERE model_id = ?;", (model_id,))}

    def model_names(self) -> list:
        """All model names with their architecture."""
        return self.query("SELECT name, architecture, n_params FROM models ORDER BY name;")

    # -- datasets ----------------------------------------------------------
    def add_dataset(self, name, n_samples, n_features, n_classes, source="synthetic") -> int:
        """Insert (or replace) a dataset record and return its id."""
        self.execute(
            "INSERT OR REPLACE INTO datasets (name, n_samples, n_features, n_classes, source) "
            "VALUES (?, ?, ?, ?, ?);",
            (name, n_samples, n_features, n_classes, source))
        return int(self.scalar("SELECT id FROM datasets WHERE name = ?;", (name,)))

    # -- training runs -----------------------------------------------------
    def start_run(self, model_id, dataset_id=None, optimizer="adam",
                  learning_rate=1e-3, epochs=1, batch_size=32) -> int:
        """Create a training-run row and return its id."""
        cur = self.execute(
            "INSERT INTO training_runs "
            "(model_id, dataset_id, optimizer, learning_rate, epochs, batch_size) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (model_id, dataset_id, optimizer, learning_rate, epochs, batch_size))
        return cur.lastrowid

    def finish_run(self, run_id, final_loss=None, final_accuracy=None) -> None:
        """Mark a run finished with its final metrics."""
        self.execute(
            "UPDATE training_runs SET final_loss = ?, final_accuracy = ?, "
            "finished_at = datetime('now') WHERE id = ?;",
            (final_loss, final_accuracy, run_id))

    def log_epoch(self, run_id, epoch, loss=None, accuracy=None,
                  val_loss=None, val_accuracy=None) -> None:
        """Append one epoch's metrics."""
        self.execute(
            "INSERT OR REPLACE INTO epoch_history "
            "(run_id, epoch, loss, accuracy, val_loss, val_accuracy) "
            "VALUES (?, ?, ?, ?, ?, ?);",
            (run_id, epoch, loss, accuracy, val_loss, val_accuracy))

    def log_metric(self, name, value, run_id=None, step=0) -> None:
        """Log a generic metric."""
        self.execute(
            "INSERT INTO metrics (run_id, name, value, step) VALUES (?, ?, ?, ?);",
            (run_id, name, float(value), step))

    def best_run(self, order_by="final_accuracy"):
        """Return the best run joined with model name."""
        return self.query_one(
            f"SELECT r.*, m.name AS model_name FROM training_runs r "
            f"JOIN models m ON m.id = r.model_id "
            f"WHERE r.{order_by} IS NOT NULL "
            f"ORDER BY r.{order_by} DESC LIMIT 1;")

    # -- gate experiments --------------------------------------------------
    def add_gate_experiment(self, gate_name, model_id=None, hidden_units=0,
                            accuracy=0.0, tau=None) -> int:
        """Record a gate-learning experiment."""
        cur = self.execute(
            "INSERT INTO gate_experiments "
            "(gate_name, model_id, hidden_units, tau, accuracy) "
            "VALUES (?, ?, ?, ?, ?);",
            (gate_name, model_id, hidden_units, tau, accuracy))
        return cur.lastrowid

    def gate_accuracy_report(self) -> list:
        """Best accuracy achieved per gate across all experiments."""
        return self.query(
            "SELECT gate_name, MAX(accuracy) AS best_accuracy, "
            "COUNT(*) AS n_experiments, MIN(hidden_units) AS min_hidden "
            "FROM gate_experiments GROUP BY gate_name "
            "ORDER BY gate_name;")

    # -- analytical queries (SQL features showcase) ------------------------
    def run_summary(self) -> list:
        """Aggregate: runs, mean accuracy and mean loss per model."""
        return self.query(
            "SELECT m.name AS model, COUNT(r.id) AS n_runs, "
            "ROUND(AVG(r.final_accuracy), 4) AS avg_accuracy, "
            "ROUND(AVG(r.final_loss), 4) AS avg_loss "
            "FROM models m LEFT JOIN training_runs r ON r.model_id = m.id "
            "GROUP BY m.id ORDER BY avg_accuracy DESC;")

    def windows_ranked(self) -> list:
        """Window function: rank runs by accuracy within each model."""
        return self.query(
            "SELECT model_name, run_id, final_accuracy, "
            "RANK() OVER (PARTITION BY model_id ORDER BY final_accuracy DESC) AS rank_in_model "
            "FROM (SELECT r.id AS run_id, r.model_id, r.final_accuracy, "
            "m.name AS model_name FROM training_runs r "
            "JOIN models m ON m.id = r.model_id "
            "WHERE r.final_accuracy IS NOT NULL);")

    def cte_progress(self) -> list:
        """Common table expression: loss improvement per run (first vs last epoch)."""
        return self.query(
            "WITH firsts AS (SELECT run_id, MIN(epoch) AS e FROM epoch_history GROUP BY run_id), "
            "lasts AS (SELECT run_id, MAX(epoch) AS e FROM epoch_history GROUP BY run_id) "
            "SELECT e.run_id, "
            "(SELECT loss FROM epoch_history WHERE run_id = e.run_id "
            " AND epoch = (SELECT e FROM firsts WHERE run_id = e.run_id)) AS first_loss, "
            "(SELECT loss FROM epoch_history WHERE run_id = e.run_id "
            " AND epoch = (SELECT e FROM lasts WHERE run_id = e.run_id)) AS last_loss "
            "FROM epoch_history e GROUP BY e.run_id;")

    def integrity_report(self) -> str:
        """Run SQLite integrity + foreign-key checks."""
        integrity = self.scalar("PRAGMA integrity_check;")
        fk = self.query("PRAGMA foreign_key_check;")
        return f"integrity_check: {integrity}\nforeign_key_check: {fk or 'OK'}"

    # -- persistence helpers ----------------------------------------------
    def dump_json(self, path: str | None = None) -> dict:
        """Export every table to a JSON-serializable dict (and optionally a file)."""
        import json
        data = {t: self.query(f"SELECT * FROM {t};") for t in self.tables()}
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        return data
