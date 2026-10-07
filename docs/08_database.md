# Chapter 8 — The SQL + Excel Database Layer

This chapter documents the storage layer that records everything the project
computes: gates, models, training runs, per-epoch curves, metrics, and the
logic-gate learning experiments.

Two technologies are combined:

* **SQL (SQLite)** — the relational source of truth.
* **Excel (.xlsx)** — the human-facing reporting / interchange format.

They are kept in sync both ways (`db_db_to_excel` / `excel_to_db`).

---

## 1. Why a relational database

Machine learning produces **structured, related** records:

```
model  ──< training_run ──< epoch_history
   │             │
   │             └──< metrics
   └──< hyperparameters

gate ──< gate_experiment >── model
dataset ──< training_run
```

A relational schema with **foreign keys** guarantees that:

* a training run cannot reference a model that does not exist,
* deleting a model cascades to its runs/hyperparameters,
* accuracy is constrained to `[0, 1]` by a `CHECK`,
* gate names are `UNIQUE`.

Relationships are the reason a spreadsheet alone is not enough.

---

## 2. The schema (DDL)

| Table | Purpose | Key relations |
|-------|---------|---------------|
| `gates` | the 16 two-input Boolean functions + truth-table bits | — |
| `datasets` | dataset metadata (samples, features, classes) | — |
| `models` | saved architectures + parameter counts | — |
| `training_runs` | one row per `fit()` call | → `models`, `datasets` |
| `hyperparameters` | key/value config per model | → `models` (unique per key) |
| `gate_experiments` | gate-learning accuracy results | → `gates`, `models` |
| `metrics` | generic `(name, value, step)` log | → `training_runs` |
| `epoch_history` | loss/accuracy per epoch (training curves) | → `training_runs` |

Integrity constraints used:

```sql
PRIMARY KEY (id)                        -- identity
UNIQUE (name) / UNIQUE (run_id, epoch)  -- no duplicates
FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE
CHECK (accuracy BETWEEN 0 AND 1)
CHECK (1 << n_inputs = length(truth_bits))
```

---

## 3. Representing truth tables in SQL

A Boolean function of `n` inputs has `2^n` outputs. We store them as a **bit
string** `truth_bits`, ordered by row index:

```
row index i  ->  inputs (i >> (n-1), ..., i & 1)  ->  truth_bits[i]
```

For two inputs, `AND = '0001'`, `XOR = '0110'`, `NAND = '1110'`. This is compact
and lets SQL do set operations, e.g. "all gates that are 1 on row 3":

```sql
SELECT name FROM gates WHERE substr(truth_bits, 4, 1) = '1';
```

The `is_universal` and `linearly_separable` flags encode the theory from
Chapter 7 directly as columns — so you can query the mathematics.

---

## 4. Query patterns used in `Database`

* **CRUD**: `add_gate`, `add_model`, `start_run`, `finish_run`, `log_epoch`.
* **Upsert**: `INSERT ... ON CONFLICT(key) DO UPDATE` for hyperparameters.
* **Aggregation** (`run_summary`): `GROUP BY model` with `AVG`, `COUNT`, `ROUND`.
* **Window functions** (`windows_ranked`):
  `RANK() OVER (PARTITION BY model_id ORDER BY accuracy DESC)`.
* **CTE** (`cte_progress`): `WITH firsts AS (...) , lasts AS (...) SELECT ...`
  to compute per-run loss improvement.
* **Integrity**: `PRAGMA integrity_check` and `PRAGMA foreign_key_check`.
* **Transactions**: a `transaction()` context manager rolls back on error.

---

## 5. Excel as a reporting surface

`db.excel` maps each SQL table to a worksheet:

```
workbook.xlsx
├── gates              (all Boolean functions)
├── models             (architectures)
├── training_runs      (run metadata + final metrics)
├── epoch_history      (training curves)
├── hyperparameters
├── gate_experiments
├── metrics
├── _run_summary       (report: avg accuracy/loss per model)
├── _gate_accuracy     (report: best accuracy per gate)
└── _window_ranking    (report: ranked runs)
```

Formatting applied by `write_workbook`: bold white-on-blue headers, frozen
header row (`freeze_panes="A2"`), auto-fit column widths, and an auto-filter on
the data range — so the file is immediately usable for analysis and charts.

### Round-tripping

```python
from db import Database
from db.excel import db_to_excel, excel_to_db, export_query

with Database("db/nn.db") as db:
    db.seed_gates()
    db_to_excel(db, "db/nn.xlsx", include_queries=True)     # SQL  -> Excel
    excel_to_db("db/nn.xlsx", db)                          # Excel -> SQL
    export_query(db, "SELECT * FROM gates WHERE is_universal=1;",
                 "db/universal_gates.xlsx", "Universal")
```

Type coercion rules: `None` → empty cell; numbers/strings stored natively;
anything else stringified. On import, only columns present in the target SQL
table are used, and rows are written with `INSERT OR REPLACE`.

---

## 6. Full worked flow

```python
from db import Database
from db.excel import db_to_excel
from nn.gates import learn_gate

with Database("db/nn.db") as db:
    db.seed_gates()                                   # 16 Boolean functions

    # --- train a network to realize XOR (needs a hidden layer) ---
    net, acc = learn_gate("XOR", hidden=8, epochs=2000)

    model_id = db.add_model(
        "xor_mlp", architecture="2-8-2", n_params=net.num_parameters(),
        loss_name="softmax_ce", task="logic",
        optimizer="adam", lr=0.05, hidden=8)          # hyperparameters

    ds = db.add_dataset("xor", 4, 2, 2, source="truth_table")
    run = db.start_run(model_id, ds, "adam", 0.05, 2000, 4)
    db.finish_run(run, final_loss=0.001, final_accuracy=acc)

    db.add_gate_experiment("XOR", model_id, hidden_units=8, accuracy=acc)

    db_to_excel(db, "db/nn.xlsx", include_queries=True)
```

---

## 7. API summary

| Layer | Function | Direction |
|-------|----------|-----------|
| SQL | `Database.sql(...)`, `.query(...)`, `.execute(...)` | generic |
| SQL | `.add_model`, `.start_run`, `.log_epoch`, `.add_gate_experiment` | domain |
| SQL | `.run_summary`, `.windows_ranked`, `.cte_progress` | analytics |
| Excel | `db_to_excel(db, path, include_queries=True)` | SQL → Excel |
| Excel | `excel_to_db(path, db)` / `import_sheet(...)` | Excel → SQL |
| Excel | `export_query(db, sql, path)` | query → file |
| Excel | `sync(db, path, direction)` | both |
