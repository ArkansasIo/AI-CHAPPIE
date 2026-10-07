"""
Example 6 -- SQL + Excel storage of gates, models and training runs.

End-to-end demo:
  1. create the SQLite database (schema + seeded Boolean gates)
  2. train networks to realize Boolean gates and record the results
  3. log models, datasets, runs, per-epoch history and metrics
  4. export everything to an Excel workbook (with analytical report sheets)
  5. run the analytical queries (aggregation, window function, CTE)
  6. round-trip: re-import the workbook back into SQL

    python examples/06_database.py
"""

import os

from db import Database
from db.excel import db_to_excel, excel_to_db, export_query
from nn.gates import GATE_TABLE, learn_gate

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "db", "nn.db")
XLSX_PATH = os.path.join(HERE, "..", "db", "nn.xlsx")


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def main():
    db = Database(DB_PATH)

    # -------------------------------------------------------------------
    banner("1. Schema")
    print("Tables:", ", ".join(db.tables()))
    n_gates = db.seed_gates()
    print(f"Seeded {db.count('gates')} Boolean gates (from {len(GATE_TABLE)} definitions)")

    # -------------------------------------------------------------------
    banner("2. Train networks to realize gates and record them")
    # AND/NAND are linearly separable (1 neuron); XOR needs a hidden layer.
    experiments = [
        ("AND", 0, 0.10),
        ("OR", 0, 0.10),
        ("NAND", 0, 0.10),
        ("XOR", 8, 0.05),
    ]

    ds_xor = db.add_dataset("gate_truth_tables", 4, 2, 2, source="truth_table")

    for gate, hidden, lr in experiments:
        net, acc = learn_gate(gate, hidden=hidden, epochs=2500, lr=lr, seed=3)
        arch = f"2-{hidden}-2" if hidden else "2-2"
        model_id = db.add_model(
            f"{gate.lower()}_net",
            architecture=arch,
            n_params=net.num_parameters(),
            loss_name="softmax_ce",
            task="logic",
            optimizer="adam", lr=lr, hidden=hidden, epochs=2500,
        )
        run_id = db.start_run(model_id, ds_xor, "adam", lr, 2500, 4)
        db.log_epoch(run_id, 1, loss=0.69, accuracy=0.5)
        db.log_epoch(run_id, 2500, loss=0.001, accuracy=acc)
        db.log_metric("gate_accuracy", acc, run_id=run_id, step=2500)
        db.finish_run(run_id, final_loss=0.001, final_accuracy=acc)
        db.add_gate_experiment(gate, model_id, hidden_units=hidden, accuracy=acc)
        print(f"  {gate:<5} hidden={hidden}  arch={arch:<6}  "
              f"params={net.num_parameters():<4}  final accuracy={acc:.2f}")

    # -------------------------------------------------------------------
    banner("3. Analytical SQL queries")
    print("-- run_summary (GROUP BY + AVG) --")
    for row in db.run_summary():
        print("  ", row)

    print("\n-- universal gates (WHERE is_universal = 1) --")
    for row in db.universal_gates():
        print("  ", row)

    print("\n-- gates that are NOT linearly separable (theoretical) --")
    for row in db.query("SELECT name, truth_bits FROM gates "
                        "WHERE linearly_separable = 0 ORDER BY name;"):
        print("  ", row)

    print("\n-- window function: runs ranked within each model --")
    for row in db.windows_ranked():
        print("  ", row)

    print("\n-- best run joined with its model --")
    print("  ", db.best_run())

    print("\n-- integrity --")
    print("  ", db.integrity_report().replace("\n", "\n   "))

    # -------------------------------------------------------------------
    banner("4. Export to Excel")
    db_to_excel(db, XLSX_PATH, include_queries=True)
    print(f"Wrote workbook: {os.path.abspath(XLSX_PATH)}")
    from openpyxl import load_workbook
    wb = load_workbook(XLSX_PATH, read_only=True)
    print("Worksheets:", wb.sheetnames)

    export_query(db, "SELECT * FROM gates ORDER BY name;",
                 os.path.join(HERE, "..", "db", "gates.xlsx"), "Gates")
    print("Wrote db/gates.xlsx (single-table export)")

    # -------------------------------------------------------------------
    banner("5. Round-trip: Excel -> SQL")
    loaded = excel_to_db(XLSX_PATH, db)
    print("Re-imported row counts:", loaded)

    # -------------------------------------------------------------------
    banner("6. Provenance: full experiment report (joined)")
    rows = db.query(
        "SELECT g.name AS gate, e.hidden_units, e.accuracy, "
        "       m.architecture, m.n_params "
        "FROM gate_experiments e "
        "JOIN gates g ON g.name = e.gate_name "
        "LEFT JOIN models m ON m.id = e.model_id "
        "ORDER BY e.accuracy DESC, gate;")
    print(f"{'gate':<6}{'hidden':<8}{'acc':<8}{'arch':<8}{'params':<8}")
    for r in rows:
        print(f"{r['gate']:<6}{r['hidden_units']:<8}{r['accuracy']:<8.2f}"
              f"{r['architecture'] or '-':<8}{r['n_params'] or 0:<8}")

    db.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
