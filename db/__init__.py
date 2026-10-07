"""
db -- SQL (SQLite) + Excel (.xlsx) storage layer for the neural-network project.

    from db import Database
    from db.excel import db_to_excel, export_query

    with Database("db/nn.db") as db:
        db.seed_gates()
        db.add_model("xor_net", "2-16-16-2", n_params=105, loss_name="softmax_ce")
        db_to_excel(db, "db/nn.xlsx", include_queries=True)

An English-language dictionary database is also available:

    from db.dictionary import Dictionary
    from db import lexicon

    d = lexicon.build()          # creates + seeds db/english.db
    d.define("bank")
"""

from . import database
from . import excel
from . import dictionary
from . import lexicon
from .database import Database, SCHEMA_SQL, DEFAULT_DB_PATH
from .dictionary import Dictionary, DEFAULT_DICT_PATH
from .excel import (
    write_workbook,
    read_workbook,
    export_query,
    import_sheet,
    db_to_excel,
    excel_to_db,
    sync,
)

__all__ = [
    "database",
    "excel",
    "dictionary",
    "lexicon",
    "Database",
    "SCHEMA_SQL",
    "DEFAULT_DB_PATH",
    "Dictionary",
    "DEFAULT_DICT_PATH",
    "write_workbook",
    "read_workbook",
    "export_query",
    "import_sheet",
    "db_to_excel",
    "excel_to_db",
    "sync",
]
