"""
db.dictionary -- a SQL (SQLite) English-language dictionary database.

A full lexical database: words, their senses, definitions, examples, parts of
speech, pronunciation, etymology, synonyms/antonyms and inter-word relations.

Schema
------
    words            one row per headword (spelling, lemma, length, first letter,
                     part-of-speech summary, frequency rank, syllable count)
    senses           one row per word meaning (part of speech, definition,
                     register, domain) -- a word can have many
    definitions      the definition text of a sense, split out so a sense can
                     carry variants (kept 1:1 here for clarity)
    examples         usage sentences attached to a sense
    synonyms         word -> word synonym edges (self-join on words)
    antonyms         word -> word antonym edges
    relations        generic lexical relations (hypernym, hyponym, derived,
                     related, see_also)
    etymology        origin / language-of-origin per word
    pronunciations   IPA and simple respelling variants
    word_frequency   corpus frequency counts (per million words)
    anagrams         precomputed sorted-letter keys for anagram lookup

Views
-----
    v_word_summary   headword + first sense + sense count + synonym count
    v_parts_of_speech  distribution of parts of speech

Typical use
-----------
    from db.dictionary import Dictionary

    with Dictionary("db/english.db") as d:
        d.define("bank")                 # all senses
        d.lookup_prefix("quan")          # prefix search
        d.anagrams("listen")             # -> silent, enlist, tinsel
        d.wildcard("c_t")                # pattern search
        d.stats()                        # database statistics

Pure standard library (`sqlite3`).
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

__all__ = ["Dictionary", "SCHEMA_SQL", "VIEWS_SQL", "DEFAULT_DICT_PATH"]

DEFAULT_DICT_PATH = os.path.join(os.path.dirname(__file__), "english.db")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 1. Headwords -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS words (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    lemma       TEXT,                       -- base form (e.g. 'run' for 'ran')
    length      INTEGER NOT NULL CHECK (length >= 1),
    first_letter TEXT   NOT NULL,
    syllables   INTEGER NOT NULL DEFAULT 1 CHECK (syllables >= 1),
    pos_summary TEXT,                       -- e.g. 'noun, verb'
    frequency_rank INTEGER,                 -- 1 = most common; NULL if unknown
    is_common   INTEGER NOT NULL DEFAULT 0 CHECK (is_common IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_words_word    ON words(word);
CREATE INDEX IF NOT EXISTS idx_words_length  ON words(length);
CREATE INDEX IF NOT EXISTS idx_words_first   ON words(first_letter);
CREATE INDEX IF NOT EXISTS idx_words_rank    ON words(frequency_rank);

-- 2. Senses (one row per meaning) -----------------------------------------
CREATE TABLE IF NOT EXISTS senses (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    sense_no    INTEGER NOT NULL DEFAULT 1 CHECK (sense_no >= 1),
    part_of_speech TEXT NOT NULL
        CHECK (part_of_speech IN
              ('noun','verb','adjective','adverb','pronoun','preposition',
               'conjunction','interjection','determiner','numeral')),
    definition  TEXT    NOT NULL,
    register    TEXT    DEFAULT 'standard',   -- standard, informal, formal, archaic, slang
    domain      TEXT,                          -- e.g. 'computing', 'biology'
    UNIQUE (word_id, sense_no)
);

CREATE INDEX IF NOT EXISTS idx_senses_word ON senses(word_id);
CREATE INDEX IF NOT EXISTS idx_senses_pos  ON senses(part_of_speech);

-- 3. Usage examples --------------------------------------------------------
CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sense_id    INTEGER NOT NULL REFERENCES senses(id) ON DELETE CASCADE,
    example     TEXT    NOT NULL,
    source      TEXT    DEFAULT 'authored'
);

CREATE INDEX IF NOT EXISTS idx_examples_sense ON examples(sense_id);

-- 4. Synonym edges ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS synonyms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    synonym     TEXT    NOT NULL,              -- may or may not be a headword
    sense_no    INTEGER DEFAULT 1,
    strength    REAL    DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    UNIQUE (word_id, synonym)
);

CREATE INDEX IF NOT EXISTS idx_syn_word ON synonyms(word_id);
CREATE INDEX IF NOT EXISTS idx_syn_text ON synonyms(synonym);

-- 5. Antonym edges ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS antonyms (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    antonym     TEXT    NOT NULL,
    sense_no    INTEGER DEFAULT 1,
    UNIQUE (word_id, antonym)
);

CREATE INDEX IF NOT EXISTS idx_ant_word ON antonyms(word_id);
CREATE INDEX IF NOT EXISTS idx_ant_text ON antonyms(antonym);

-- 6. Other lexical relations ----------------------------------------------
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    relation    TEXT    NOT NULL
        CHECK (relation IN ('hypernym','hyponym','derived','related',
                            'see_also','meronym','holonym','participle',
                            'plural','comparative','superlative')),
    target      TEXT    NOT NULL,
    note        TEXT,
    UNIQUE (word_id, relation, target)
);

CREATE INDEX IF NOT EXISTS idx_rel_word ON relations(word_id);
CREATE INDEX IF NOT EXISTS idx_rel_kind ON relations(relation);

-- 7. Etymology -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS etymology (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL UNIQUE REFERENCES words(id) ON DELETE CASCADE,
    origin_language TEXT,             -- e.g. 'Latin', 'Old English', 'Greek'
    root        TEXT,                 -- e.g. 'banca'
    note        TEXT
);

-- 8. Pronunciations --------------------------------------------------------
CREATE TABLE IF NOT EXISTS pronunciations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    ipa         TEXT,                 -- e.g. '/bæŋk/'
    respelling  TEXT,                 -- e.g. 'bangk'
    accent      TEXT DEFAULT 'US',    -- US, UK, AU
    UNIQUE (word_id, accent)
);

-- 9. Corpus frequency ------------------------------------------------------
CREATE TABLE IF NOT EXISTS word_frequency (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    count       INTEGER NOT NULL DEFAULT 0 CHECK (count >= 0),
    per_million REAL    NOT NULL DEFAULT 0,
    corpus      TEXT    DEFAULT 'generic'
);

-- 10. Anagram keys ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS anagrams (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word_id     INTEGER NOT NULL REFERENCES words(id) ON DELETE CASCADE,
    letter_key  TEXT    NOT NULL,             -- letters sorted alphabetically
    UNIQUE (word_id)
);

CREATE INDEX IF NOT EXISTS idx_anagram_key ON anagrams(letter_key);
"""


VIEWS_SQL = """
-- Headword overview: first sense, sense count, synonym/anagram counts.
CREATE VIEW IF NOT EXISTS v_word_summary AS
SELECT
    w.id,
    w.word,
    w.length,
    w.pos_summary,
    w.frequency_rank,
    (SELECT s.part_of_speech FROM senses s WHERE s.word_id = w.id
     ORDER BY s.sense_no LIMIT 1) AS first_pos,
    (SELECT s.definition FROM senses s WHERE s.word_id = w.id
     ORDER BY s.sense_no LIMIT 1) AS first_definition,
    (SELECT COUNT(*) FROM senses s WHERE s.word_id = w.id) AS n_senses,
    (SELECT COUNT(*) FROM synonyms y WHERE y.word_id = w.id) AS n_synonyms,
    (SELECT COUNT(*) FROM examples e
     JOIN senses s2 ON s2.id = e.sense_id
     WHERE s2.word_id = w.id) AS n_examples
FROM words w;

-- How many words have each part of speech (counting senses).
CREATE VIEW IF NOT EXISTS v_parts_of_speech AS
SELECT part_of_speech, COUNT(*) AS n_senses,
       COUNT(DISTINCT word_id) AS n_words
FROM senses GROUP BY part_of_speech ORDER BY n_senses DESC;

-- Word-length distribution.
CREATE VIEW IF NOT EXISTS v_length_distribution AS
SELECT length, COUNT(*) AS n_words FROM words GROUP BY length ORDER BY length;
"""


class Dictionary:
    """A SQLite-backed English dictionary with rich lexical queries."""

    def __init__(self, path: str = DEFAULT_DICT_PATH, echo: bool = False):
        self.path = path
        self.echo = echo
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------
    def init_schema(self) -> None:
        """Create tables and views if they do not exist."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.executescript(VIEWS_SQL)
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
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- low level ---------------------------------------------------------
    def execute(self, sql: str, params=()) -> sqlite3.Cursor:
        if self.echo:
            print(f"[SQL] {sql.strip()}  {params}")
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def executemany(self, sql: str, rows) -> sqlite3.Cursor:
        rows = list(rows)
        if self.echo:
            print(f"[SQL] {sql.strip()}  x{len(rows)}")
        cur = self.conn.executemany(sql, rows)
        self.conn.commit()
        return cur

    def query(self, sql: str, params=()) -> list:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def query_one(self, sql: str, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params=()):
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return None if row is None else row[0]

    def sql(self, sql: str, params=()):
        head = sql.lstrip().split(None, 1)[0].upper() if sql.strip() else ""
        if head in ("SELECT", "WITH", "PRAGMA", "EXPLAIN"):
            return self.query(sql, params)
        return self.execute(sql, params)

    def tables(self) -> list:
        return [r["name"] for r in self.query(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name;")]

    def views(self) -> list:
        return [r["name"] for r in self.query(
            "SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;")]

    def count(self, table: str) -> int:
        return int(self.scalar(f"SELECT COUNT(*) FROM {table};"))

    # =====================================================================
    # Writing
    # =====================================================================
    @staticmethod
    def _letter_key(word: str) -> str:
        """Sorted lowercase letters, used for anagram matching."""
        return "".join(sorted(word.lower().replace(" ", "")))

    def add_word(self, word: str, *, syllables: int = 1,
                 frequency_rank: int | None = None, lemma: str | None = None,
                 is_common: bool | None = None) -> int:
        """Insert a headword (idempotent on spelling) and return its id."""
        w = word.strip()
        if not w:
            raise ValueError("empty word")
        common = int(is_common) if is_common is not None else \
            int(frequency_rank is not None and frequency_rank <= 3000)
        self.execute(
            "INSERT INTO words (word, lemma, length, first_letter, syllables, "
            "frequency_rank, is_common) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(word) DO UPDATE SET "
            "syllables=excluded.syllables, frequency_rank=excluded.frequency_rank, "
            "is_common=excluded.is_common;",
            (w, lemma, len(w), w[0].upper(), syllables, frequency_rank, common))
        wid = int(self.scalar("SELECT id FROM words WHERE word = ?;", (w,)))
        self.execute(
            "INSERT INTO anagrams (word_id, letter_key) VALUES (?, ?) "
            "ON CONFLICT(word_id) DO UPDATE SET letter_key=excluded.letter_key;",
            (wid, self._letter_key(w)))
        return wid

    def add_sense(self, word: str, part_of_speech: str, definition: str,
                  sense_no: int | None = None, register: str = "standard",
                  domain: str | None = None) -> int:
        """Add a meaning (sense) for a word; returns the sense id."""
        wid = self.word_id(word)
        if wid is None:
            wid = self.add_word(word)
        if sense_no is None:
            sense_no = int(self.scalar(
                "SELECT COALESCE(MAX(sense_no), 0) + 1 FROM senses WHERE word_id = ?;",
                (wid,)))
        self.execute(
            "INSERT INTO senses (word_id, sense_no, part_of_speech, definition, "
            "register, domain) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(word_id, sense_no) DO UPDATE SET "
            "part_of_speech=excluded.part_of_speech, definition=excluded.definition, "
            "register=excluded.register, domain=excluded.domain;",
            (wid, sense_no, part_of_speech, definition, register, domain))
        return int(self.scalar(
            "SELECT id FROM senses WHERE word_id = ? AND sense_no = ?;",
            (wid, sense_no)))

    def add_example(self, word: str, sentence: str, sense_no: int = 1,
                    source: str = "authored") -> int:
        """Attach a usage example to a sense."""
        sid = self._sense_id(word, sense_no)
        if sid is None:
            raise KeyError(f"no sense {sense_no} for '{word}'")
        cur = self.execute(
            "INSERT INTO examples (sense_id, example, source) VALUES (?, ?, ?);",
            (sid, sentence, source))
        return cur.lastrowid

    def add_synonym(self, word: str, synonym: str, strength: float = 1.0,
                    sense_no: int = 1) -> None:
        wid = self._word_id_required(word)
        self.execute(
            "INSERT INTO synonyms (word_id, synonym, sense_no, strength) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(word_id, synonym) DO UPDATE SET "
            "strength=excluded.strength;",
            (wid, synonym.strip(), sense_no, strength))

    def add_antonym(self, word: str, antonym: str, sense_no: int = 1) -> None:
        wid = self._word_id_required(word)
        self.execute(
            "INSERT INTO antonyms (word_id, antonym, sense_no) VALUES (?, ?, ?) "
            "ON CONFLICT(word_id, antonym) DO UPDATE SET sense_no=excluded.sense_no;",
            (wid, antonym.strip(), sense_no))

    def add_relation(self, word: str, relation: str, target: str,
                     note: str | None = None) -> None:
        wid = self._word_id_required(word)
        self.execute(
            "INSERT INTO relations (word_id, relation, target, note) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(word_id, relation, target) "
            "DO UPDATE SET note=excluded.note;",
            (wid, relation, target, note))

    def add_etymology(self, word: str, origin_language: str | None = None,
                      root: str | None = None, note: str | None = None) -> None:
        wid = self._word_id_required(word)
        self.execute(
            "INSERT INTO etymology (word_id, origin_language, root, note) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(word_id) DO UPDATE SET "
            "origin_language=excluded.origin_language, root=excluded.root, "
            "note=excluded.note;",
            (wid, origin_language, root, note))

    def add_pronunciation(self, word: str, ipa: str | None = None,
                          respelling: str | None = None, accent: str = "US") -> None:
        wid = self._word_id_required(word)
        self.execute(
            "INSERT INTO pronunciations (word_id, ipa, respelling, accent) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(word_id, accent) DO UPDATE SET "
            "ipa=excluded.ipa, respelling=excluded.respelling;",
            (wid, ipa, respelling, accent))

    def add_frequency(self, word: str, count: int, per_million: float,
                      corpus: str = "generic") -> None:
        self.execute(
            "INSERT INTO word_frequency (word, count, per_million, corpus) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(word) DO UPDATE SET "
            "count=excluded.count, per_million=excluded.per_million;",
            (word.strip(), int(count), float(per_million), corpus))

    def add_entry(self, word: str, pos: str, definition: str, *,
                  example: str | None = None, synonyms=(),
                  antonyms=(), origin_language: str | None = None,
                  root: str | None = None, ipa: str | None = None,
                  respelling: str | None = None, syllables: int = 1,
                  frequency_rank: int | None = None) -> int:
        """Add a complete one-sense entry in a single call (convenience)."""
        wid = self.add_word(word, syllables=syllables,
                            frequency_rank=frequency_rank)
        self.add_sense(word, pos, definition)
        if example:
            self.add_example(word, example)
        for s in synonyms:
            self.add_synonym(word, s)
        for a in antonyms:
            self.add_antonym(word, a)
        if origin_language or root:
            self.add_etymology(word, origin_language, root)
        if ipa or respelling:
            self.add_pronunciation(word, ipa, respelling)
        return wid

    # -- id helpers --------------------------------------------------------
    def word_id(self, word: str):
        return self.scalar("SELECT id FROM words WHERE word = ? COLLATE NOCASE;",
                           (word.strip(),))

    def _word_id_required(self, word: str) -> int:
        wid = self.word_id(word)
        if wid is None:
            raise KeyError(f"word '{word}' is not in the dictionary")
        return wid

    def _sense_id(self, word: str, sense_no: int = 1):
        wid = self.word_id(word)
        if wid is None:
            return None
        return self.scalar(
            "SELECT id FROM senses WHERE word_id = ? AND sense_no = ?;",
            (wid, sense_no))

    # =====================================================================
    # Reading / querying
    # =====================================================================
    def define(self, word: str) -> dict:
        """Full entry for a word: senses, examples, synonyms, antonyms, etc."""
        row = self.query_one(
            "SELECT * FROM words WHERE word = ? COLLATE NOCASE;", (word.strip(),))
        if row is None:
            return {}
        wid = row["id"]
        senses = self.query(
            "SELECT sense_no, part_of_speech, definition, register, domain "
            "FROM senses WHERE word_id = ? ORDER BY sense_no;", (wid,))
        for s in senses:
            s["examples"] = [e["example"] for e in self.query(
                "SELECT example FROM examples e JOIN senses s2 ON s2.id = e.sense_id "
                "WHERE s2.word_id = ? AND s2.sense_no = ?;", (wid, s["sense_no"]))]
        row["senses"] = senses
        row["synonyms"] = [r["synonym"] for r in self.query(
            "SELECT synonym FROM synonyms WHERE word_id = ? ORDER BY strength DESC;",
            (wid,))]
        row["antonyms"] = [r["antonym"] for r in self.query(
            "SELECT antonym FROM antonyms WHERE word_id = ?;", (wid,))]
        row["relations"] = self.query(
            "SELECT relation, target, note FROM relations WHERE word_id = ? "
            "ORDER BY relation, target;", (wid,))
        row["etymology"] = self.query_one(
            "SELECT origin_language, root, note FROM etymology WHERE word_id = ?;",
            (wid,))
        row["pronunciations"] = self.query(
            "SELECT ipa, respelling, accent FROM pronunciations WHERE word_id = ?;",
            (wid,))
        row["anagram_key"] = self._letter_key(row["word"])
        return row

    def lookup(self, pattern: str, limit: int = 20) -> list:
        """Case-insensitive exact-ish lookup using LIKE (supports `%` and `_`)."""
        return self.query(
            "SELECT w.word, w.length, (SELECT definition FROM senses s "
            " WHERE s.word_id = w.id ORDER BY sense_no LIMIT 1) AS definition "
            "FROM words w WHERE w.word LIKE ? COLLATE NOCASE "
            "ORDER BY w.frequency_rank IS NULL, w.frequency_rank, w.word LIMIT ?;",
            (pattern, limit))

    def prefix(self, prefix: str, limit: int = 25) -> list:
        """All words starting with `prefix`."""
        return self.query(
            "SELECT word, length FROM words WHERE word LIKE ? COLLATE NOCASE "
            "ORDER BY length, word LIMIT ?;", (prefix + "%", limit))

    def suffix(self, suffix: str, limit: int = 25) -> list:
        """All words ending with `suffix`."""
        return self.query(
            "SELECT word, length FROM words WHERE word LIKE ? COLLATE NOCASE "
            "ORDER BY length, word LIMIT ?;", ("%" + suffix, limit))

    def wildcard(self, pattern: str, limit: int = 25) -> list:
        """Pattern search: `_` = any one letter, `%` = any run of letters.

        Example: `wildcard('c_t')` matches cat, cot, cut.
        """
        return self.query(
            "SELECT word, length FROM words WHERE word LIKE ? COLLATE NOCASE "
            "ORDER BY length, word LIMIT ?;", (pattern, limit))

    def anagram_letters(self, letters: str, limit: int = 25) -> list:
        """Words that are exact anagrams of `letters`."""
        key = self._letter_key(letters)
        return self.query(
            "SELECT w.word FROM words w JOIN anagrams a ON a.word_id = w.id "
            "WHERE a.letter_key = ? AND w.word <> ? COLLATE NOCASE "
            "ORDER BY w.word LIMIT ?;", (key, letters.strip(), limit))

    def anagrams(self, word: str, limit: int = 25) -> list:
        """Exact anagrams of `word` (using the precomputed letter key)."""
        return self.anagram_letters(word, limit)

    def sub_anagrams(self, letters: str, min_len: int = 3, limit: int = 25) -> list:
        """Words using a subset of the supplied letters (Scrabble-style)."""
        from collections import Counter
        pool = Counter(letters.lower())
        out = []
        for row in self.query(
                "SELECT word FROM words WHERE length >= ? AND length <= ? "
                "ORDER BY length DESC, word LIMIT 5000;",
                (min_len, len(letters))):
            w = row["word"].lower()
            need = Counter(w)
            if all(need[c] <= pool.get(c, 0) for c in need):
                out.append(row["word"])
                if len(out) >= limit:
                    break
        return out

    def rhymes(self, word: str, min_match: int = 2, limit: int = 25) -> list:
        """Approximate rhymes: same final `min_match` letters."""
        w = word.strip().lower()
        tail = w[-min_match:]
        return self.query(
            "SELECT word FROM words WHERE word LIKE ? COLLATE NOCASE "
            "AND word <> ? COLLATE NOCASE ORDER BY length, word LIMIT ?;",
            ("%" + tail, word.strip(), limit))

    def by_length(self, length: int, limit: int = 50) -> list:
        """All words of exactly `length` letters."""
        return self.query(
            "SELECT word FROM words WHERE length = ? ORDER BY word LIMIT ?;",
            (length, limit))

    def by_pos(self, part_of_speech: str, limit: int = 25) -> list:
        """Words having a sense with the given part of speech."""
        return self.query(
            "SELECT DISTINCT w.word, s.definition FROM words w "
            "JOIN senses s ON s.word_id = w.id WHERE s.part_of_speech = ? "
            "ORDER BY w.word LIMIT ?;", (part_of_speech.lower(), limit))

    def synonyms_of(self, word: str) -> list:
        return [r["synonym"] for r in self.query(
            "SELECT synonym FROM synonyms WHERE word_id = ? ORDER BY strength DESC;",
            (self._word_id_required(word),))]

    def antonyms_of(self, word: str) -> list:
        return [r["antonym"] for r in self.query(
            "SELECT antonym FROM antonyms WHERE word_id = ?;",
            (self._word_id_required(word),))]

    def relations_of(self, word: str, relation: str | None = None) -> list:
        wid = self._word_id_required(word)
        if relation:
            return self.query(
                "SELECT relation, target FROM relations WHERE word_id = ? "
                "AND relation = ? ORDER BY target;", (wid, relation))
        return self.query(
            "SELECT relation, target FROM relations WHERE word_id = ? "
            "ORDER BY relation, target;", (wid,))

    def search_definitions(self, text: str, limit: int = 25) -> list:
        """Full-text style search over definitions (LIKE based)."""
        return self.query(
            "SELECT w.word, s.part_of_speech, s.definition FROM senses s "
            "JOIN words w ON w.id = s.word_id WHERE s.definition LIKE ? "
            "ORDER BY w.word LIMIT ?;", (f"%{text}%", limit))

    def random_word(self, min_length: int = 1) -> dict:
        """A random headword (with its first definition)."""
        return self.query_one(
            "SELECT w.word, (SELECT definition FROM senses s WHERE s.word_id = w.id "
            "ORDER BY sense_no LIMIT 1) AS definition FROM words w "
            "WHERE w.length >= ? ORDER BY RANDOM() LIMIT 1;", (min_length,))

    def word_of_the_day(self, day_index: int | None = None) -> dict:
        """Deterministic daily pick by frequency rank of common words."""
        import datetime
        if day_index is None:
            day_index = datetime.date.today().toordinal()
        n = self.count("words")
        if n == 0:
            return {}
        offset = day_index % n
        return self.query_one(
            "SELECT w.word, (SELECT definition FROM senses s WHERE s.word_id = w.id "
            "ORDER BY sense_no LIMIT 1) AS definition FROM words w "
            "ORDER BY w.word LIMIT 1 OFFSET ?;", (offset,))

    # -- statistics --------------------------------------------------------
    def stats(self) -> dict:
        """Summary statistics about the dictionary."""
        return {
            "words": self.count("words"),
            "senses": self.count("senses"),
            "examples": self.count("examples"),
            "synonyms": self.count("synonyms"),
            "antonyms": self.count("antonyms"),
            "relations": self.count("relations"),
            "etymologies": self.count("etymology"),
            "avg_senses_per_word": round(
                (self.count("senses") / self.count("words"))
                if self.count("words") else 0.0, 3),
            "by_pos": self.query("SELECT * FROM v_parts_of_speech;"),
            "by_length": self.query("SELECT * FROM v_length_distribution;"),
        }

    def report(self) -> str:
        """Human-readable statistics report."""
        s = self.stats()
        lines = [f"English dictionary @ {self.path}",
                 f"  headwords : {s['words']}",
                 f"  senses    : {s['senses']}  (avg {s['avg_senses_per_word']}/word)",
                 f"  examples  : {s['examples']}",
                 f"  synonyms  : {s['synonyms']}   antonyms: {s['antonyms']}",
                 f"  relations : {s['relations']}   etymologies: {s['etymologies']}"]
        lines.append("  parts of speech:")
        for r in s["by_pos"]:
            lines.append(f"    {r['part_of_speech']:<13} {r['n_senses']:>4} senses, "
                         f"{r['n_words']:>4} words")
        return "\n".join(lines)

    def integrity_report(self) -> str:
        integrity = self.scalar("PRAGMA integrity_check;")
        fk = self.query("PRAGMA foreign_key_check;")
        return f"integrity_check: {integrity}\nforeign_key_check: {fk or 'OK'}"

    def dump_json(self, path: str | None = None) -> dict:
        import json
        data = {t: self.query(f"SELECT * FROM {t};") for t in self.tables()}
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return data
