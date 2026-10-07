# Chapter 11 — The English Dictionary Database

This chapter documents the lexical database built on SQL (SQLite) with
`db/dictionary.py` and seeded by `db/lexicon.py`. It follows the same design
principles as Chapter 8 (the ML metadata database) but models **language**
instead of models.

---

## 1. Why a relational model fits a dictionary

A dictionary is naturally relational. A headword is not one thing — it is:

```
word ──< sense ──< example
  │        │
  │        └── part of speech, register, domain
  ├──< synonym  (word → word edge)
  ├──< antonym  (word → word edge)
  ├──< relation (hypernym, hyponym, derived, …)
  ├──   etymology      (1:1)
  ├──< pronunciation   (per accent)
  ├──   frequency      (corpus stats)
  └──   anagram key    (precomputed, for O(1) anagram lookup)
```

A flat spreadsheet cannot express "this sense has these three examples but that
sense has none." A relational schema can, and SQLite enforces it.

---

## 2. Schema (DDL) — `db/dictionary.py`

| Table | Rows are | Key notes |
|-------|----------|-----------|
| `words` | one headword | `word` UNIQUE NOCASE, `length`, `first_letter`, `syllables`, `frequency_rank` |
| `senses` | one *meaning* | FK → `words` (CASCADE), `UNIQUE(word_id, sense_no)`, POS `CHECK` |
| `examples` | usage sentences | FK → `senses` (CASCADE) |
| `synonyms` | synonym edges | FK → `words`, `UNIQUE(word_id, synonym)`, `strength` 0–1 |
| `antonyms` | antonym edges | FK → `words`, `UNIQUE(word_id, antonym)` |
| `relations` | typed lexical links | `relation ∈ {hypernym, hyponym, derived, related, see_also, meronym, holonym, …}` |
| `etymology` | origin (1:1) | `origin_language`, `root`, `note` |
| `pronunciations` | per accent | `UNIQUE(word_id, accent)`, `ipa`, `respelling` |
| `word_frequency` | corpus counts | `count`, `per_million`, `corpus` |
| `anagrams` | letter keys | `letter_key` = letters sorted; indexed |

### Integrity constraints used

```sql
FOREIGN KEY (word_id) REFERENCES words(id) ON DELETE CASCADE
UNIQUE (word_id, sense_no)          -- a word has senses 1..k, no duplicates
UNIQUE (word_id, synonym)           -- no repeated synonym edges
CHECK (part_of_speech IN (...))     -- closed vocabulary of POS tags
CHECK (length >= 1)
CHECK (strength >= 0 AND strength <= 1)
```

Cascading deletes mean removing a headword cleans up every dependent row.

---

## 3. Views (the analytical layer)

```sql
v_word_summary       -- word + first sense + counts (senses, synonyms, examples)
v_parts_of_speech    -- senses/words per POS tag
v_length_distribution-- how many words of each length
```

Views keep complex joins out of application code; both `stats()` and the Excel
exporter consume them.

---

## 4. Query API

| Method | What it does | Example |
|--------|--------------|---------|
| `define(word)` | full entry (senses, examples, synonyms, antonyms, relations, etymology, pronunciation) | `define("bank")` |
| `lookup(pattern)` | LIKE-based search (`%` / `_`) | `lookup("%our%")` |
| `prefix(s)` / `suffix(s)` | words starting/ending with `s` | `prefix("qu")` |
| `wildcard(pat)` | `_` = one letter, `%` = any run | `wildcard("c_t")` → cat, cot, cut |
| `anagrams(word)` | exact anagrams via the precomputed key | `anagrams("listen")` → silence-free set |
| `sub_anagrams(letters)` | Scrabble-style subset words | `sub_anagrams("silent")` |
| `rhymes(word, min_match)` | same word ending | `rhymes("light", 2)` |
| `by_length(n)` / `by_pos(tag)` | structural filters | `by_length(3)` |
| `synonyms_of` / `antonyms_of` | edge lookups | `synonyms_of("happy")` |
| `relations_of(word, rel)` | typed relations | `relations_of("cat", "hypernym")` |
| `search_definitions(text)` | text search over definitions | `search_definitions("temperature")` |
| `random_word` / `word_of_the_day` | sampling | `word_of_the_day()` |
| `stats()` / `report()` / `integrity_report()` | summaries | `report()` |

### How anagrams work

Naively, anagrams require comparing every word. Instead we store a **letter
key** — the word's letters sorted alphabetically:

```
listen -> eilnst
silent -> eilnst
enlist -> eilnst
tinsel -> eilnst
```

Anagram lookup is then an indexed equality join on `letter_key`: `O(log n)`
instead of `O(n · |w|)`.

---

## 5. Seeding — `db/lexicon.py`

The seed data is a curated list of common English words. Each entry carries:

```python
(word, syllables, frequency_rank, origin_language, root, ipa, respelling,
 [(pos, definition, register, domain, example), ...],
 [synonyms], [antonyms], [(relation, target, note), ...])
```

`seed(dictionary)` inserts everything idempotently (`ON CONFLICT … DO UPDATE`),
so re-running never duplicates data:

```python
from db import lexicon
d = lexicon.build()          # creates db/english.db if absent
print(d.report())
```

Current seed: 54 headwords, 72 senses, 149 synonym edges, 45 antonym edges, 21
typed relations, full etymology and IPA for every entry.

---

## 6. Excel export

`db.excel.dictionary_to_excel(dict, path)` writes one worksheet per table plus
three report sheets:

```
english.xlsx
├── words                ├── relations          ├── anagrams
├── senses               ├── etymology          ├── _parts_of_speech
├── examples             ├── pronunciations     ├── _length_distribution
├── synonyms             ├── word_frequency     └── _word_summary
└── antonyms
```

Formatting: bold header row, frozen panes, auto-fitted widths, auto-filter.

---

## 7. Worked example

```python
from db.lexicon import build

with build() as d:
    # multi-sense lookup
    entry = d.define("bank")
    for s in entry["senses"]:
        print(s["part_of_speech"], "-", s["definition"])
        for ex in s["examples"]:
            print("   e.g.", ex)

    print(d.anagrams("listen"))       # enlist, silent, tinsel
    print(d.wildcard("c_t"))          # cat, cot, cut
    print(d.synonyms_of("happy"))     # glad, joyful, cheerful, content
    print(d.stats()["by_pos"])
```

---

## 8. Extending the dictionary

* **More words**: append entries to `db/lexicon.py`'s `ENTRIES`, or use
  `d.add_entry(...)` for a single word.
* **More relations**: `d.add_relation("cat", "meronym", "whisker")`.
* **Import a word list**: load a file and loop `d.add_word(...)`.
* **Full-text search**: SQLite's FTS5 extension can be added with a virtual
  table (`CREATE VIRTUAL TABLE senses_fts USING fts5(definition)`).
* **Multiple languages**: add a `language` column to `words` and index it.
