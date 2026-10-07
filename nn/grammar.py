"""
nn.grammar -- a complete English grammar system.

Everything needed to describe, generate and analyse English sentences:

1. **Parts of speech** (`POS`) -- the 9 classical word classes plus fine tags
   (determiner, preposition, conjunction, pronoun, modal, ...).

2. **Morphology** -- word formation and analysis:
   * `pluralize` / `singularize` (with irregulars: man->men, child->children)
   * `conjugate`  (5 forms: base, 3rd-person singular, past, past participle,
     present participle) with regular rules + irregular verb table
   * `comparative` / `superlative` (big->bigger->biggest, happy->happier)
   * `possessive`, `derive_adverb`, `derive_noun`
   * `morphological_parse` -- take a surface form and recover its lemma + features

3. **Syntax** -- phrase and clause structure:
   * `Phrase` / `Clause` structures (NP, VP, PP, AdjP, AdvP)
   * a rule-based POS **tagger** and a **parser** producing a tree
   * dependency arcs (subject, object, modifier)
   * constituent tests

4. **Agreement & rules** -- subject-verb agreement, article selection (a/an),
   pronoun case, tense consistency, and a **grammar checker** that reports
   diagnostics with spans.

5. **Generation** -- `Sentence` builders and `conjugate_phrase` to produce
   correct surface forms from a feature bundle.

The system is rule-based (deterministic, inspectable) and dependency-free.
Letter/number features use small closed sets so the rules are exact.

Usage
-----
    from nn.grammar import conjugate, pluralize, tag, parse, check

    conjugate("go", "3sg")        # 'goes'
    conjugate("go", "past")       # 'went'
    pluralize("child")            # 'children'
    tag("The dogs bark loudly")   # [('The','DT'), ('dogs','NNS'), ...]
    parse("The dogs bark")        # a tree with subject/predicate
    check("The dogs barks")       # [agreement error diagnostic]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    # parts of speech
    "POS", "POS_NAMES", "is_pos", "FUNCTION_WORDS",
    # morphology
    "pluralize", "singularize", "conjugate", "all_forms", "comparative",
    "superlative", "possessive", "derive_adverb", "derive_noun",
    "morphological_parse", "VerbForm", "Number", "Tense", "Person",
    # tagging
    "Token", "tag", "tag_sentence", "Lexicon",
    # syntax
    "Node", "Phrase", "Clause", "parse", "find_subject", "find_predicate",
    "find_objects", "noun_phrase", "verb_phrase",
    # agreement / checks
    "agree_subject_verb", "article_for", "check", "Diagnostic",
    # generation
    "Sentence", "conjugate_phrase",
]


# ===========================================================================
# 1. Parts of speech
# ===========================================================================
class POS(str, Enum):
    """The word classes of English (classical 9 + fine-grained tags)."""
    NOUN = "NOUN"
    VERB = "VERB"
    ADJECTIVE = "ADJECTIVE"
    ADVERB = "ADVERB"
    PRONOUN = "PRONOUN"
    PREPOSITION = "PREPOSITION"
    CONJUNCTION = "CONJUNCTION"
    INTERJECTION = "INTERJECTION"
    DETERMINER = "DETERMINER"
    NUMERAL = "NUMERAL"
    MODAL = "MODAL"
    AUXILIARY = "AUXILIARY"
    PARTICLE = "PARTICLE"
    UNKNOWN = "UNKNOWN"


# Penn-Treebank-style fine tags used by the tagger.
POS_NAMES = {
    "DT": "determiner",
    "NN": "noun, singular",
    "NNS": "noun, plural",
    "NNP": "proper noun",
    "VB": "verb, base form",
    "VBZ": "verb, 3rd person singular",
    "VBP": "verb, non-3rd person singular",
    "VBD": "verb, past tense",
    "VBN": "verb, past participle",
    "VBG": "verb, gerund/present participle",
    "MD": "modal",
    "JJ": "adjective",
    "JJR": "adjective, comparative",
    "JJS": "adjective, superlative",
    "RB": "adverb",
    "PRP": "personal pronoun",
    "PRP$": "possessive pronoun",
    "IN": "preposition/subordinating conjunction",
    "CC": "coordinating conjunction",
    "CD": "cardinal number",
    "TO": "the word 'to'",
    "UH": "interjection",
    "RP": "particle",
    "EX": "existential 'there'",
    "POS_FIN": "possessive ending",
    ".": "punctuation",
}

_FINE_TO_POS = {
    "DT": POS.DETERMINER, "NN": POS.NOUN, "NNS": POS.NOUN, "NNP": POS.NOUN,
    "VB": POS.VERB, "VBZ": POS.VERB, "VBP": POS.VERB, "VBD": POS.VERB,
    "VBN": POS.VERB, "VBG": POS.VERB, "MD": POS.MODAL, "JJ": POS.ADJECTIVE,
    "JJR": POS.ADJECTIVE, "JJS": POS.ADJECTIVE, "RB": POS.ADVERB,
    "PRP": POS.PRONOUN, "PRP$": POS.PRONOUN, "IN": POS.PREPOSITION,
    "CC": POS.CONJUNCTION, "CD": POS.NUMERAL, "TO": POS.PARTICLE,
    "UH": POS.INTERJECTION, "RP": POS.PARTICLE, "EX": POS.PRONOUN,
    "POS_FIN": POS.PARTICLE, ".": POS.UNKNOWN,
}


def is_pos(tag: str, category: POS) -> bool:
    """True if fine tag `tag` belongs to coarse category `category`."""
    return _FINE_TO_POS.get(tag) == category


# Closed-class function words (never nouns/verbs).
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your",
               "his", "her", "its", "our", "their", "some", "any", "each",
               "every", "no", "both", "either", "neither", "many", "much",
               "few", "little", "several", "all"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "myself", "yourself", "himself", "herself", "itself",
            "ourselves", "themselves", "who", "whom", "whose", "which", "what",
            "someone", "anyone", "everyone", "nobody", "something", "anything",
            "everything", "nothing", "there"}
PREPOSITIONS = {"in", "on", "at", "by", "for", "with", "about", "against",
                "between", "into", "through", "during", "before", "after",
                "above", "below", "to", "from", "up", "down", "of", "off",
                "over", "under", "again", "near", "without", "within", "across",
                "behind", "beyond", "toward", "towards", "upon", "among"}
CONJUNCTIONS = {"and", "or", "but", "nor", "for", "yet", "so", "because",
                "although", "though", "while", "whereas", "if", "unless",
                "since", "until", "when", "whenever", "where", "whether"}
MODALS = {"can", "could", "may", "might", "must", "shall", "should", "will",
          "would", "ought"}
AUXILIARIES = {"be", "am", "is", "are", "was", "were", "been", "being",
               "have", "has", "had", "having", "do", "does", "did", "done"}
INTERJECTIONS = {"oh", "wow", "hey", "hi", "hello", "alas", "ouch", "hooray",
                 "oops", "aha", "hmm", "uh"}

FUNCTION_WORDS = (DETERMINERS | PRONOUNS | PREPOSITIONS | CONJUNCTIONS
                  | MODALS | AUXILIARIES | INTERJECTIONS)


# ===========================================================================
# 2. Morphology
# ===========================================================================
class Number(str, Enum):
    SINGULAR = "sg"
    PLURAL = "pl"


class Person(int, Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


class Tense(str, Enum):
    PRESENT = "present"
    PAST = "past"


class VerbForm(str, Enum):
    BASE = "base"                 # go, walk
    S3 = "3sg"                    # goes, walks      (present, 3rd, singular)
    PAST = "past"                 # went, walked
    PAST_PART = "pp"              # gone, walked
    PRESENT_PART = "ing"          # going, walking


# ---------------------------------------------------------------------------
# Irregular tables
# ---------------------------------------------------------------------------
IRREGULAR_PLURALS = {
    "man": "men", "woman": "women", "child": "children", "person": "people",
    "tooth": "teeth", "foot": "feet", "goose": "geese", "mouse": "mice",
    "louse": "lice", "ox": "oxen", "die": "dice", "leaf": "leaves",
    "life": "lives", "knife": "knives", "wife": "wives", "wolf": "wolves",
    "shelf": "shelves", "thief": "thieves", "loaf": "loaves", "half": "halves",
    "calf": "calves", "elf": "elves", "self": "selves", "crisis": "crises",
    "analysis": "analyses", "basis": "bases", "datum": "data",
    "criterion": "criteria", "phenomenon": "phenomena", "index": "indices",
    "matrix": "matrices", "medium": "media", "axis": "axes", "alumnus": "alumni",
    "cactus": "cacti", "fungus": "fungi", "nucleus": "nuclei", "radius": "radii",
    "stimulus": "stimuli", "syllabus": "syllabi", "sheep": "sheep",
    "fish": "fish", "deer": "deer", "series": "series", "species": "species",
    "aircraft": "aircraft", "salmon": "salmon", "trout": "trout",
}
IRREGULAR_PLURALS_REVERSE = {v: k for k, v in IRREGULAR_PLURALS.items()}

# verb -> (past, past participle, 3rd-sg)   [ing form is derived by rule]
IRREGULAR_VERBS = {
    "be": ("was", "been", "is"),
    "have": ("had", "had", "has"),
    "do": ("did", "done", "does"),
    "go": ("went", "gone", "goes"),
    "say": ("said", "said", "says"),
    "get": ("got", "gotten", "gets"),
    "make": ("made", "made", "makes"),
    "know": ("knew", "known", "knows"),
    "think": ("thought", "thought", "thinks"),
    "take": ("took", "taken", "takes"),
    "see": ("saw", "seen", "sees"),
    "come": ("came", "come", "comes"),
    "want": ("wanted", "wanted", "wants"),
    "give": ("gave", "given", "gives"),
    "find": ("found", "found", "finds"),
    "tell": ("told", "told", "tells"),
    "become": ("became", "become", "becomes"),
    "leave": ("left", "left", "leaves"),
    "feel": ("felt", "felt", "feels"),
    "bring": ("brought", "brought", "brings"),
    "begin": ("began", "begun", "begins"),
    "keep": ("kept", "kept", "keeps"),
    "hold": ("held", "held", "holds"),
    "write": ("wrote", "written", "writes"),
    "stand": ("stood", "stood", "stands"),
    "hear": ("heard", "heard", "hears"),
    "let": ("let", "let", "lets"),
    "mean": ("meant", "meant", "means"),
    "set": ("set", "set", "sets"),
    "meet": ("met", "met", "meets"),
    "run": ("ran", "run", "runs"),
    "pay": ("paid", "paid", "pays"),
    "sit": ("sat", "sat", "sits"),
    "speak": ("spoke", "spoken", "speaks"),
    "lie": ("lay", "lain", "lies"),
    "lead": ("led", "led", "leads"),
    "read": ("read", "read", "reads"),
    "grow": ("grew", "grown", "grows"),
    "lose": ("lost", "lost", "loses"),
    "fall": ("fell", "fallen", "falls"),
    "send": ("sent", "sent", "sends"),
    "build": ("built", "built", "builds"),
    "understand": ("understood", "understood", "understands"),
    "draw": ("drew", "drawn", "draws"),
    "break": ("broke", "broken", "breaks"),
    "spend": ("spent", "spent", "spends"),
    "cut": ("cut", "cut", "cuts"),
    "rise": ("rose", "risen", "rises"),
    "drive": ("drove", "driven", "drives"),
    "buy": ("bought", "bought", "buys"),
    "wear": ("wore", "worn", "wears"),
    "choose": ("chose", "chosen", "chooses"),
    "eat": ("ate", "eaten", "eats"),
    "drink": ("drank", "drunk", "drinks"),
    "sing": ("sang", "sung", "sings"),
    "swim": ("swam", "swum", "swims"),
    "fly": ("flew", "flown", "flies"),
    "throw": ("threw", "thrown", "throws"),
    "catch": ("caught", "caught", "catches"),
    "teach": ("taught", "taught", "teaches"),
    "fight": ("fought", "fought", "fights"),
    "sleep": ("slept", "slept", "sleeps"),
    "win": ("won", "won", "wins"),
    "forget": ("forgot", "forgotten", "forgets"),
    "freeze": ("froze", "frozen", "freezes"),
}
IRREGULAR_VERBS_REVERSE = {}
for _base, (_past, _pp, _s3) in IRREGULAR_VERBS.items():
    for _form, _kind in ((_past, "past"), (_pp, "pp"), (_s3, "s3")):
        IRREGULAR_VERBS_REVERSE.setdefault(_form, []).append((_base, _kind))

IRREGULAR_COMPARATIVES = {
    "good": ("better", "best"), "bad": ("worse", "worst"),
    "far": ("farther", "farthest"), "little": ("less", "least"),
    "much": ("more", "most"), "many": ("more", "most"),
    "well": ("better", "best"), "old": ("older", "oldest"),
}

# Spellings that the general -ing rules cannot derive unambiguously.
_ING_EXCEPTIONS = {
    "dye": "dyeing", "eye": "eyeing", "age": "aging", "singe": "singeing",
    "tinge": "tingeing", "picnic": "picnicking", "traffic": "trafficking",
    "mimic": "mimicking", "panic": "panicking", "focus": "focusing",
    "travel": "traveling", "cancel": "canceling", "label": "labeling",
    "argue": "arguing", "true": "truing", "queue": "queuing",
}

_VOWELS = set("aeiou")


def _ends_consonant_y(word: str) -> bool:
    return len(word) >= 2 and word[-1] == "y" and word[-2] not in _VOWELS


def _is_cvc(word: str) -> bool:
    """Consonant-vowel-consonant ending (for doubling: run -> running)."""
    if len(word) < 3:
        return False
    a, b, c = word[-3].lower(), word[-2].lower(), word[-1].lower()
    return (a not in _VOWELS and b in _VOWELS and c not in _VOWELS
            and c not in "wxy")


# ---------------------------------------------------------------------------
# Nouns
# ---------------------------------------------------------------------------
def pluralize(word: str, count: float | None = None) -> str:
    """Return the plural of a noun (irregulars handled).

    `count` (optional): if provided and equal to 1, returns the singular, so
    `pluralize('cat', 1)` -> 'cat' and `pluralize('cat', 3)` -> 'cats'.
    """
    if count is not None and count == 1:
        return word
    w = word.strip()
    low = w.lower()
    if not w:
        return w
    if low in IRREGULAR_PLURALS:
        return _match_case(w, IRREGULAR_PLURALS[low])
    # already plural?  (heuristic: ends in s but not ss/us/is)
    if low.endswith("s") and not (low.endswith(("ss", "us", "is", "as"))):
        return w
    if low.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if _ends_consonant_y(low):
        return w[:-1] + "ies"
    if low.endswith("o") and len(low) > 1 and low[-2] not in _VOWELS:
        return w + "es"          # potato -> potatoes (with common exceptions)
    if low.endswith("f"):
        return w[:-1] + "ves"    # (covered mostly by the irregular table)
    if low.endswith("fe"):
        return w[:-2] + "ves"
    return w + "s"


def singularize(word: str) -> str:
    """Return the singular of a plural noun (irregulars handled)."""
    w = word.strip()
    low = w.lower()
    if not w:
        return w
    if low in IRREGULAR_PLURALS_REVERSE:
        return _match_case(w, IRREGULAR_PLURALS_REVERSE[low])
    if low.endswith("ies") and len(low) > 3:
        return w[:-3] + "y"
    if low.endswith("ves"):
        base = w[:-3] + "f"
        if base.lower() in IRREGULAR_PLURALS:
            return base
        return w[:-3] + "fe"
    if low.endswith("es") and low[-4:-2] in ("ch", "sh", "ss", "x", "z"):
        return w[:-2]
    if low.endswith("s") and not low.endswith("ss"):
        return w[:-1]
    return w


def _match_case(original: str, replacement: str) -> str:
    """Copy the capitalization style of `original` onto `replacement`."""
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


# ---------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------
def _ing_form(base: str) -> str:
    low = base.lower()
    # Spelling exceptions where the general rules would be ambiguous.
    if low in _ING_EXCEPTIONS:
        return _match_case(base, _ING_EXCEPTIONS[low])
    if low.endswith("ie"):
        return base[:-2] + "ying"          # lie -> lying, die -> dying
    # Silent -e is dropped (make -> making), but the e is KEPT after a vowel
    # (see, agree), in very short roots (be), or to keep the consonant soft
    # (singe -> singeing, dye -> dyeing).
    if low.endswith("e"):
        if len(low) <= 2:
            return base + "ing"             # be -> being
        if low[-2] in "aeiou":
            if low.endswith("ye"):
                return base + "ing"         # dye -> dyeing, eye -> eyeing
            if low.endswith("ue"):
                return base[:-1] + "ing"        # argue -> arguing, true -> truing
            if low.endswith(("oe", "ee", "ie")):
                return base + "ing"         # hoe -> hoeing, see -> seeing
            return base[:-1] + "ing"        # have -> having
        # consonant + 'e': drop the e, but keep soft g/c spellings intact
        if low.endswith("ge"):
            return base + "ing"             # singe -> singeing
        return base[:-1] + "ing"            # make -> making
    # consonant doubling: run -> running, panic -> panicking, travel -> travelling
    if _is_cvc(low) and low[-1] not in "wxy":
        if low[-1] == "c":
            return base + "king"            # panic -> panicking, mimic -> mimicking
        return base + low[-1] + "ing"       # run -> running, stop -> stopping
    return base + "ing"                    # study -> studying, fix -> fixing


def _s3_form(base: str) -> str:
    low = base.lower()
    if low.endswith("c"):                   # panic -> panics  (k is not added)
        return base + "s"
    if low.endswith(("s", "x", "z", "ch", "sh", "o")):
        return base + "es"
    if _ends_consonant_y(low):
        return base[:-1] + "ies"
    return base + "s"


def _past_regular(base: str) -> str:
    low = base.lower()
    if low.endswith("e"):
        return base + "d"                  # love -> loved
    if _ends_consonant_y(low):
        return base[:-1] + "ied"           # try -> tried
    if low.endswith("c"):                   # panic -> panicked, mimic -> mimicked
        return base + "ked"
    if _is_cvc(low) and len(low) <= 4:
        return base + low[-1] + "ed"       # stop -> stopped
    return base + "ed"


def conjugate(verb: str, form: VerbForm | str) -> str:
    """Return the requested form of a verb.

    form: 'base' | '3sg' | 'past' | 'pp' | 'ing'
          (aliases: 's3', 'past_participle', 'present_participle', ...)
    """
    key = str(form.value if isinstance(form, VerbForm) else form).lower()
    aliases = {"s3": "3sg", "third": "3sg", "present3": "3sg",
               "past_participle": "pp", "participle": "pp",
               "present_participle": "ing", "gerund": "ing", "progressive": "ing",
               "infinitive": "base", "vbn": "pp", "vbz": "3sg", "vbd": "past",
               "vbg": "ing"}
    key = aliases.get(key, key)

    v = verb.strip()
    low = v.lower()

    if key == "base":
        return v
    if key in ("3sg", "past"):
        idx = 2 if key == "3sg" else 0
        if low in IRREGULAR_VERBS:
            return _match_case(v, IRREGULAR_VERBS[low][idx])
        return _match_case(v, _s3_form(low) if key == "3sg"
                           else _past_regular(low))
    if key == "pp":
        if low in IRREGULAR_VERBS:
            return _match_case(v, IRREGULAR_VERBS[low][1])
        return _match_case(v, _past_regular(low))
    if key == "ing":
        return _match_case(v, _ing_form(low))
    raise ValueError(f"unknown verb form '{form}'")


def all_forms(verb: str) -> dict:
    """Return all five principal parts of a verb."""
    return {
        "base": conjugate(verb, "base"),
        "3sg": conjugate(verb, "3sg"),
        "past": conjugate(verb, "past"),
        "past_participle": conjugate(verb, "pp"),
        "present_participle": conjugate(verb, "ing"),
    }


# ---------------------------------------------------------------------------
# Adjectives / adverbs
# ---------------------------------------------------------------------------
def comparative(adj: str) -> str:
    """Comparative form: big -> bigger, happy -> happier, good -> better."""
    w = adj.strip()
    low = w.lower()
    if low in IRREGULAR_COMPARATIVES:
        return _match_case(w, IRREGULAR_COMPARATIVES[low][0])
    if _ends_consonant_y(low):
        return w[:-1] + "ier"
    if _is_cvc(low) and len(low) <= 5:
        return w + low[-1] + "er"
    if low.endswith("e"):
        return w + "r"
    # long adjectives take "more"
    if len(low) > 6:
        return "more " + w
    return w + "er"


def superlative(adj: str) -> str:
    """Superlative form: big -> biggest, happy -> happiest, good -> best."""
    w = adj.strip()
    low = w.lower()
    if low in IRREGULAR_COMPARATIVES:
        return _match_case(w, IRREGULAR_COMPARATIVES[low][1])
    if _ends_consonant_y(low):
        return w[:-1] + "iest"
    if _is_cvc(low) and len(low) <= 5:
        return w + low[-1] + "est"
    if low.endswith("e"):
        return w + "st"
    if len(low) > 6:
        return "most " + w
    return w + "est"


def possessive(noun: str) -> str:
    """Possessive form: dog -> dog's, dogs -> dogs', James -> James'."""
    w = noun.strip()
    if w.endswith(("s", "'", "’")):
        return w + "'" if not w.endswith("'") else w
    return w + "'s"


def derive_adverb(adj: str) -> str:
    """Form an adverb from an adjective (quick -> quickly, happy -> happily)."""
    w = adj.strip()
    low = w.lower()
    irregular = {"good": "well", "fast": "fast", "hard": "hard", "late": "late",
                 "early": "early", "high": "high", "low": "low", "long": "long",
                 "straight": "straight"}
    if low in irregular:
        return _match_case(w, irregular[low])
    if low.endswith("le"):
        return w[:-1] + "y"                # simple -> simply
    if _ends_consonant_y(low):
        return w[:-1] + "ily"              # happy -> happily
    if low.endswith("ic"):
        return w + "ally"                  # basic -> basically
    return w + "ly"


def derive_noun(adj: str) -> str:
    """Form an abstract noun from an adjective (happy -> happiness)."""
    w = adj.strip()
    low = w.lower()
    irregular = {"long": "length", "strong": "strength", "wide": "width",
                 "deep": "depth", "high": "height", "true": "truth",
                 "young": "youth", "warm": "warmth", "free": "freedom",
                 "wise": "wisdom", "hot": "heat", "proud": "pride"}
    if low in irregular:
        return _match_case(w, irregular[low])
    if low.endswith("y") and not low.endswith(("ay", "ey", "oy")):
        return w[:-1] + "iness"            # happy -> happiness
    if low.endswith("e"):
        return w + "ness"                  # pale -> paleness
    if low.endswith(("t", "s", "d")):
        return w + "ness"
    return w + "ness"


# ---------------------------------------------------------------------------
# Morphological analysis
# ---------------------------------------------------------------------------
def morphological_parse(form: str) -> list:
    """Recover possible (lemma, features) analyses of a surface form.

    Returns a list of `{'lemma', 'pos', 'features'}` dicts, ordered by
    likelihood. This is the inverse of the generation functions above.
    """
    w = form.strip()
    low = w.lower()
    if not w:
        return []
    out = []

    # irregular verbs: exact match on past / pp / 3sg
    if low in IRREGULAR_VERBS_REVERSE:
        for base, kind in IRREGULAR_VERBS_REVERSE[low]:
            feat = {"s3": {"number": "sg", "person": 3, "tense": "present"},
                    "past": {"tense": "past"},
                    "pp": {"form": "past_participle"}}[kind]
            out.append({"lemma": base, "pos": POS.VERB, "features": feat})
    if low in IRREGULAR_VERBS:               # base form that is irregular
        out.append({"lemma": low, "pos": POS.VERB,
                    "features": {"form": "base"}})

    # plural nouns
    singular = singularize(low)
    if singular != low:
        out.append({"lemma": singular, "pos": POS.NOUN,
                    "features": {"number": "plural"}})

    # verb endings
    if low.endswith("ing") and len(low) > 4:
        base = _strip_ing(low)
        out.append({"lemma": base, "pos": POS.VERB,
                    "features": {"form": "present_participle"}})
    if low.endswith("ed") and len(low) > 3:
        base = low[:-2] if not low.endswith("ed") or low[-3] not in _VOWELS \
            else low[:-1]
        if low.endswith("ied"):
            base = low[:-3] + "y"
        elif low.endswith("ed") and len(low) > 4 and low[-3] == low[-4] \
                and low[-4] not in _VOWELS:
            base = low[:-3]                   # stopped -> stop
        out.append({"lemma": base, "pos": POS.VERB, "features": {"tense": "past"}})
    if low.endswith("s") and not low.endswith("ss"):
        out.append({"lemma": low[:-1], "pos": POS.VERB,
                    "features": {"3sg": True, "tense": "present"}})

    # adverb / adjective derivations
    if low.endswith("ly"):
        out.append({"lemma": low[:-2], "pos": POS.ADVERB,
                    "features": {"derived_from": "adjective"}})
    if low.endswith(("er",)) and len(low) > 3:
        out.append({"lemma": low[:-2], "pos": POS.ADJECTIVE,
                    "features": {"degree": "comparative"}})
    if low.endswith(("est",)) and len(low) > 4:
        out.append({"lemma": low[:-3], "pos": POS.ADJECTIVE,
                    "features": {"degree": "superlative"}})

    # possessive
    if low.endswith("'s") or low.endswith("'s"):
        out.append({"lemma": low[:-2], "pos": POS.NOUN,
                    "features": {"case": "possessive"}})

    if not out:
        out.append({"lemma": low, "pos": POS.UNKNOWN, "features": {}})
    return out


def _strip_ing(low: str) -> str:
    """Invert the -ing rule: running -> run, making -> make, lying -> lie."""
    stem = low[:-3]
    if stem.endswith("y") and len(stem) > 1 and stem[-2] not in _VOWELS:
        return low[:-3]                       # lying handled below
    if low.endswith("ying"):
        return low[:-4] + "ie"                # lying -> lie
    if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in _VOWELS:
        return stem[:-1]                      # running -> run
    if stem.endswith(("at", "iz", "iv")):
        return stem + "e"                     # making-ish
    return stem
