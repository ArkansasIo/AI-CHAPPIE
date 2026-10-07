"""
nn.conjugation -- a complete English verb conjugation system.

This is the *paradigm* layer above `nn.grammar`: it produces every inflected
form of a verb across the full English tense/aspect/mood system, with person and
number agreement, do-support, negation, questions, contractions and voice.

Covered
-------
Simple tenses      present simple, past simple, future (will), future (going to)
Continuous         present, past, future continuous
Perfect            present, past, future perfect
Perfect contin.    present, past, future perfect continuous
Conditional        would / would have / would have been + -ing
Subjunctive        present ("if I were"), past perfect, mandative ("that he go")
Imperative         affirmative & negative
Passive            all tenses, all persons
Modals             can/could/may/might/must/shall/should/will/would/ought
Negation           full & contracted ("do not" / "don't", "cannot" / "can't")
Questions          yes/no and wh-, including do-support and inversion
Contractions       I'm, you're, he's, we've, they'll, isn't, won't, ...

Data model
----------
* `Person` (1/2/3) and `Number` (sg/pl) -> a `Subject`.
* `Tense`, `Aspect`, `Mood`, `Voice` enums describe the paradigm slot.
* `conjugate(verb, ...)` returns the surface form(s) for a slot.
* `conjugate_all(verb)` returns the full `ConjugationTable`.
* `ConjugatedVerb` bundles a verb with its subject and options, exposing
  `.form()`, `.negative()`, `.question()`, `.passive()`, `.contract()`.

Everything is deterministic and dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .grammar import (
    Number as GNumber, Person as GPerson, Tense as GTense,
    VerbForm, conjugate as morph_conjugate, all_forms,
)

__all__ = [
    "Person", "Number", "Tense", "Aspect", "Mood", "Voice",
    "Subject", "SUBJECT_PRONOUNS", "POSSESSIVE_DETERMINERS",
    "ConjugatedVerb", "ConjugationTable",
    "conjugate", "conjugate_all", "conjugate_table",
    "be_form", "have_form", "do_form", "modal_form",
    "negate", "question", "to_be", "to_have", "to_do",
    "CONTRACTIONS", "contract", "expand_contractions",
    "IRREGULAR_BE", "AUX_BE", "AUX_HAVE", "AUX_DO", "MODALS",
]


# ===========================================================================
# 1. Feature enums
# ===========================================================================
class Person(int, Enum):
    FIRST = 1
    SECOND = 2
    THIRD = 3


class Number(str, Enum):
    SINGULAR = "sg"
    PLURAL = "pl"


class Tense(str, Enum):
    PRESENT = "present"
    PAST = "past"
    FUTURE = "future"
    CONDITIONAL = "conditional"


class Aspect(str, Enum):
    SIMPLE = "simple"
    CONTINUOUS = "continuous"
    PERFECT = "perfect"
    PERFECT_CONTINUOUS = "perfect_continuous"


class Mood(str, Enum):
    INDICATIVE = "indicative"
    SUBJUNCTIVE = "subjunctive"
    IMPERATIVE = "imperative"


class Voice(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


# ===========================================================================
# 2. Subjects and pronouns
# ===========================================================================
# (person, number) -> pronoun
SUBJECT_PRONOUNS = {
    (Person.FIRST, Number.SINGULAR): "I",
    (Person.SECOND, Number.SINGULAR): "you",
    (Person.THIRD, Number.SINGULAR): "he/she/it",
    (Person.FIRST, Number.PLURAL): "we",
    (Person.SECOND, Number.PLURAL): "you",
    (Person.THIRD, Number.PLURAL): "they",
}

OBJECT_PRONOUNS = {
    (Person.FIRST, Number.SINGULAR): "me",
    (Person.SECOND, Number.SINGULAR): "you",
    (Person.THIRD, Number.SINGULAR): "him/her/it",
    (Person.FIRST, Number.PLURAL): "us",
    (Person.SECOND, Number.PLURAL): "you",
    (Person.THIRD, Number.PLURAL): "them",
}

POSSESSIVE_DETERMINERS = {
    (Person.FIRST, Number.SINGULAR): "my",
    (Person.SECOND, Number.SINGULAR): "your",
    (Person.THIRD, Number.SINGULAR): "his/her/its",
    (Person.FIRST, Number.PLURAL): "our",
    (Person.SECOND, Number.PLURAL): "your",
    (Person.THIRD, Number.PLURAL): "their",
}


@dataclass(frozen=True)
class Subject:
    """A grammatical subject: person + number (+ optional explicit noun)."""
    person: Person = Person.THIRD
    number: Number = Number.SINGULAR
    noun: str | None = None          # e.g. "the dog" instead of "it"

    @property
    def pronoun(self) -> str:
        return SUBJECT_PRONOUNS[(self.person, self.number)]

    @property
    def is_third_singular(self) -> bool:
        return self.person == Person.THIRD and self.number == Number.SINGULAR

    @property
    def word(self) -> str:
        """The surface subject (explicit noun or pronoun)."""
        return self.noun if self.noun else self.pronoun

    def agree_verb(self, base_form: str, present_form: str) -> str:
        """Pick the correct present-tense verb form for this subject."""
        return present_form if self.is_third_singular else base_form

    @staticmethod
    def i() -> "Subject":
        return Subject(Person.FIRST, Number.SINGULAR)

    @staticmethod
    def you() -> "Subject":
        return Subject(Person.SECOND, Number.SINGULAR)

    @staticmethod
    def he() -> "Subject":
        return Subject(Person.THIRD, Number.SINGULAR)

    @staticmethod
    def we() -> "Subject":
        return Subject(Person.FIRST, Number.PLURAL)

    @staticmethod
    def they() -> "Subject":
        return Subject(Person.THIRD, Number.PLURAL)

    def __str__(self):
        return self.word


# ===========================================================================
# 3. Auxiliary verb tables (fully irregular -- must be stored, not derived)
# ===========================================================================
IRREGULAR_BE = {
    (Tense.PRESENT, Person.FIRST, Number.SINGULAR): "am",
    (Tense.PRESENT, Person.SECOND, Number.SINGULAR): "are",
    (Tense.PRESENT, Person.THIRD, Number.SINGULAR): "is",
    (Tense.PRESENT, Person.FIRST, Number.PLURAL): "are",
    (Tense.PRESENT, Person.SECOND, Number.PLURAL): "are",
    (Tense.PRESENT, Person.THIRD, Number.PLURAL): "are",
    (Tense.PAST, Person.FIRST, Number.SINGULAR): "was",
    (Tense.PAST, Person.SECOND, Number.SINGULAR): "were",
    (Tense.PAST, Person.THIRD, Number.SINGULAR): "was",
    (Tense.PAST, Person.FIRST, Number.PLURAL): "were",
    (Tense.PAST, Person.SECOND, Number.PLURAL): "were",
    (Tense.PAST, Person.THIRD, Number.PLURAL): "were",
}

AUX_BE = ["am", "is", "are", "was", "were", "be", "been", "being"]
AUX_HAVE = ["have", "has", "had", "having"]
AUX_DO = ["do", "does", "did", "done", "doing"]

MODALS = ["can", "could", "may", "might", "must", "shall", "should",
          "will", "would", "ought"]

# Modals are defective: no -s, no -ing, no infinitive; the next verb is bare.
MODAL_PAST = {"can": "could", "may": "might", "shall": "should",
              "will": "would", "must": "must", "ought": "ought"}


# ===========================================================================
# 4. Auxiliary generators
# ===========================================================================
def be_form(tense: Tense, person: Person, number: Number) -> str:
    """The correct form of *be* for a tense/person/number slot."""
    if tense == Tense.PAST:
        key = (Tense.PAST, person, number)
    else:
        key = (Tense.PRESENT, person, number)
    return IRREGULAR_BE.get(key, "be")


def have_form(tense: Tense, person: Person, number: Number) -> str:
    """The correct form of *have* (present: have/has; past: had)."""
    if tense == Tense.PAST:
        return "had"
    return "has" if (person == Person.THIRD and number == Number.SINGULAR) else "have"


def do_form(tense: Tense, person: Person, number: Number) -> str:
    """The correct form of *do* (do/does/did)."""
    if tense == Tense.PAST:
        return "did"
    return "does" if (person == Person.THIRD and number == Number.SINGULAR) else "do"


def modal_form(modal: str, tense: Tense = Tense.PRESENT) -> str:
    """A modal in the requested tense (can -> could)."""
    if tense == Tense.PAST:
        return MODAL_PAST.get(modal, modal)
    return modal


def _be_participle() -> str:
    return "been"


# ===========================================================================
# 5. Core conjugation
# ===========================================================================
def conjugate(verb: str, tense: Tense | str = Tense.PRESENT,
              person: Person | int = Person.THIRD,
              number: Number | str = Number.SINGULAR,
              aspect: Aspect | str = Aspect.SIMPLE,
              mood: Mood | str = Mood.INDICATIVE,
              voice: Voice | str = Voice.ACTIVE,
              modal: str | None = None) -> str:
    """Conjugate `verb` into the requested paradigm slot.

    Returns the complete verb phrase (auxiliaries + main verb), e.g.
        conjugate("walk", "past", 3, "sg", "perfect", voice="passive")
        -> "had been walked"
    """
    tense = Tense(tense) if not isinstance(tense, Tense) else tense
    person = Person(int(person)) if not isinstance(person, Person) else person
    number = Number(number) if not isinstance(number, Number) else number
    aspect = Aspect(aspect) if not isinstance(aspect, Aspect) else aspect
    mood = Mood(mood) if not isinstance(mood, Mood) else mood
    voice = Voice(voice) if not isinstance(voice, Voice) else voice

    base = verb.strip().lower()
    forms = all_forms(base)
    v3sg = forms["3sg"]
    past = forms["past"]
    pp = forms["past_participle"]
    ing = forms["present_participle"]

    # ---- mood overrides -------------------------------------------------
    if mood == Mood.IMPERATIVE:
        return ing if aspect == Aspect.CONTINUOUS else base

    if mood == Mood.SUBJUNCTIVE:
        # Present subjunctive: bare base for all persons ("that he go").
        if tense == Tense.PRESENT:
            return base
        # Past subjunctive of 'be' is 'were' for all persons.
        if base == "be":
            return "were"
        return past

    third_sg = (person == Person.THIRD and number == Number.SINGULAR)

    # ---- build the active verb phrase -----------------------------------
    def active_phrase() -> str:
        # modal + bare infinitive
        if modal:
            m = modal_form(modal, tense)
            if aspect == Aspect.SIMPLE:
                return f"{m} {base}"
            if aspect == Aspect.CONTINUOUS:
                return f"{m} be {ing}"
            if aspect == Aspect.PERFECT:
                return f"{m} have {pp}"
            if aspect == Aspect.PERFECT_CONTINUOUS:
                return f"{m} have been {ing}"
            return f"{m} {base}"

        if tense == Tense.FUTURE:
            if aspect == Aspect.SIMPLE:
                return f"will {base}"
            if aspect == Aspect.CONTINUOUS:
                return f"will be {ing}"
            if aspect == Aspect.PERFECT:
                return f"will have {pp}"
            return f"will have been {ing}"

        if tense == Tense.CONDITIONAL:
            if aspect == Aspect.SIMPLE:
                return f"would {base}"
            if aspect == Aspect.CONTINUOUS:
                return f"would be {ing}"
            if aspect == Aspect.PERFECT:
                return f"would have {pp}"
            return f"would have been {ing}"

        # present / past
        if aspect == Aspect.SIMPLE:
            if base == "be":
                return be_form(tense, person, number)
            return v3sg if (tense == Tense.PRESENT and third_sg) else \
                (past if tense == Tense.PAST else base)

        be = be_form(tense, person, number)
        have = have_form(tense, person, number)

        if aspect == Aspect.CONTINUOUS:
            return f"{be} {ing}"
        if aspect == Aspect.PERFECT:
            return f"{have} {pp}"
        # perfect continuous
        return f"{have} been {ing}"

    if voice == Voice.ACTIVE:
        return active_phrase()

    # ---- passive: be + past participle, in the matching tense/aspect ----
    if modal:
        m = modal_form(modal, tense)
        if aspect == Aspect.SIMPLE:
            return f"{m} be {pp}"
        if aspect == Aspect.CONTINUOUS:
            return f"{m} be being {pp}"
        if aspect == Aspect.PERFECT:
            return f"{m} have been {pp}"
        return f"{m} have been being {pp}"

    if tense == Tense.FUTURE:
        tail = {"simple": f"be {pp}", "continuous": f"be being {pp}",
                "perfect": f"have been {pp}",
                "perfect_continuous": f"have been being {pp}"}[aspect.value]
        return f"will {tail}"
    if tense == Tense.CONDITIONAL:
        tail = {"simple": f"be {pp}", "continuous": f"be being {pp}",
                "perfect": f"have been {pp}",
                "perfect_continuous": f"have been being {pp}"}[aspect.value]
        return f"would {tail}"

    be = be_form(tense, person, number)
    have = have_form(tense, person, number)

    if aspect == Aspect.SIMPLE:
        return f"{be} {pp}"
    if aspect == Aspect.CONTINUOUS:
        return f"{be} being {pp}"
    if aspect == Aspect.PERFECT:
        return f"{have} been {pp}"
    return f"{have} been being {pp}"


# Convenience wrappers for the three irregular auxiliaries.
def to_be(tense=Tense.PRESENT, person=Person.THIRD, number=Number.SINGULAR):
    return be_form(Tense(tense) if not isinstance(tense, Tense) else tense,
                   Person(int(person)), Number(number))


def to_have(tense=Tense.PRESENT, person=Person.THIRD, number=Number.SINGULAR):
    return have_form(Tense(tense) if not isinstance(tense, Tense) else tense,
                     Person(int(person)), Number(number))


def to_do(tense=Tense.PRESENT, person=Person.THIRD, number=Number.SINGULAR):
    return do_form(Tense(tense) if not isinstance(tense, Tense) else tense,
                   Person(int(person)), Number(number))


# ===========================================================================
# 6. Contractions
# ===========================================================================
CONTRACTIONS = {
    # be
    "i am": "i'm", "you are": "you're", "he is": "he's", "she is": "she's",
    "it is": "it's", "we are": "we're", "they are": "they're",
    "that is": "that's", "there is": "there's", "what is": "what's",
    "who is": "who's", "here is": "here's",
    "am not": "ain't", "is not": "isn't", "are not": "aren't",
    "was not": "wasn't", "were not": "weren't",
    # have
    "i have": "i've", "you have": "you've", "we have": "we've",
    "they have": "they've", "has not": "hasn't", "have not": "haven't",
    "had not": "hadn't",
    # will / would
    "i will": "i'll", "you will": "you'll", "he will": "he'll",
    "she will": "she'll", "it will": "it'll", "we will": "we'll",
    "they will": "they'll", "will not": "won't",
    "i would": "i'd", "you would": "you'd", "he would": "he'd",
    "she would": "she'd", "we would": "we'd", "they would": "they'd",
    "would not": "wouldn't",
    # do / modals
    "do not": "don't", "does not": "doesn't", "did not": "didn't",
    "can not": "can't", "cannot": "can't", "could not": "couldn't",
    "should not": "shouldn't", "must not": "mustn't",
    "may not": "mayn't", "might not": "mightn't", "shall not": "shan't",
    "let us": "let's",
}


def contract(text: str) -> str:
    """Apply contractions to a phrase (case-insensitive, longest match first).

    Capitalization is preserved: contractions that begin the string keep their
    initial capital ('I am' -> "I'm"), mid-sentence ones stay lowercase.
    """
    import re as _re
    out = text
    for full in sorted(CONTRACTIONS, key=len, reverse=True):
        short = CONTRACTIONS[full]
        def _repl(m, _short=short):
            matched = m.group(0)
            if matched[:1].isupper():
                return _short[:1].upper() + _short[1:]
            return _short
        out = _re.sub(_re.escape(full), _repl, out, flags=_re.IGNORECASE)
    return out


def expand_contractions(text: str) -> str:
    """Inverse of `contract`: expand contracted forms to full forms."""
    reverse = {}
    for full, short in CONTRACTIONS.items():
        reverse.setdefault(short, full)
    out = text
    import re as _re
    for short in sorted(reverse, key=len, reverse=True):
        out = _re.sub(_re.escape(short), reverse[short], out, flags=_re.IGNORECASE)
    return out


# ===========================================================================
# 7. Negation and questions
# ===========================================================================
def negate(verb_phrase: str, tense: Tense | str = Tense.PRESENT,
           person: Person | int = Person.THIRD,
           number: Number | str = Number.SINGULAR) -> str:
    """Negate a verb phrase, inserting the correct auxiliary.

    Uses *do-support* when the phrase has no auxiliary:
        "walks"        -> "does not walk"
        "walked"       -> "did not walk"
        "is walking"   -> "is not walking"   (no do-support needed)
    """
    tense = Tense(tense) if not isinstance(tense, Tense) else tense
    person = Person(int(person)) if not isinstance(person, Person) else person
    number = Number(number) if not isinstance(number, Number) else number

    words = verb_phrase.split()
    if not words:
        return verb_phrase

    # If the phrase already starts with an auxiliary/modal, just insert 'not'.
    first = words[0].lower()
    if first in AUX_BE + AUX_HAVE + AUX_DO + MODALS:
        # 'am not' -> keep as is; otherwise "aux not rest"
        return " ".join([words[0], "not"] + words[1:])

    # Otherwise: do-support. The main verb reverts to the base form.
    base = words[-1]
    if first.endswith("ly") or first in ("will", "would", "can", "could"):
        return " ".join([do_form(tense, person, number), "not"] + words)
    # Reduce an inflected main verb to its base before adding do-support.
    from .grammar import morphological_parse
    parses = morphological_parse(base)
    lemma = parses[0]["lemma"] if parses else base
    return f"{do_form(tense, person, number)} not {lemma}"


def question(verb_phrase: str, subject: str = "you",
             tense: Tense | str = Tense.PRESENT,
             person: Person | int = Person.THIRD,
             number: Number | str = Number.SINGULAR,
             wh: str | None = None) -> str:
    """Form a yes/no or wh-question by subject-auxiliary inversion.

    * with an auxiliary: invert ("is walking" -> "Is she walking?")
    * without: do-support ("walks" -> "Does she walk?")
    """
    tense = Tense(tense) if not isinstance(tense, Tense) else tense
    person = Person(int(person)) if not isinstance(person, Person) else person
    number = Number(number) if not isinstance(number, Number) else number

    words = verb_phrase.split()
    if not words:
        return verb_phrase

    first = words[0].lower()
    if first in AUX_BE + AUX_HAVE + AUX_DO + MODALS:
        core = " ".join([words[0]] + [subject] + words[1:])
    else:
        from .grammar import morphological_parse
        base = words[-1]
        parses = morphological_parse(base)
        lemma = parses[0]["lemma"] if parses else base
        core = f"{do_form(tense, person, number)} {subject} {lemma}"

    core = core[0].upper() + core[1:]
    if wh:
        core = f"{wh} {core[0].lower() + core[1:]}"
    return core + "?"


# ===========================================================================
# 8. Full paradigm tables
# ===========================================================================
@dataclass
class ConjugationTable:
    """Every form of a verb across the paradigm, keyed by label."""
    verb: str
    forms: dict = field(default_factory=dict)

    def get(self, *labels) -> str:
        """Fetch a nested entry, e.g. `get('present', '3sg')`."""
        node = self.forms
        for label in labels:
            node = node[label]
        return node

    def to_rows(self) -> list:
        """Flatten to `[(tense, aspect, person, number, form), ...]`."""
        rows = []
        for tense, aspects in self.forms.items():
            for aspect, slots in aspects.items():
                for person, numbers in slots.items():
                    for number, form in numbers.items():
                        rows.append((tense, aspect, person, number, form))
        return rows

    def display(self) -> str:
        """Human-readable conjugation table."""
        lines = [f"Conjugation of '{self.verb}'", "=" * 56]
        persons = [(1, "sg"), (2, "sg"), (3, "sg"), (1, "pl"), (2, "pl"), (3, "pl")]
        for tense, aspects in self.forms.items():
            for aspect, slots in aspects.items():
                lines.append(f"\n{tense.upper()} {aspect.upper().replace('_', ' ')}")
                for p, n in persons:
                    form = slots.get(p, {}).get(n)
                    if form:
                        subj = SUBJECT_PRONOUNS[(Person(p), Number(n))]
                        lines.append(f"    {subj:<8} {form}")
        return "\n".join(lines)

    def __repr__(self):
        return f"ConjugationTable('{self.verb}', {len(self.to_rows())} forms)"


def conjugate_table(verb: str, persons=(1, 2, 3),
                    numbers=("sg", "pl")) -> ConjugationTable:
    """Build the full active-voice paradigm for `verb`."""
    table = ConjugationTable(verb=verb)
    tenses = [Tense.PRESENT, Tense.PAST, Tense.FUTURE, Tense.CONDITIONAL]
    aspects = [Aspect.SIMPLE, Aspect.CONTINUOUS, Aspect.PERFECT,
               Aspect.PERFECT_CONTINUOUS]

    for tense in tenses:
        table.forms[tense.value] = {}
        for aspect in aspects:
            table.forms[tense.value][aspect.value] = {}
            for p in persons:
                table.forms[tense.value][aspect.value][p] = {}
                for n in numbers:
                    table.forms[tense.value][aspect.value][p][n] = conjugate(
                        verb, tense=tense, person=p, number=n, aspect=aspect)
    return table


def conjugate_all(verb: str) -> ConjugationTable:
    """Alias of `conjugate_table` (full paradigm)."""
    return conjugate_table(verb)


# ===========================================================================
# 9. ConjugatedVerb -- a verb bound to a subject and options
# ===========================================================================
class ConjugatedVerb:
    """A verb bound to a subject, exposing the full set of sentence builders.

    Example
    -------
    >>> v = ConjugatedVerb("walk", Subject.he(), tense="present")
    >>> v.form()                 # 'walks'
    >>> v.negative()             # 'does not walk'
    >>> v.question()             # 'Does he walk?'
    >>> v.negative(contract=True)# "doesn't walk"
    >>> v.passive()              # 'is walked'
    """

    def __init__(self, verb: str, subject: Subject | None = None, *,
                 tense: Tense | str = Tense.PRESENT,
                 aspect: Aspect | str = Aspect.SIMPLE,
                 mood: Mood | str = Mood.INDICATIVE,
                 voice: Voice | str = Voice.ACTIVE,
                 modal: str | None = None):
        self.verb = verb.strip().lower()
        self.subject = subject or Subject.they()
        self.tense = Tense(tense) if not isinstance(tense, Tense) else tense
        self.aspect = Aspect(aspect) if not isinstance(aspect, Aspect) else aspect
        self.mood = Mood(mood) if not isinstance(mood, Mood) else mood
        self.voice = Voice(voice) if not isinstance(voice, Voice) else voice
        self.modal = modal

    # -- core -------------------------------------------------------------
    def form(self) -> str:
        """The conjugated verb phrase (no subject)."""
        return conjugate(self.verb, tense=self.tense, person=self.subject.person,
                         number=self.subject.number, aspect=self.aspect,
                         mood=self.mood, voice=self.voice, modal=self.modal)

    def sentence(self, complement: str = "") -> str:
        """A complete sentence: subject + verb + optional complement."""
        text = f"{self.subject.word} {self.form()}"
        if complement:
            text += " " + complement
        return text[0].upper() + text[1:] + "."

    # -- transformations --------------------------------------------------
    def negative(self, contract: bool = False) -> str:
        """Negated verb phrase (optionally contracted)."""
        out = negate(self.form(), self.tense, self.subject.person,
                     self.subject.number)
        return contract_text(out) if contract else out

    def question(self, wh: str | None = None, contract: bool = False) -> str:
        """Yes/no or wh- question."""
        pronoun = self.subject.word if self.subject.noun else \
            SUBJECT_PRONOUNS[(self.subject.person, self.subject.number)]
        out = question(self.form(), pronoun, self.tense,
                       self.subject.person, self.subject.number, wh)
        return contract_text(out) if contract else out

    def passive_form(self) -> str:
        """The passive voice verb phrase for this slot."""
        return conjugate(self.verb, tense=self.tense, person=self.subject.person,
                         number=self.subject.number, aspect=self.aspect,
                         mood=self.mood, voice=Voice.PASSIVE, modal=self.modal)

    def passive(self, agent: str | None = None) -> str:
        """Passive sentence, optionally with a `by`-agent."""
        text = f"{self.subject.word} {self.passive_form()}"
        if agent:
            text += f" by {agent}"
        return text[0].upper() + text[1:] + "."

    def contracted(self) -> str:
        """The verb phrase with contractions applied."""
        return contract(self.form())

    def with_modal(self, modal: str) -> "ConjugatedVerb":
        """Return a copy using the given modal."""
        return ConjugatedVerb(self.verb, self.subject, tense=self.tense,
                              aspect=self.aspect, mood=self.mood,
                              voice=self.voice, modal=modal)

    def table(self) -> ConjugationTable:
        """Full paradigm table for this verb."""
        return conjugate_table(self.verb)

    def __repr__(self):
        return (f"ConjugatedVerb('{self.verb}', {self.subject.word}, "
                f"{self.tense.value}/{self.aspect.value})")


def contract_text(text: str) -> str:
    """Public alias for `contract` (applies contractions to a phrase)."""
    return contract(text)
