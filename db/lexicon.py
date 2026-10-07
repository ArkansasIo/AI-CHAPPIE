"""
db.lexicon -- seed data for the English dictionary database.

Contains a curated lexicon of common English words with:

* part of speech + definition (some words have two senses)
* a usage example
* synonyms and antonyms
* etymology (origin language + root) for many entries
* rough frequency ranks (1 = most common)

Call `seed(dictionary)` to populate a `db.dictionary.Dictionary`, or
`build(path)` to create a fresh, fully-populated database file.

The data is intentionally compact but realistic; it is enough to exercise every
query in `db.dictionary` (senses, anagrams, wildcards, rhymes, POS filters,
relations, etc.).
"""

from __future__ import annotations

# Each entry: (word, syllables, freq_rank, origin_language, root, ipa, respelling,
#              [(pos, definition, register, domain, example), ...],
#              [synonyms], [antonyms], [(relation, target), ...])
#
# Synonyms/antonyms may name words not present as headwords; the schema stores
# them as plain text edges, which is realistic for a lexical database.

ENTRIES = [
    ("time", 1, 1, "Old English", "tīma", "/taɪm/", "tym",
     [("noun", "The indefinite continued progress of existence and events.",
       "standard", None, "Time heals all wounds."),
      ("verb", "To arrange or measure when something happens.",
       "standard", None, "She timed the race precisely.")],
     ["period", "duration", "epoch", "moment"], [],
     [("related", "temporal", "adjective form")]),

    ("person", 2, 2, "Latin", "persona", "/ˈpɜːrsən/", "pur-suhn",
     [("noun", "A human being regarded as an individual.",
       "standard", None, "She is a kind person.")],
     ["individual", "human", "being"], [],
     [("hypernym", "organism", "a person is a kind of organism")]),

    ("year", 1, 3, "Old English", "gēar", "/jɪər/", "yeer",
     [("noun", "The time taken by the earth to orbit the sun.",
       "standard", None, "We met last year.")],
     ["twelvemonth"], [], []),

    ("way", 1, 4, "Old English", "weg", "/weɪ/", "way",
     [("noun", "A method, style, or manner of doing something.",
       "standard", None, "That is the right way to do it."),
      ("noun", "A road or path leading to a place.",
       "standard", None, "The way home is long.")],
     ["method", "manner", "route", "path"], [], []),

    ("day", 1, 5, "Old English", "dæg", "/deɪ/", "day",
     [("noun", "A period of twenty-four hours as a unit of time.",
       "standard", None, "It rained all day.")],
     ["date", "daytime"], ["night"], []),

    ("thing", 1, 6, "Old English", "þing", "/θɪŋ/", "thing",
     [("noun", "An object that need not be named specifically.",
       "standard", None, "Put that thing over there.")],
     ["object", "item", "entity"], [], []),

    ("man", 1, 7, "Old English", "mann", "/mæn/", "man",
     [("noun", "An adult male human being.",
       "standard", None, "The man walked his dog."),
      ("verb", "To take charge of a place or machine.",
       "standard", None, "Two sailors manned the boat.")],
     ["gentleman", "fellow"], ["woman"], []),

    ("world", 1, 8, "Old English", "weorold", "/wɜːrld/", "world",
     [("noun", "The earth, together with all of its countries and peoples.",
       "standard", None, "She travelled the world.")],
     ["earth", "globe", "planet"], [], []),

    ("life", 1, 9, "Old English", "līf", "/laɪf/", "lyf",
     [("noun", "The condition that distinguishes animals and plants from matter.",
       "standard", None, "Water is essential to life.")],
     ["existence", "being", "vitality"], ["death"], []),

    ("hand", 1, 10, "Old English", "hand", "/hænd/", "hand",
     [("noun", "The end part of the arm beyond the wrist.",
       "standard", None, "She held the cup in her hand."),
      ("verb", "To give or pass something to someone.",
       "standard", None, "He handed me the letter.")],
     ["palm", "give"], [], [("participle", "handed", None)]),

    # --- words chosen for rich cross-links and anagram/wildcard demos ---
    ("bank", 1, 900, "Italian", "banca", "/bæŋk/", "bangk",
     [("noun", "A financial institution that accepts deposits and makes loans.",
       "standard", "finance", "She deposited the cheque at the bank."),
      ("noun", "The land alongside a river or lake.",
       "standard", "geography", "We sat on the river bank.")],
     ["shore", "coast", "institution"], [],
     [("related", "banking", "trade"), ("hyponym", "savings bank", None)]),

    ("banker", 2, 4000, "French", "banquier", "/ˈbæŋkər/", "bank-er",
     [("noun", "A person who manages or owns a bank.",
       "standard", "finance", "The banker approved the loan.")],
     ["financier"], [], [("related", "bank", "derived from 'bank'")]),

    ("light", 1, 300, "Old English", "lēoht", "/laɪt/", "lyt",
     [("noun", "The natural agent that makes things visible.",
       "standard", "physics", "Light travels at a constant speed."),
      ("adjective", "Of little weight; not heavy.",
       "standard", None, "The bag was light enough to carry.")],
     ["illumination", "bright", "weightless"], ["dark", "heavy"], []),

    ("dark", 1, 500, "Old English", "deorc", "/dɑːrk/", "dark",
     [("adjective", "With little or no light.",
       "standard", None, "The room was dark."),
      ("noun", "The absence of light.",
       "standard", None, "Creatures that hunt in the dark.")],
     ["dim", "gloomy", "shadowy"], ["light", "bright"], []),

    ("happy", 2, 600, "Old Norse", "happ", "/ˈhæpi/", "hap-ee",
     [("adjective", "Feeling or showing pleasure and contentment.",
       "standard", None, "She was happy to see him.")],
     ["glad", "joyful", "cheerful", "content"], ["sad", "unhappy", "miserable"], []),

    ("sad", 1, 800, "Old English", "sæd", "/sæd/", "sad",
     [("adjective", "Feeling or showing sorrow; unhappy.",
       "standard", None, "He felt sad after the news.")],
     ["unhappy", "sorrowful", "downcast"], ["happy", "glad", "cheerful"], []),

    ("fast", 1, 700, "Old English", "fæst", "/fɑːst/", "fahst",
     [("adjective", "Moving or capable of moving at high speed.",
       "standard", None, "The car was very fast."),
      ("adverb", "At high speed.",
       "standard", None, "She ran fast to catch the bus.")],
     ["quick", "swift", "rapid", "speedy"], ["slow"], []),

    ("slow", 1, 1200, "Old English", "slāw", "/sloʊ/", "sloh",
     [("adjective", "Moving or operating at a low speed.",
       "standard", None, "Traffic was slow this morning.")],
     ["sluggish", "leisurely"], ["fast", "quick", "rapid"], []),

    ("big", 1, 200, "Old Norse", "bugge", "/bɪɡ/", "big",
     [("adjective", "Of considerable size or extent.",
       "standard", None, "They live in a big house.")],
     ["large", "huge", "enormous", "vast"], ["small", "little", "tiny"], []),

    ("small", 1, 400, "Old English", "smæl", "/smɔːl/", "smawl",
     [("adjective", "Of a size that is less than normal.",
       "standard", None, "A small bird landed nearby.")],
     ["little", "tiny", "minute"], ["big", "large", "huge"], []),

    ("hot", 1, 1100, "Old English", "hāt", "/hɒt/", "hot",
     [("adjective", "Having a high temperature.",
       "standard", None, "The soup was too hot to eat.")],
     ["warm", "scorching", "boiling"], ["cold", "cool", "freezing"], []),

    ("cold", 1, 1000, "Old English", "cald", "/koʊld/", "kohld",
     [("adjective", "Of or at a low temperature.",
       "standard", None, "It was a cold winter morning."),
      ("noun", "A common viral infection of the nose and throat.",
       "standard", "medicine", "She caught a cold last week.")],
     ["chilly", "freezing", "frosty"], ["hot", "warm"], []),

    ("water", 2, 350, "Old English", "wæter", "/ˈwɔːtər/", "waw-ter",
     [("noun", "A transparent, odorless liquid essential for life.",
       "standard", "chemistry", "Drink plenty of water.")],
     ["liquid", "H2O"], [], [("hypernym", "liquid", None),
                             ("related", "watery", "adjective form")]),

    ("fire", 1, 650, "Old English", "fȳr", "/ˈfaɪər/", "fyr",
     [("noun", "Combustion producing heat, light and flame.",
       "standard", None, "The fire warmed the room."),
      ("verb", "To discharge a gun; to dismiss from a job.",
       "standard", None, "They had to fire the manager.")],
     ["flame", "blaze", "dismiss"], [], [("related", "fiery", "adjective form")]),

    ("earth", 1, 750, "Old English", "eorþe", "/ɜːrθ/", "urth",
     [("noun", "The planet on which we live.",
       "standard", "astronomy", "Earth is the third planet."),
      ("noun", "Soil or ground.",
       "standard", None, "He dug into the earth.")],
     ["world", "globe", "soil", "ground"], [], []),

    ("wind", 1, 850, "Old English", "wind", "/wɪnd/", "wind",
     [("noun", "The perceptible natural movement of the air.",
       "standard", "meteorology", "The wind blew all night.")],
     ["breeze", "gust", "draft"], [], []),

    ("rain", 1, 950, "Old English", "regn", "/reɪn/", "rayn",
     [("noun", "Water falling in drops from the clouds.",
       "standard", "meteorology", "The rain soaked the fields."),
      ("verb", "For rain to fall.",
       "standard", None, "It rained for three days.")],
     ["precipitation", "shower", "drizzle"], ["drought"], []),

    ("sun", 1, 780, "Old English", "sunne", "/sʌn/", "sun",
     [("noun", "The star around which the earth orbits.",
       "standard", "astronomy", "The sun rises in the east.")],
     ["star", "sunshine"], [], [("hypernym", "star", None)]),

    ("moon", 1, 1400, "Old English", "mōna", "/muːn/", "moon",
     [("noun", "The natural satellite of the earth.",
       "standard", "astronomy", "The moon was full that night.")],
     ["satellite", "luna"], [], [("hypernym", "satellite", None)]),

    ("star", 1, 900, "Old English", "steorra", "/stɑːr/", "stahr",
     [("noun", "A luminous point in the night sky; a massive celestial body.",
       "standard", "astronomy", "The stars were bright."),
      ("noun", "A famous performer.",
       "informal", "entertainment", "She became a movie star.")],
     ["sun", "celebrity"], [], []),

    ("word", 1, 250, "Old English", "word", "/wɜːrd/", "wurd",
     [("noun", "A single distinct meaningful element of speech or writing.",
       "standard", "linguistics", "Every word counts.")],
     ["term", "expression", "vocable"], [], []),

    ("book", 1, 320, "Old English", "bōc", "/bʊk/", "buk",
     [("noun", "A written or printed work consisting of pages.",
       "standard", None, "She read the book in one sitting."),
      ("verb", "To reserve accommodation or a ticket.",
       "standard", None, "They booked a hotel room.")],
     ["volume", "text", "reserve"], [], [("plural", "books", None)]),

    ("computer", 3, 1500, "Latin", "computare", "/kəmˈpjuːtər/", "kum-pyoo-ter",
     [("noun", "An electronic device for storing and processing data.",
       "standard", "computing", "The computer crashed again.")],
     ["machine", "PC"], [], [("related", "compute", "derived from 'compute'"),
                             ("hypernym", "machine", None)]),

    ("data", 2, 1600, "Latin", "datum", "/ˈdeɪtə/", "day-tuh",
     [("noun", "Facts and statistics collected for reference or analysis.",
       "standard", "computing", "The data supports the theory.")],
     ["information", "facts", "figures"], [], [("related", "datum", "singular form")]),

    ("logic", 2, 2100, "Greek", "logikē", "/ˈlɒdʒɪk/", "loj-ik",
     [("noun", "Reasoning conducted according to strict principles of validity.",
       "standard", "philosophy", "The argument rests on sound logic.")],
     ["reasoning", "rationale"], [], [("related", "logical", "adjective form")]),

    ("gate", 1, 2600, "Old Norse", "gat", "/ɡeɪt/", "gayt",
     [("noun", "A hinged barrier used to close an opening in a wall or fence.",
       "standard", None, "Please close the gate behind you."),
      ("noun", "A device that outputs a signal based on input conditions.",
       "technical", "computing", "A NAND gate is universal.")],
     ["door", "barrier"], [], []),

    ("neuron", 2, 6000, "Greek", "neuron", "/ˈnjʊərɒn/", "nyoor-on",
     [("noun", "A cell that carries electrical impulses in the nervous system.",
       "technical", "biology", "Each neuron connects to thousands of others.")],
     ["nerve cell"], [], [("related", "neural", "adjective form")]),

    ("network", 2, 2200, "Old English", "net + weorc", "/ˈnɛtwɜːrk/", "net-wurk",
     [("noun", "A group of interconnected people or things.",
       "standard", None, "A neural network learns from data."),
      ("noun", "A system of computers linked together.",
       "technical", "computing", "The network was down for an hour.")],
     ["web", "grid", "system"], [], []),

    ("learn", 1, 1800, "Old English", "leornian", "/lɜːrn/", "lurn",
     [("verb", "To gain knowledge or skill by study or experience.",
       "standard", None, "Children learn quickly.")],
     ["study", "master", "acquire"], ["forget", "unlearn"], []),

    ("think", 1, 240, "Old English", "þencan", "/θɪŋk/", "think",
     [("verb", "To use the mind to consider ideas.",
       "standard", None, "Think carefully before you answer.")],
     ["consider", "ponder", "reason"], [], []),

    ("quick", 1, 1050, "Old English", "cwic", "/kwɪk/", "kwik",
     [("adjective", "Moving fast or doing something in a short time.",
       "standard", None, "She gave a quick reply.")],
     ["fast", "rapid", "swift"], ["slow"], []),

    ("soft", 1, 1300, "Old English", "sōfte", "/sɒft/", "soft",
     [("adjective", "Easy to mould, cut, or compress; not hard.",
       "standard", None, "The pillow was soft.")],
     ["gentle", "supple", "yielding"], ["hard", "firm"], []),

    ("hard", 1, 550, "Old English", "heard", "/hɑːrd/", "hahrd",
     [("adjective", "Solid and firm; resistant to pressure.",
       "standard", None, "The ground was hard and dry."),
      ("adjective", "Requiring great effort; difficult.",
       "standard", None, "It was a hard question.")],
     ["solid", "firm", "difficult"], ["soft", "easy"], []),

    ("run", 1, 450, "Old English", "rinnan", "/rʌn/", "run",
     [("verb", "To move at a speed faster than a walk.",
       "standard", None, "She runs every morning."),
      ("verb", "To operate or manage something.",
       "standard", None, "He runs a small company.")],
     ["sprint", "jog", "operate"], ["walk", "stop"], []),

    ("walk", 1, 700, "Old English", "wealcan", "/wɔːk/", "wawk",
     [("verb", "To move at a regular pace by lifting and setting down each foot.",
       "standard", None, "They walked along the beach.")],
     ["stroll", "amble", "stride"], ["run"], []),

    ("silent", 2, 3200, "Latin", "silens", "/ˈsaɪlənt/", "sy-luhnt",
     [("adjective", "Making no sound; completely quiet.",
       "standard", None, "The room fell silent.")],
     ["quiet", "mute", "noiseless", "soundless"], ["loud", "noisy"], []),

    ("listen", 2, 1900, "Old English", "hlysnan", "/ˈlɪsən/", "lis-uhn",
     [("verb", "To give one's attention to a sound.",
       "standard", None, "Listen to the music.")],
     ["hear", "attend", "heed"], ["ignore"], []),

    ("enlist", 2, 8000, "French", "enlister", "/ɪnˈlɪst/", "in-list",
     [("verb", "To enrol or enrol someone in the armed forces or a cause.",
       "standard", None, "He decided to enlist in the army.")],
     ["recruit", "join", "sign up"], ["withdraw"], []),

    ("tinsel", 2, 15000, "French", "estincelle", "/ˈtɪnsəl/", "tin-suhl",
     [("noun", "A strip of shiny material used as a decoration.",
       "standard", None, "The tree was covered in tinsel.")],
     ["glitter", "spangle"], [], []),

    ("cat", 1, 1600, "Old English", "catt", "/kæt/", "kat",
     [("noun", "A small domesticated carnivorous mammal.",
       "standard", None, "The cat slept on the windowsill.")],
     ["feline", "kitty", "puss"], [], [("hypernym", "animal", None),
                                         ("hyponym", "kitten", None)]),

    ("cot", 1, 12000, "Hindi", "khāṭ", "/kɒt/", "kot",
     [("noun", "A small bed with high sides for a baby.",
       "standard", None, "The baby slept in the cot.")],
     ["crib", "bassinet"], [], []),

    ("cut", 1, 1000, "Middle English", "cutten", "/kʌt/", "kut",
     [("verb", "To make an opening or divide with a sharp tool.",
       "standard", None, "She cut the paper in half.")],
     ["slice", "sever", "trim"], ["join", "mend"], []),

    ("dog", 1, 1700, "Old English", "docga", "/dɒɡ/", "dawg",
     [("noun", "A domesticated carnivorous mammal kept as a pet.",
       "standard", None, "The dog barked at the postman.")],
     ["hound", "canine", "pooch"], [], [("hypernym", "animal", None),
                                       ("hyponym", "puppy", None)]),

    ("god", 1, 500, "Old English", "god", "/ɡɒd/", "god",
     [("noun", "A superhuman being regarded as having power over nature.",
       "standard", "religion", "They prayed to their god.")],
     ["deity", "creator"], [], []),

    ("water", 2, 350, None, None, None, None, [], [], [], []),  # placeholder guard
]

# Drop the placeholder guard row (kept simple: filter rows with a real senses list)
ENTRIES = [e for e in ENTRIES if e[7]]


def _dedupe(entries):
    """Keep the first occurrence of each headword."""
    seen, out = set(), []
    for e in entries:
        if e[0].lower() in seen:
            continue
        seen.add(e[0].lower())
        out.append(e)
    return out


ENTRIES = _dedupe(ENTRIES)


def seed(dictionary) -> int:
    """Populate a `Dictionary` with every entry. Returns the word count added."""
    n = 0
    for (word, syl, rank, origin, root, ipa, respell, senses,
         syns, ants, rels) in ENTRIES:
        dictionary.add_word(word, syllables=syl, frequency_rank=rank)
        for i, (pos, definition, register, domain, example) in enumerate(senses, 1):
            dictionary.add_sense(word, pos, definition, sense_no=i,
                                 register=register, domain=domain)
            if example:
                dictionary.add_example(word, example, sense_no=i)
        for s in syns:
            dictionary.add_synonym(word, s)
        for a in ants:
            dictionary.add_antonym(word, a)
        for rel, target, note in rels:
            dictionary.add_relation(word, rel, target, note)
        if origin or root:
            dictionary.add_etymology(word, origin, root)
        if ipa or respell:
            dictionary.add_pronunciation(word, ipa, respell)
        # crude frequency proxy: per-million from rank (Zipf-ish)
        if rank:
            dictionary.add_frequency(word, max(1, 100_000_000 // rank),
                                     round(1000.0 / rank, 4))
        n += 1
    return n


def build(path: str = None, force: bool = False):
    """Create (or open) and seed a dictionary database; return the Dictionary."""
    from db.dictionary import Dictionary, DEFAULT_DICT_PATH
    path = path or DEFAULT_DICT_PATH
    if force and path and not path.startswith(":memory:"):
        import os
        if os.path.exists(path):
            os.remove(path)
    d = Dictionary(path)
    if d.count("words") == 0:
        seed(d)
    return d


if __name__ == "__main__":
    d = build(force=True)
    print(d.report())
    d.close()
