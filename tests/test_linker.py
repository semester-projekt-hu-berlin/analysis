import yaml

from analysis.analyze import link_comment
from analysis.linker import SEED_FILE, Registry, strip_titles
from analysis.textutil import search_tokens, slugify, split_sentences


def reg():
    return Registry.from_yaml()


def ids(text, spans=(), ctx="", country="DE"):
    return [m.entity_id for m in reg().gazetteer(text, list(spans), ctx, country)]


def test_seed_file_is_valid():
    data = yaml.safe_load(SEED_FILE.read_text(encoding="utf-8"))
    ents = data["entities"]
    assert len(ents) > 250
    seen = set()
    for e in ents:
        assert e["id"] == e["id"].lower() and " " not in e["id"]
        assert e["id"] not in seen
        seen.add(e["id"])
        assert e["type"] in {"PER", "ORG", "PARTY"}
        assert str(e.get("wikidata", "")).startswith("Q"), e["id"]
    qids = [e["wikidata"] for e in ents]
    assert len(qids) == len(set(qids)), "duplicate Wikidata ids"


def test_merz_variants():
    for text in ["Friedrich Merz redet.", "Merz redet.", "Kanzler Merz redet.", "Merz' Rede", "merz hat recht"]:
        assert ids(text) == ["friedrich-merz"], text


def test_party_nicknames_and_compounds():
    assert ids("Die Sozis wieder.") == ["spd"]
    assert ids("Die AfD-Wähler sind laut.") == ["afd"]
    assert ids("CDU/CSU und SPD regieren.") == ["cdu-csu", "spd"]
    assert ids("Trumps Zölle") == ["donald-trump"]


def test_longest_match_wins():
    assert ids("Siemens Energy legt zu, Siemens auch.") == ["siemens-energy", "siemens"]
    assert ids("Bayer 04 Leverkusen ist Meister.") == ["bayer-04-leverkusen"]


def test_contextual_aliases_need_context():
    assert ids("Leverkusen hat das Spiel gewonnen.") == ["bayer-04-leverkusen"]
    assert ids("In Leverkusen regnet es.") == []
    # article title can provide the context
    assert ids("Leverkusen war stark.", ctx="Bundesliga: 3:1 gegen Mainz") == ["bayer-04-leverkusen"]
    assert ids("Bayern ist ein schönes Bundesland.") == []
    assert ids("Mit dem ICE nach Hamburg.") == []


def test_weak_aliases_need_ner_or_context():
    assert ids("Die Reiche Elite zahlt nichts.") == []
    assert ids("Ministerin Reiche will Gas.", spans=[(11, 17, "PER", 0.99)]) == ["katherina-reiche"]
    # an NER span of the wrong type does not confirm a person alias
    assert ids("Ministerin Reiche will Gas.", spans=[(11, 17, "LOC", 0.99)]) == []


def test_country_disambiguation():
    assert ids("Die Grünen sind schuld.", country="AT") == ["gruene-at"]
    assert ids("Die Grünen sind schuld.", country="DE") == ["gruene"]
    assert ids("Die Grünen sind schuld.", country=None) == ["gruene"]


def test_case_sensitive_aliases():
    assert ids("Das ist kurz gesagt falsch.") == []
    assert ids("Die UN tagen.") == ["uno"]
    assert ids("Was die un-sinnige Debatte soll") == []


def test_private_person_is_not_linked_without_wikidata():
    text = "Mein Nachbar Klaus Mösenbacher sagt das auch."
    sents = split_sentences(text)
    ner = [[(13, 30, "PER", 0.99)]]
    matches = link_comment(text, sents, ner, reg(), wikidata=None)
    assert matches == []


class StubWikidata:
    """Mimics WikidataLinker: only returns items that passed the public-figure filter."""

    lookups = 0

    def __init__(self, items):
        self.items = items

    def lookup(self, surface, tag):
        return self.items.get((surface, tag))


def test_wikidata_fallback_reuses_seed_qid_and_drops_private_names():
    r = reg()
    text = "Altkanzlerin Dorothea Kasner und Wirtschaftsminister Max Mustermann."
    a = text.index("Dorothea Kasner")
    b = text.index("Max Mustermann")
    ner = [[(a, a + 15, "PER", 0.99), (b, b + 14, "PER", 0.99)]]
    # only the public item is returned by the (filtered) Wikidata linker
    wd = StubWikidata({("Dorothea Kasner", "PER"): {"qid": "Q567", "name": "Angela Merkel", "type": "PER"}})
    matches = link_comment(text, split_sentences(text), ner, r, wikidata=wd)
    assert [m.entity_id for m in matches] == ["angela-merkel"]  # same QID -> seed entity
    assert "max-mustermann" not in r.entities


def test_wikidata_discovered_entity_added():
    r = reg()
    text = "Ich finde Hape Kerkeling super."
    a = text.index("Hape Kerkeling")
    wd = StubWikidata({("Hape Kerkeling", "PER"): {"qid": "Q57653", "name": "Hape Kerkeling", "type": "PER"}})
    matches = link_comment(text, split_sentences(text), [[(a, a + 14, "PER", 0.99)]], r, wikidata=wd)
    assert [m.entity_id for m in matches] == ["hape-kerkeling"]
    assert r.entities["hape-kerkeling"].source == "wikidata"


def test_strip_titles():
    assert strip_titles("Bundeskanzler Friedrich Merz") == "Friedrich Merz"
    assert strip_titles("Herr Dr. Söder") == "Söder"


def test_slug_and_tokens():
    assert slugify("Markus Söder") == "markus-soeder"
    assert slugify("1. FC Köln") == "1-fc-koeln"
    toks = search_tokens(["Markus Söder", "Söder"])
    assert "sö" in toks and "söder" in toks and "markus söder" in toks and "ma" in toks
    assert "s" not in toks


def test_sentence_split_caps_length():
    text = ("Das ist ein sehr langer Satz, " * 40).strip()
    spans = split_sentences(text)
    assert all(b - a <= 400 for a, b in spans)
    assert spans[0][0] == 0 and spans[-1][1] == len(text)


def test_football_disambiguation():
    assert ids("Der FCB hat das Spiel gewonnen.", country="CH") == ["fc-basel"]
    assert ids("Der FCB hat das Spiel gewonnen.", country="DE") == ["fc-bayern-muenchen"]
    assert ids("Leverkusen hat das Spiel gewonnen.") == ["bayer-04-leverkusen"]
    assert ids("Die Eintracht hat das Spiel gewonnen.") == ["eintracht-frankfurt"]
    assert ids("Die Eintracht hat das Spiel gewonnen, ganz Braunschweig feiert.")[0] == "eintracht-braunschweig"
    assert ids("RWE erhöht die Strompreise.") == ["rwe"]
    assert ids("RWE hat das Spiel gegen Aachen gewonnen.")[0] == "rot-weiss-essen"
    assert ids("Schalke und der HSV im Abstiegskampf.") == ["fc-schalke-04", "hamburger-sv"]
    assert ids("Die Fohlen haben gewonnen, Tor in der 90.") == ["borussia-moenchengladbach"]
    assert ids("Union hat das Spiel in der Liga gewonnen.") == ["1-fc-union-berlin"]
    assert ids("Die Union hat die Wahl verloren.") == ["cdu-csu"]
    assert ids("Das ÖFB-Team und die DFB-Elf.") == [
        "oesterreichische-fussballnationalmannschaft",
        "deutsche-fussballnationalmannschaft",
    ]
    assert ids("Die Nationalmannschaft hat das Spiel verloren.", country="AT") == [
        "oesterreichische-fussballnationalmannschaft"
    ]
    assert ids("Die Nati hat das Spiel gewonnen.", country="CH") == ["schweizer-fussballnationalmannschaft"]
    assert ids("Rapid hat das Derby gegen die Austria verloren.", country="AT") == ["sk-rapid-wien", "fk-austria-wien"]


def test_commons_thumb_url():
    from analysis.wikidata import commons_thumb

    u = commons_thumb("Example.jpg")
    assert u == "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Example.jpg/250px-Example.jpg"
    assert commons_thumb("Logo FC.svg").endswith("/250px-Logo_FC.svg.png")
    assert commons_thumb("Scan.tif") is None


def test_local_aliases_only_in_their_country():
    assert ids("Die Linken wollen das.", country="DE") == ["die-linke"]
    assert ids("Die Linken wollen das.", country="CH") == []
    assert ids("Die Sozis wieder.", country="AT") == ["spoe"]
    assert ids("Die Sozis wieder.", country="CH") == ["sp-schweiz"]
    assert ids("Wir brauchen eine starke Mitte.", country="CH") == []


def test_social_text_cleaning():
    from analysis.textutil import clean_text

    t = clean_text("@anna@norden.social #Merz muss weg! #SPDVersagt https://x.y/z")
    assert "@" not in t and "http" not in t
    assert ids(t) == ["friedrich-merz", "spd"]
    assert clean_text("#AfDVerbotJetzt") == "AfD Verbot Jetzt"
