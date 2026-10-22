"""Collision tests for the Wikidata-generated entities (data/entities_generated.yaml).

Real registry (seed + generated). Surnames shared by several politicians, very common
surnames and surnames that are ordinary German words must never link on their own.
"""

import functools

import yaml

from analysis.linker import GENERATED_FILE, Registry


@functools.lru_cache(maxsize=1)
def reg():
    return Registry.from_yaml()


def ids(text, spans=(), ctx="", country="DE"):
    return [m.entity_id for m in reg().gazetteer(text, list(spans), ctx, country)]


def per(text, word):
    i = text.index(word)
    return [(i, i + len(word), "PER", 0.99)]


def test_generated_file_is_valid():
    ents = yaml.safe_load(GENERATED_FILE.read_text(encoding="utf-8"))["entities"]
    assert len(ents) > 1000
    assert len({e["id"] for e in ents}) == len(ents)
    assert len({e["wikidata"] for e in ents}) == len(ents)
    r = reg()
    assert r.entities["lars-klingbeil"].source == "seed"  # curated wins
    assert sum(1 for e in r.entities.values() if e.source == "generated") > 1000


def test_same_surname_politicians_need_full_name():
    # Joschka Fischer, Andrea Fischer, Simone Fischer ...: the surname alone is ambiguous
    assert ids("Fischer sagt das.", per("Fischer sagt das.", "Fischer")) == []
    assert ids("Joschka Fischer sagt das.") == ["joschka-fischer"]
    assert ids("Andrea Fischer sagt das.") == ["andrea-fischer"]
    assert ids("Köhler sagt das.", per("Köhler sagt das.", "Köhler")) == []
    assert ids("Horst Köhler war Bundespräsident.") == ["horst-koehler"]


def test_merz_is_the_chancellor_only_with_curated_alias():
    assert ids("Merz redet.") == ["friedrich-merz"]
    assert ids("Friedrich Merz redet.") == ["friedrich-merz"]
    # another Merz: the full name does not resolve to the chancellor
    assert ids("Hermann Merz redet.") == []


def test_common_surnames_never_link_alone():
    for sur in ["Wagner", "Müller", "Schmidt", "Klein"]:
        text = f"{sur} sagt das."
        assert ids(text) == [], sur
        assert ids(text, per(text, sur)) == [], sur
        assert ids(text, ctx="Bundestag Koalition SPD CDU") == [], sur
    assert ids("Das ist ein kleines Problem, klein halt.") == []
    assert ids("Richard Wagner komponierte.") == []
    assert ids("Carolin Wagner sagt das.") == ["carolin-wagner"]
    assert ids("Ulla Schmidt war Ministerin.") == ["ulla-schmidt"]
    assert ids("Gerd Müller sagt das.") == ["gerd-mueller"]
    assert ids("Felix Klein ist Antisemitismusbeauftragter.") == ["felix-klein"]


def test_generated_surnames_need_ner_not_context():
    # "Wiese" (Dirk Wiese) is a word; the politics context ("grüne") must not confirm it
    assert ids("Wir lieben die grüne Wiese.") == []
    assert ids("Dirk Wiese sagt das.") == ["dirk-wiese"]


def test_linke_and_gruene():
    assert ids("Die Linke will mehr.") == ["die-linke"]
    assert ids("Ich wähle Linke.") == ["die-linke"]
    assert ids("Das linke Lager ist leer.") == []
    assert ids("Die Grünen wollen das.") == ["gruene"]
    assert ids("Wir lieben die grüne Wiese.") == []
    assert ids("Die Linke in Österreich", country="AT") == []


def test_company_common_words_need_org_or_context():
    assert ids("Die Post kommt spät.") == []
    assert ids("Mit der Bahn ist es schlimm.") == []
    assert ids("Ich lese die Zeit.") == []
    assert ids("Die Welt ist schlecht.") == []
    assert ids("Die Deutsche Post erhöht das Porto.") == ["dhl-group"]
    assert ids("Bosch baut Stellen ab.") == ["robert-bosch"]
    assert ids("Aldi und Lidl senken Preise.") == ["aldi", "lidl"]
    assert ids("Lufthansa streikt.") == ["lufthansa"]


def test_universities_and_acronyms():
    assert ids("Die TU München forscht.") == ["technische-universitaet-muenchen"]
    assert ids("Die LMU ist gut.") == []  # acronym needs NER ORG
    t = "Die LMU ist gut."
    assert ids(t, [(4, 7, "ORG", 0.99)]) != []


def test_production_spot_check_errors():
    # errors found in the spot check of the first rollout; none of these may link again
    t = "Trump, Musk, Zuckerberg, Thiel, Bezos machen ja kein Geheimnis daraus."
    assert "barbara-thiel" not in ids(t, per(t, "Thiel"))
    t = "Und da sagte der gute alte Kant: Freiheit ist Verantwortung."
    assert ids(t, per(t, "Kant")) == []
    assert ids("Derzeit 9k Netto. Gut leben mit 3000 Euro Netto.") == []
    assert ids("China zeigt erste EUV Lithographie.", [(19, 22, "ORG", 0.9)]) == []
    t = "Der Westen hat gegenüber MBS geschwiegen."
    assert ids(t, [(25, 28, "ORG", 0.9)]) == []
    assert ids("Das Innenministerium spart.", country="AT") == []
    t = "Clemens Setz ist einer der Besten."
    assert "conrad-clemens" not in ids(t, per(t, "Clemens"))
    t = "Fallweise kaufe ich Moser-Roth vom Hofer."
    assert ids(t) == []


def test_second_spot_check_errors():
    t = "Klassiker von Norman Finkelstein und Ilan Pappé, Produkte des Konzerns."
    assert "norma" not in ids(t)
    assert "smart" not in ids("Solche smarten Produkte und Konzerne meiden.")
    assert "basler-ag" not in ids("Feuerzeuge an Basler Schulen.", country="CH")
    assert "netto-marken-discount" not in ids("Für die Mitarbeiter mehr Netto vom Brutto.")
    assert ids("Grünen und SPD") == ["gruene", "spd"]  # curated inflection still works
