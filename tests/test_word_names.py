"""Names that are ordinary words: brand names, newspapers and parties whose name is also a German
noun or phrase ("Sinn", "Die Welt", "Nordsee", "Bild", "Spiegel", "Die Mitte").

Real registry (seed + generated). The first block repeats the six errors of the production spot
check of the generated entities (October 2026, 44/50 correct); none of them may link again.
"""

import functools

import pytest

from analysis.analyze import link_comment
from analysis.linker import Registry, parse_alias
from analysis.textutil import split_sentences


@functools.lru_cache(maxsize=1)
def reg():
    return Registry.from_yaml()


def ids(text, spans=(), ctx="", country="DE"):
    return [m.entity_id for m in reg().gazetteer(text, list(spans), ctx, country)]


def span(text, word, tag="ORG", score=0.99):
    i = text.index(word)
    return [(i, i + len(word), tag, score)]


# ------------------------------------------------------------------ the six spot-check errors


def test_sinn_is_not_the_watch_brand():
    t = "Die Eingriffe in den Goldpreis ergeben wirtschaftswissenschaftlich allerdings nur begrenzt Sinn."
    assert "sinn" not in ids(t, ctx="Goldpreis Aktie Börse Gewinn Unternehmen")
    assert "sinn" not in ids(t, span(t, "Sinn"), ctx="Goldpreis Aktie")  # an ORG tag is not enough either
    assert ids("Das macht Sinn.") == [] and ids("Sinn hat recht.", ctx="Konzern Gewinn") == []
    assert ids("Meine Sinn 556 ist die beste Fliegeruhr.") == ["sinn"]
    assert ids("Hans-Werner Sinn warnt vor der Inflation.") == ["hans-werner-sinn"]


def test_die_welt_is_not_the_world():
    t = "Es ist schlimmer als je zuvor. Die Welt steuere auf 2,8 Grad Erwärmung bis zum Jahr 2100 zu."
    assert ids(t, ctx="Bericht Zeitung Artikel Journalist") == []
    assert ids("Die Welt ist kompliziert und chaotisch.", ctx="Zeitung") == []
    # the newspaper: its own spelling, its domain, its TV channel, cues
    assert "die-welt" in ids("Empfohlen von WELT News für Android")
    assert ids("Ich empfehle Ihnen den WELT-Artikel zur Asylpolitik.") == ["die-welt"]
    assert ids("Quelle: WELT") == ["die-welt"]
    assert ids("Das stand gestern auf welt.de.") == ["die-welt"]
    assert ids("Jan Fleischhauer hat bei WELT TV darauf hingewiesen.") == ["die-welt"]
    assert ids("Das stand in Die Welt, nicht in der taz.") == ["die-welt"]
    assert ids("Die Welt berichtet, dass die Regierung wackelt.") == ["die-welt"]
    # capitals in all-caps text are shouting, not the newspaper
    assert ids("DAS IST DIE SCHÖNE NEUE WELT DER GAS-KOHLE-KATI") == []
    assert ids("Österreich hat die strengsten Regeln der GANZEN WELT!") == []


def test_nordsee_is_not_the_sea():
    t = "Offshore-Windbranche vor Auftragsflaute: Tausende Jobs wackeln, Nordsee-Projekte stecken fest"
    assert "nordsee" not in ids(t, ctx="Unternehmen Konzern Mitarbeiter Stellen")
    t = "Die Erschließung eigener Gasvorkommen in der Nordsee und der Ostsee."
    assert "nordsee" not in ids(t, span(t, "Nordsee", "LOC"))
    assert ids("In der Nordsee-Filiale am Bahnhof gab es nur noch Backfisch.") == ["nordsee"]


def test_br_on_swiss_sources_is_the_bundesrat():
    t = "Die sollen besser mal in den Länder schauen wo Probleme sind. BR darf nicht einknicken."
    assert ids(t, span(t, "BR"), country="CH") == []
    assert ids("Aeschi passt nicht in den BR.", ctx="Sendung Fernsehen", country="CH") == []
    t = "Der BR hat darüber berichtet."
    assert ids(t, span(t, "BR"), country="DE") == ["bayerischer-rundfunk"]
    t = "The plane is currently in flight. [source pt-BR] [en] [de]"
    assert ids(t, span(t, "BR")) == []


def test_one_is_not_apple_one():
    t = "Haben seit gestern Apple One in der Familiengruppe."
    assert "one" not in ids(t, span(t, "One"))
    assert "one" not in ids(t, span(t, "Apple One"), ctx="Sender Serien Mediathek")
    for t in ["Plus One", "Ausser man kann sich eine eigene Air Force One leisten.", "Xbox One und PS4"]:
        assert "one" not in ids(t, span(t, "One")), t
    assert ids("ONE PIECE: Grand Gourmet erscheint für die Switch") == []
    assert "one" in ids("Am 31.12.2026 wird ONE eingestellt, ebenso ARD Alpha und Tagesschau24.")
    assert "one" in ids("ARD One zeigt heute Abend Serien.")


def test_the_focus_is_not_the_bike_brand():
    t = "„The Focus“, eine jüdische Gruppierung, hat enorme Summen aufgewendet."
    assert ids(t, span(t, "Focus")) == []
    t = "Mein Apple Focus läuft, ganz ohne E-Bike."  # right after another entity's name
    assert "focus" not in ids(t, span(t, "Focus"))
    # the magazine is not the bike brand (Focus the magazine is no entity yet)
    assert "focus" not in ids("Der Focus berichtet über Merz.", ctx="Zeitung Artikel Unternehmen")
    assert ids("Ich fahre ein Focus E-Bike, ein tolles Fahrrad.") == ["focus"]


# ------------------------------------------------------------------ other names that are words


def test_bild_is_the_newspaper_only_with_a_cue():
    for t in [
        "Das ergibt ein klares Bild.",
        "Der Artikel ist unvollständig und verzerrt damit das Bild: Deutschland zahlt nicht.",
        "Ein Bild sagt mehr als 1000 Worte.",
        "Man sieht es auf dem Bild, wie der Kanzler lacht.",
    ]:
        assert "bild" not in ids(t, span(t, "Bild"), ctx="Zeitung Artikel Schlagzeile"), t
    assert ids("Laut Bild ist das falsch.") == ["bild"]
    assert ids("Dobrindt sagte am Wochenende der »Bild«, er wolle das.") == ["alexander-dobrindt", "bild"]
    assert ids("Schlagzeile in der Bild: Ladenbesitzer haben Angst.") == ["bild"]
    assert ids("Das berichtet die „Bild“.") == ["bild"]
    assert ids("Merz hat die Desinformation von NIUS und Bild übernommen.") == ["friedrich-merz", "bild"]
    assert ids("Der Bild-Chefredakteur und die BILD-Zeitung") == ["bild", "bild"]


def test_spiegel_is_the_magazine_only_with_a_cue():
    for t in ["Söder, schau in den Spiegel!", "Ein Lächeln im Spiegel.", "einen kritischen Spiegel vorhalten"]:
        assert "der-spiegel" not in ids(t, span(t, "Spiegel"), ctx="Artikel Zeitung"), t
    t = "Oder wurde der Spiegel wirklich so schlecht verarbeitet?"
    assert ids(t) == []
    assert ids("Laut Spiegel liegt kein Favorit vorn.") == ["der-spiegel"]
    assert ids("Der Spiegel hat sich umgehört.") == ["der-spiegel"]
    assert ids("Ziel seien Kostensenkungen, berichten Spiegel und FAZ.") == [
        "der-spiegel",
        "frankfurter-allgemeine-zeitung",
    ]
    assert ids("ein Spiegel-Artikel zu den Ergebnissen") == ["der-spiegel"]


def test_zeit_in_capitals():
    assert ids("MAL WAS VERNÜNFTIGES IN DIESER IRREN ZEIT: es geht") == []
    assert ids("Vom Faktor Z für ZEIT gibt es noch keine Erklärung.") == []
    assert ids("Journoclown, der kleine Junge von der ZEIT.") == ["die-zeit"]
    assert ids("DIE ZEIT: ZDF-Politbarometer, Mehrheit will Neuwahlen") == ["die-zeit", "zdf"]
    assert ids("Die Zeit kann ich jetzt sinnvoller nutzen.", ctx="Artikel") == []


def test_ner_lookup_respects_the_spelling():
    r = reg()
    # an NER span "Zeit" / "Welt" / "Spiegel" is not the newspaper written "ZEIT" / "WELT" / "%Spiegel"
    assert r.lookup_surface("Zeit", "ORG", "DE") is None
    assert r.lookup_surface("Welt", "ORG", "DE") is None
    assert r.lookup_surface("Spiegel", "ORG", "DE") is None
    assert r.lookup_surface("Die Mitte", "ORG", "DE") is None  # local to Swiss sources
    # brand names vary in case, weak names do not
    assert r.lookup_surface("ARTE", "ORG", "DE") == "arte"
    assert r.lookup_surface("LIDL", "ORG", "DE") == "lidl"
    assert r.lookup_surface("Vp", "ORG", "AT") == "oevp"  # short acronyms in any case
    assert r.lookup_surface("reiche", "PER", "DE") is None
    assert r.lookup_surface("Reiche", "PER", "DE") == "katherina-reiche"
    t = "AfD-Leak: In einem Interview mit der Zeit bekennen sich zwei anonyme Täter."
    sents = split_sentences(t)
    i = t.index("Zeit")
    ner = [[(i, i + 4, "ORG", 0.99)]]
    assert "die-zeit" not in [m.entity_id for m in link_comment(t, sents, ner, r, wikidata=None)]


def test_die_mitte_only_on_swiss_sources():
    t = "Die Mitte darf nicht länger aus CDU/CSU, SPD und Grünen bestehen."
    assert "die-mitte" not in ids(t, ctx="Partei Wahl")
    assert ids("Immer schön die Mitte fixieren.") == []
    assert ids("Die Mitte stellt ab 2027 den Gemeindepräsidenten.", country="CH") == ["die-mitte"]
    assert "die-mitte" in ids("Die Linken und die Mitte wollen keinen starken SVP-Bundesrat.", country="CH")


def test_brand_names_that_are_words_or_names():
    t = "Ohne sich selbst etwas in die Taschen zu räumen, wie die Aktionäre."
    assert "taschen" not in ids(t, ctx="Konzern Gewinn Unternehmen")
    assert ids("Hr. Müller steht genau für ein Weiter so.", ctx="Unternehmen Konzern") == []
    assert ids("Bei Müller und Rossmann gibt es das günstiger, in jeder Filiale.") == ["mueller", "rossmann"]
    t = "Lanz und Michael Bröcker hetzen wieder."
    assert "heinrich-lanz-ag" not in ids(t)
    assert ids("Sky is the limit - pah!") == []
    assert ids("Book Sky illustration buchgestaltung") == []
    assert ids("Die Bundesliga läuft nur noch bei Sky.") == ["sky-deutschland"]
    t = "Tagesseminar in Fulda am 7.11.: Antifa, trotz alledem!"
    assert "fulda-reifen" not in ids(t, span(t, "Fulda", "LOC"))
    assert "saturn" not in ids("Die Bahnen von Jupiter, Saturn, Uranus und Neptun.")
    assert ids("Im Saturn und bei MediaMarkt gekauft.") == ["saturn", "mediamarkt"]
    assert ids("Ein kleiner Tante Emma Laden bekommt das nicht.", ctx="Unternehmen") == []
    assert ids("Alice Schwarzer stellt ihr Magazin „Emma“ ein.") == ["alice-schwarzer", "emma"]


def test_bosch_keeps_its_short_name():
    assert ids("Bosch baut Stellen ab.") == ["robert-bosch"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Merz", ("Merz", "plain")),
        ("=SAP", ("SAP", "cs")),
        ("~Reiche", ("Reiche", "weak")),
        ("@Leverkusen", ("Leverkusen", "ctx")),
        ("%Bild", ("Bild", "cue")),
        ("^Linke", ("Linke", "local")),
        ("^~BR", ("BR", "local-weak")),
        ("^@Mitte", ("Mitte", "local-ctx")),
    ],
)
def test_alias_markers(raw, expected):
    assert parse_alias(raw) == expected
