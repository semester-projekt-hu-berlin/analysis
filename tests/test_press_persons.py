"""People in the press (scripts/generate_entities.py, PRESS_PERSONS): investors and "Die Höhle der
Löwen", TV and music stars, YouTube/Twitch creators. Real registry (seed + generated).

Their names are often surnames shared with others, words or first names, so the tests check both
sides: the person links with the evidence their alias mode needs, and the ordinary word or the
namesake does not.
"""

import functools

import yaml

from analysis.linker import GENERATED_FILE, Registry


@functools.lru_cache(maxsize=1)
def reg():
    return Registry.from_yaml()


def ids(text, spans=(), ctx="", country="DE"):
    return [m.entity_id for m in reg().gazetteer(text, list(spans), ctx, country)]


def per(text, word, tag="PER"):
    i = text.index(word)
    return [(i, i + len(word), tag, 0.99)]


def test_press_persons_are_generated_with_qid_type_country_and_context():
    ents = {e["wikidata"]: e for e in yaml.safe_load(GENERATED_FILE.read_text(encoding="utf-8"))["entities"]}
    expect = {
        "Q15465039": ("Frank Thelen", "investors"),
        "Q1045616": ("Carsten Maschmeyer", "investors"),
        "Q23062132": ("Ralf Dümmel", "investors"),
        "Q66424819": ("Nils Glagau", "investors"),
        "Q124385918": ("Tillman Schulz", "investors"),
        "Q117856227": ("Janna Ensthaler", "investors"),
        "Q77807": ("Barbara Schöneberger", "entertainment"),
        "Q1702610": ("Joko Winterscheidt", "entertainment"),
        "Q24008612": ("Capital Bra", "entertainment"),
        "Q51750015": ("Rezo", "creators"),
        "Q58603743": ("Evelyn Palla", "transit"),
    }
    for qid, (name, ctx) in expect.items():
        e = ents[qid]
        assert (e["name"], e["type"], e["country"], e.get("context")) == (name, "PER", "DE", ctx), qid
        assert e["description"]


def test_full_names_link():
    for text, eid in [
        ("Frank Thelen hat wieder investiert.", "frank-thelen"),
        ("Carsten Maschmeyer und Ralf Dümmel streiten.", "carsten-maschmeyer"),
        ("Judith Williams ist raus.", "judith-williams"),
        ("Joko Winterscheidt moderiert.", "joko-winterscheidt"),
        ("Barbara Schöneberger war gut.", "barbara-schoeneberger"),
        ("Atze Schröder tritt auf.", "hubertus-albers"),
        ("Amira Aly hat sich getrennt.", "amira-aly"),
        ("Pamela Reif verkauft Müsli.", "pamela-reif"),
        ("Bianca Claßen hört auf.", "bianca-heinicke"),
        ("MontanaBlack hat gestreamt.", "marcel-eris"),
        ("Evelyn Palla will sparen.", "evelyn-palla"),
    ]:
        assert ids(text)[:1] == [eid], text


def test_distinctive_surnames_need_ner_or_context():
    assert ids("Maschmeyer ist eine Zumutung.") == []
    t = "Maschmeyer ist eine Zumutung."
    assert ids(t, per(t, "Maschmeyer")) == ["carsten-maschmeyer"]
    assert ids("Maschmeyer hat in das Startup investiert.") == ["carsten-maschmeyer"]
    assert ids("Thelen und der nächste Deal mit einem Startup.") == ["frank-thelen"]
    assert ids("Thelen sagt das.") == []
    assert ids("Böhmermann hat in seiner Sendung übertrieben.") == ["jan-boehmermann"]
    assert ids("Palla muss die Bahn endlich pünktlich machen.") == ["evelyn-palla"]
    assert ids("Die Bahn ist palla.") == []  # case-sensitive


def test_nicknames_only_with_context():
    assert ids("Maschi macht das schon.") == []
    assert ids("Maschi hat bei den Löwen wieder zugeschlagen.") == ["carsten-maschmeyer"]
    assert ids("Der Poptitan sitzt wieder in der DSDS-Jury.", ctx="DSDS: neue Staffel") == ["dieter-bohlen"]
    assert ids("Ein Titan der Branche.") == []


def test_surnames_shared_or_words_stay_strict():
    # Raab: NER-only (the river, Julius Raab); never as part of another person's name
    assert ids("Die Raab fließt bei Győr in die Donau.") == []
    t = "Raab ist zurück."
    assert ids(t, per(t, "Raab")) == ["stefan-raab"]
    t = "Frank Raab ist zurück."
    assert ids(t, [(0, 10, "PER", 0.99)]) == []
    # Gottschalk: Kay Gottschalk (AfD) shares the surname, so only with show business words
    t = "Gottschalk hat in der Debatte gesprochen."
    assert ids(t) == [] and ids(t, per(t, "Gottschalk")) == []
    assert ids("Gottschalk war bei Wetten dass der beste Moderator.") == ["thomas-gottschalk"]
    # Bohlen, Klum, Jauch: NER person tag
    t = "Bohlen und Klum sitzen in der Jury."
    assert ids(t, per(t, "Bohlen") + per(t, "Klum")) == ["dieter-bohlen", "heidi-klum"]
    # common surnames and words never alone
    for word in ["Fischer", "Müller", "Becker", "Schweiger", "Williams", "Schulz", "Lichter", "Reif"]:
        t = f"{word} war gestern in der Show."
        assert ids(t) == [], word
        assert ids(t, per(t, word)) == [], word
    assert ids("Die Lichter gehen aus.") == []


def test_word_nicknames_from_wikidata_are_gone():
    assert ids("Die Katze liegt auf dem Sofa.") == []
    assert ids("Ich hab die Katze gefüttert.") == []
    assert ids("Der Boss hat entschieden.") == []
    assert ids("Er ist der Boss im Haus.") == []
    assert ids("Die Axt im Haus erspart den Zimmermann.") == []


def test_first_names_and_brand_words_never_alone():
    assert ids("Bibi hat ein neues Video hochgeladen.") == []
    assert ids("Laut Capital ist die Aktie überbewertet.") == []
    assert ids("Capital Bra hat ein neues Album.") == ["capital-bra"]
    assert ids("Der Apache ist ein Kampfhubschrauber, Apache ein Webserver.") == []
    assert ids("Apache 207 füllt Stadien.") == ["apache-207"]
    assert ids("Joko macht Quatsch.") == []
    assert ids("Atze, komm mal her.") == []


def test_contextual_one_word_names():
    # Elton: only with show business words, never inside "Elton John"
    assert ids("Elton hat heute Geburtstag.") == []
    assert ids("Elton moderiert die Show.") == ["elton"]
    t = "Elton John spielt ein Konzert."
    assert ids(t, [(0, 10, "PER", 0.99)]) == []
    # Luciano: only with music words, never "Luciano Pavarotti"
    assert ids("Luciano ist mein Nachbar.") == []
    assert ids("Luciano hat ein neues Album.") == ["luciano"]
    t = "Luciano Pavarotti sang ein Konzert."
    assert ids(t, [(0, 17, "PER", 0.99)]) == []
    # Unge: creator words
    assert ids("Unge hat gestreamt.") == ["unge"]
    assert ids("Unge Leute, unge Ideen.") == []


def test_one_word_artist_and_creator_names():
    assert ids("Kollegah hat ein neues Album.") == ["kollegah"]
    assert ids("Knossi und Gronkh streamen zusammen.") == ["jens-knossalla", "gronkh"]
    assert ids("Rezo hat ein Video über die CDU gemacht.") == ["rezo", "cdu"]
    assert ids("Das ist ein rezo.") == []
    t = "Bushido steht vor Gericht."
    assert ids(t, per(t, "Bushido")) == ["bushido"]
    assert ids("Bushido rappt auf dem neuen Album.") == ["bushido"]
    assert ids("Der Bushido ist der Ehrenkodex der Samurai.") == []
    # one-word artist names now link with an NER person tag ("Sido"), first names and words do not
    t = "Sido war in der Talkshow."
    assert ids(t, per(t, "Sido")) == ["sido"]
    for word in ["Nicki", "Nana", "Fancy"]:
        t = f"{word} war in der Talkshow."
        assert ids(t, per(t, word)) == [], word


def test_namesakes_keep_or_lose_the_bare_surname():
    # Claus Kleber keeps "Kleber" (NER-only) although the basketball player Maxi Kleber came in
    t = "Kleber hat das wieder moderiert."
    assert ids(t, per(t, "Kleber")) == ["claus-kleber"]
    # the company Würth keeps its name; Reinhold Würth only with his full name
    assert ids("Würth baut Stellen ab.") == ["wuerth"]
    assert ids("Reinhold Würth wird 90.") == ["reinhold-wuerth"]
    # Oliver Pocher keeps "Pocher" (NER-only); Amira links by full name only
    t = "Pocher hat wieder ausgeteilt."
    assert ids(t, per(t, "Pocher")) == ["oliver-pocher"]


def test_hyphenated_pairs_are_two_people():
    # "Wegner-Giffey" tagged as one person span: both names still link (the longer-name rule only
    # applies to names separated by a space, like "Elton John")
    t = "Die Regierung Wegner-Giffey hat versagt."
    i = t.index("Wegner")
    assert "kai-wegner" in ids(t, [(i, i + len("Wegner-Giffey"), "PER", 0.99)])
