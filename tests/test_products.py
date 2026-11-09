"""Entity type PRODUCT ("Apps & Dienste"): curated list, tricky aliases, Wikidata type inference,
and the rollout jobs (seed queues a relink of new entities and a prune of moved aliases)."""

import functools

from analysis import pipeline
from analysis.analyze import link_comment
from analysis.entity_requests import validate
from analysis.linker import TRACKED_TYPES, Registry, _ner_compatible
from analysis.registry_store import _add_entries
from analysis.textutil import split_sentences
from analysis.wikidata import entity_type, public_figure_check
from tests.test_pipeline import fs as fs


@functools.lru_cache(maxsize=1)
def reg():
    return Registry.from_yaml()


def ids(text, spans=(), ctx="", country="DE"):
    return [m.entity_id for m in reg().gazetteer(text, list(spans), ctx, country)]


def span(text, word, tag, score=0.99):
    i = text.index(word)
    return [(i, i + len(word), tag, score)]


# ---------------------------------------------------------------------------- the list


def test_product_list():
    r = reg()
    prods = [e for e in r.entities.values() if e.type == "PRODUCT"]
    assert "PRODUCT" in TRACKED_TYPES
    assert 150 <= len(prods) <= 300
    for e in prods:
        assert e.wikidataId and e.wikidataId.startswith("Q"), e.id
        assert e.description, e.id
        assert e.context, e.id
        assert e.source == "seed", e.id
    # TikTok moved from ORG to PRODUCT and kept its id; generated companies that are services
    # (Netflix, PayPal ...) are curated products with their old ids
    for eid in ("tiktok", "netflix", "spotify", "paypal", "uber", "temu", "n26", "trivago"):
        assert r.entities[eid].type == "PRODUCT", eid
    # the platform is the product, the company stays an organisation
    assert r.entities["facebook"].type == "PRODUCT"
    assert r.entities["meta-platforms"].type == "ORG"
    assert "Facebook" not in r.entities["meta-platforms"].aliases


def test_platform_vs_company():
    assert ids("Facebook ist tot, nur Boomer posten da noch.") == ["facebook"]
    assert ids("Der Facebook-Konzern verdient Milliarden.") == ["meta-platforms"]
    assert ids("Meta Platforms legt Zahlen vor.") == ["meta-platforms"]
    assert ids("Google weiß alles.") == ["google"]
    assert ids("Die Google-Suche ist voller Werbung.") == ["google-suche"]
    assert ids("Ich schaue Prime Video und Disney+.") == ["amazon-prime-video", "disney-plus"]
    assert ids("Amazon liefert schnell.") == ["amazon"]


# ---------------------------------------------------------------------------- tricky aliases


def test_x_never_matches_alone():
    assert ids("X ist ein Platzhalter, zahle X Euro.") == []
    assert ids("Auf X Euro kommt es nicht an.") == []
    assert ids("Der Faktor x ist unbekannt.") == []
    assert ids("Er hat es auf X gepostet.") == ["x-twitter"]  # "auf X" + a social-media keyword
    assert ids("Musk hat bei X die Moderation abgeschafft.") == ["elon-musk", "x-twitter"]
    assert ids("Das stand auf X (ehemals Twitter).") == ["x-twitter"]
    assert ids("Twitter war früher besser.") == ["x-twitter"]
    assert ids("Twitter/X verliert Nutzer.") == ["x-twitter"]


def test_signal():
    assert ids("Das ist ein klares Signal an Putin.") == ["wladimir-putin"]
    assert ids("Signal ist sicherer.") == []
    assert ids("Ich nutze nur noch Signal statt WhatsApp.") == ["signal", "whatsapp"]
    assert ids("Die Signal-App ist verschlüsselt.") == ["signal"]
    assert ids("Der Chat lief über Signal.", ctx="Hegseth teilte Angriffspläne") == ["signal"]
    assert "signal" not in ids("Im Signal Iduna Park war die Stimmung super.")


def test_threads():
    assert ids("In anderen Threads hier wurde das schon gesagt.") == []
    assert ids("Threads wächst schneller als Bluesky.") == ["threads", "bluesky"]
    assert ids("Die Threads-App von Meta.") == ["threads"]


def test_claude_is_also_a_first_name():
    assert ids("Claude ist mein Nachbar.") == []  # "Claude" itself is no AI keyword
    assert ids("Jean-Claude Juncker und die KI-Regulierung.") == []
    assert ids("Ich habe Claude gefragt, der KI-Chatbot war besser.") == ["claude"]
    # a person span vetoes the contextual alias even with AI keywords nearby
    t = "Claude erzählt gern von KI."
    assert ids(t, span(t, "Claude", "PER")) == []
    assert ids("Anthropic Claude kann programmieren.") == ["claude"]


def test_gemini():
    assert ids("Gemini ist mein Sternzeichen.") == []
    assert ids("Gemini kann als KI-Chatbot nichts.") == ["gemini"]
    assert ids("Google Gemini ist schlechter als ChatGPT.") == ["gemini", "chatgpt"]


def test_uber_and_ueber():
    assert ids("Über die Straße fahren wir morgen.") == []
    assert ids("Das ist uber cool.") == []
    assert ids("Uber ist teurer als ein Taxi.") == ["uber"]
    assert ids("Die Uber-Fahrer streiken.") == ["uber"]
    assert ids("Uber Eats liefert kalt.") == ["uber-eats"]


def test_bolt():
    assert ids("Bolt ist billiger als ein Taxi.") == ["bolt"]
    assert ids("Usain Bolt läuft schnell.") == []
    t = "Bolt ist mit dem Taxi gekommen."
    assert ids(t, span(t, "Bolt", "PER")) == []


def test_steam():
    assert ids("Full Steam ahead.") == []
    assert ids("Steam ist die größte Plattform für Spiele.") == ["steam"]
    # weak alias: an ORG or MISC span confirms it
    for tag in ("ORG", "MISC"):
        t = "Steam ist toll."
        assert ids(t, span(t, "Steam", tag)) == ["steam"], tag
    t = "Steam ist toll."
    assert ids(t, span(t, "Steam", "LOC")) == []


def test_wolt_and_temu():
    assert ids("Wolt ihr das wirklich?") == []  # typo of "wollt"
    assert ids("Wolt liefert schneller als Lieferando.") == ["wolt", "lieferando"]
    t = "Wolt ist teuer."
    assert ids(t, span(t, "Wolt", "ORG")) == ["wolt"]
    assert ids("Temu ist Schrott.") == ["temu"]
    assert ids("Bei Temus Preisen kann keiner mithalten.") == ["temu"]


def test_more_ambiguous_names():
    assert ids("Alexa ist meine Tochter.") == []
    assert ids("Alexa hört immer mit, Amazon weiß alles.") == ["alexa", "amazon"]
    assert ids("Er war Copilot im Flugzeug.") == []
    assert ids("Microsoft Copilot ist nutzlos.") == ["microsoft-copilot"]
    assert ids("Die Elster klaut Silber.") == []
    assert ids("ELSTER ist abgestürzt bei der Steuererklärung.") == ["elster"]
    assert ids("Nina Warken sagt das.") == ["nina-warken"]
    assert ids("NINA hat vor dem Unwetter gewarnt.") == ["nina-warn-app"]
    assert ids("Die Teams spielen morgen.") == []
    assert ids("Das Meeting lief über Teams, Microsoft sei Dank.") == ["microsoft-teams", "microsoft"]
    assert ids("Zoom mal ran an das Bild.") == []
    assert ids("Kleinanzeigen in der Zeitung") == []
    assert ids("Ich habe es auf Kleinanzeigen verkauft.") == ["kleinanzeigen"]
    assert ids("Sora ist ein schöner Name.") == []
    assert ids("Markus Babbel war Trainer.") == ["markus-babbel"]


def test_links_and_dialect():
    assert ids("Dafür den Wikipedia Artikel: de.wikipedia.org/wiki/Massaker") == ["wikipedia"]
    assert ids("Siehe paulkrugman.substack.com/p/the-starlink") == []
    assert ids("Der Link: twitter.com/foo") == ["x-twitter"]  # longest alias across case variants
    assert ids("Das steht auf x.com/elonmusk.") == ["x-twitter"]
    assert ids("Hinge ma hoid morgen.") == []  # Austrian dialect, not the dating app
    assert ids("Auf Hinge und Tinder nur Fakes.") == ["hinge", "tinder"]


def test_ner_compatibility():
    assert _ner_compatible("ORG", "PRODUCT") and _ner_compatible("MISC", "PRODUCT")
    assert not _ner_compatible("PER", "PRODUCT") and not _ner_compatible("LOC", "PRODUCT")


def test_wikidata_fallback_does_not_bypass_contextual_alias():
    class Stub:
        lookups = 0

        def lookup(self, surface, tag):
            return {"qid": "Q19718090", "name": "Signal", "type": "PRODUCT"} if surface == "Signal" else None

    text = "Signal ist sicherer."
    matches = link_comment(text, split_sentences(text), [[(0, 6, "ORG", 0.99)]], reg(), wikidata=Stub())
    assert matches == []


# ---------------------------------------------------------------------------- Wikidata types


def item(p31, sitelinks=30, label="Testapp"):
    return {
        "labels": {"de": {"value": label}},
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": q}}}} for q in p31]},
        "sitelinks": {f"wiki{i}": {} for i in range(sitelinks)},
    }


def test_entity_type_inference():
    assert entity_type(item(["Q620615"])) == "PRODUCT"  # mobile app
    assert entity_type(item(["Q7889"])) == "PRODUCT"  # video game
    assert entity_type(item(["Q19967801", "Q4830453"])) == "PRODUCT"  # online service + business (Netflix)
    assert entity_type(item(["Q11032", "Q35127"])) == "ORG"  # newspaper with a website
    assert entity_type(item(["Q4830453"])) == "ORG"
    assert entity_type(item(["Q7278"])) == "PARTY"
    assert entity_type(item(["Q5"])) == "PER"
    assert entity_type(item(["Q515"])) is None  # city


def test_entity_requests_accept_products():
    assert validate(item(["Q620615"], sitelinks=12)) == ("PRODUCT", "")
    assert validate(item(["Q35127"], sitelinks=3))[0] is None
    assert validate(item(["Q515"], sitelinks=50))[0] is None


def test_fallback_needs_well_known_products():
    assert public_figure_check(item(["Q620615"], 25, "Testapp"), "Testapp", "ORG") == (True, "app or digital service")
    assert not public_figure_check(item(["Q620615"], 12, "Testapp"), "Testapp", "ORG")[0]
    assert public_figure_check(item(["Q4830453"], 12, "Testfirma"), "Testfirma", "ORG")[0]


def test_registry_doc_keeps_products():
    r = Registry.from_yaml()
    entry = {"name": "Testapp", "type": "PRODUCT", "wikidataId": "Q999999902", "aliases": ["Testapp"]}
    _add_entries(r, {"testapp": entry})
    assert r.entities["testapp"].type == "PRODUCT"


# ---------------------------------------------------------------------------- rollout jobs


def test_seed_queues_relink_of_new_entities_and_prune_of_moved_aliases(fs, monkeypatch):
    monkeypatch.setattr(pipeline, "_fill_images", lambda reg: None)
    # Firestore state before the PRODUCT list: no facebook entity, "Facebook" was an alias of Meta
    del fs.docs["entities/facebook"]
    fs.docs["entities/meta-platforms"]["aliases"] = ["Meta Platforms", "Facebook", "Meta"]
    out = pipeline.seed(fs, with_images=False)
    jobs = fs.docs["meta/analysis"]["jobs"]
    assert jobs["relink"]["entityIds"] == ["facebook"] and jobs["relink"]["cursor"] is None
    assert jobs["prune"] == {**jobs["prune"], "ids": ["meta-platforms"], "add": ["facebook"]}
    assert out["followups"]["relink"]["queued"] == "relink"
    # seeding again changes nothing and queues nothing new
    out = pipeline.seed(fs, with_images=False)
    assert "followups" not in out


def test_seed_queues_prune_of_dropped_aliases(fs, monkeypatch):
    # "Die Katze" was a plain alias of Daniela Katzenberger and is dropped without a new owner:
    # the mentions it produced must be re-checked
    monkeypatch.setattr(pipeline, "_fill_images", lambda reg: None)
    doc = fs.docs["entities/daniela-katzenberger"]
    doc["aliases"] = [*doc["aliases"], "Die Katze"]
    out = pipeline.seed(fs, with_images=False)
    assert fs.docs["meta/analysis"]["jobs"]["prune"]["ids"] == ["daniela-katzenberger"]
    assert fs.docs["meta/analysis"]["jobs"]["prune"]["add"] == []
    assert "relink" not in out["followups"]


def test_queue_relink_merges_or_waits(fs):
    pipeline.queue_job(fs, "relink", {"cutoff": "2026-10-10", "entityIds": ["a", "b"]})
    pipeline.queue_job(fs, "relink", {"cutoff": "2026-10-09", "entityIds": ["c"]})
    job = fs.docs["meta/analysis"]["jobs"]["relink"]
    assert job["entityIds"] == ["a", "b", "c"] and job["cutoff"] == "2026-10-09"
    # once the scan has started, new entities wait for their own scan
    fs.docs["meta/analysis"]["jobs"]["relink"]["cursor"] = ["2026-10-12", "x"]
    assert pipeline.queue_job(fs, "relink", {"cutoff": "2026-10-10", "entityIds": ["b"]})["alreadyQueued"]
    pipeline.queue_job(fs, "relink", {"cutoff": "2026-10-10", "entityIds": ["b", "d"]})
    jobs = fs.docs["meta/analysis"]["jobs"]
    assert jobs["relink"]["entityIds"] == ["a", "b", "c"] and jobs["relink_next"]["entityIds"] == ["d"]
    pipeline._finish_relink(fs)
    jobs = fs.docs["meta/analysis"]["jobs"]
    assert jobs["relink"]["entityIds"] == ["d"] and "relink_next" not in jobs


def test_queued_prune_moves_mentions_to_the_product(fs, monkeypatch):
    from tests.test_budget import big
    from tests.test_pipeline import comment

    fs.docs["comments/f1"] = comment("f1", "Facebook ist ein Datenkraken, sagt Merz.")
    pipeline.run(fs, limit=100, use_wikidata=False, budget=big(fs))
    assert "mentions/f1_facebook" in fs.docs
    # simulate the old state: the mention belonged to Meta Platforms
    fs.docs["mentions/f1_meta-platforms"] = {**fs.docs.pop("mentions/f1_facebook"), "entityId": "meta-platforms"}
    fs.docs["mentions/f1_friedrich-merz"]["coEntityIds"] = ["meta-platforms"]
    pipeline.queue_job(fs, "prune", {"ids": ["meta-platforms"], "add": ["facebook"]})
    out = pipeline.continue_jobs(fs, big(fs))
    assert out["prune"]["mentionsDeleted"] == 1 and out["prune"]["mentionsAdded"] == 1
    assert "mentions/f1_meta-platforms" not in fs.docs and "mentions/f1_facebook" in fs.docs
    assert fs.docs["mentions/f1_friedrich-merz"]["coEntityIds"] == ["facebook"]
    assert "prune" not in fs.docs["meta/analysis"]["jobs"]
    assert fs.docs["entities/facebook"]["mentions"] == 1 and fs.docs["entities/meta-platforms"]["mentions"] == 0
