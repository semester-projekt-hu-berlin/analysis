"""Notability for community proposals: German or English Wikipedia, or MIN_SITELINKS overall."""

from analysis.entity_requests import MIN_SITELINKS, validate


def _org(sitelinks: dict) -> dict:
    # Q4830453 (business) is an organisation class known to entity_type().
    return {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q4830453"}}}}]}, "sitelinks": sitelinks}


def test_german_article_is_enough():
    etype, reason = validate(_org({"dewiki": {}, "nlwiki": {}}))
    assert reason == "" and etype == "ORG"


def test_english_article_is_enough():
    etype, _ = validate(_org({"enwiki": {}}))
    assert etype == "ORG"


def test_neither_german_nor_english_with_few_sitelinks_is_rejected():
    etype, reason = validate(_org({"frwiki": {}, "nlwiki": {}}))
    assert etype is None and "sitelinks" in reason


def test_many_sitelinks_without_english_is_enough():
    etype, _ = validate(_org({f"wiki{i}": {} for i in range(MIN_SITELINKS)}))
    assert etype == "ORG"


def test_api_type_is_used_for_subclass_only_organisations():
    disco = {
        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q1228895"}}}}]},
        "sitelinks": {"dewiki": {}},
    }
    assert validate(disco) == (None, validate(disco)[1])
    assert validate(disco, "ORG") == ("ORG", "")
    assert validate(disco, "PER")[0] is None  # a person needs Q5, not a hint
