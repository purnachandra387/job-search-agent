import fetch_remotive_jobs as fetch


def test_strip_html_removes_tags_and_truncates():
    assert fetch.strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert len(fetch.strip_html("x" * 5000)) == 1500


def test_stable_id_is_deterministic_and_clear_of_sample_ids():
    a = fetch.stable_id("Backend Dev", "Acme", "Remote")
    assert a == fetch.stable_id("backend dev", "ACME", "remote")
    assert a >= fetch.ID_OFFSET


def test_make_record_skips_incomplete_postings():
    assert fetch.make_record("", "Acme", "Remote", "<p>desc</p>") is None
    rec = fetch.make_record("Dev", "Acme", "", "<p>desc</p>")
    assert rec is not None and rec[3] == "Remote"
