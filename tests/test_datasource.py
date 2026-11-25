"""The open-data source layer: the local database, Zetaris behind the same
interface, and the one setting that picks between them. Uses the committed
open data in data/open/, which is public; no document text."""

from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import PASSAGES, FakeRetriever, FakeZetaris, hit
from dalil import datasource as ds
from dalil.app import create_app
from dalil.draft import ExtractiveDrafter
from dalil.opendata import RULES
from dalil.pipeline import Pipeline
from dalil.store import Store
from dalil.verify import Verifier, data_lookup

ROOT = Path(__file__).resolve().parent.parent
OPEN = ROOT / "data" / "open"
RECORDED = json.loads((Path(__file__).parent / "fixtures" / "zetaris_recorded_rows.json").read_text())


@pytest.fixture(scope="module")
def local() -> ds.LocalSource:
    return ds.LocalSource(OPEN)


def csv_rows(name: str) -> list[dict[str, str]]:
    with (OPEN / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# --- the local database ----------------------------------------------------------


def test_local_serves_every_committed_table_with_its_row_count(local):
    with (OPEN / "SOURCES.csv").open(newline="", encoding="utf-8") as fh:
        listed = {r["file"]: int(r["rows"]) for r in csv.DictReader(fh) if r["rows"]}
    with (OPEN / "tables.csv").open(newline="", encoding="utf-8") as fh:
        tables = {f"dalil.{r['table']}": r["file"] for r in csv.DictReader(fh)}
    assert local.tables() == sorted(tables)
    for table, file in tables.items():
        assert local.row_counts[table] == listed[file]
        assert local.sql(f"SELECT COUNT(*) AS n FROM {table}").scalar() == str(listed[file])


def test_every_fixed_query_runs_unchanged_and_returns_one_row_of_strings(local):
    for rule in RULES:
        rows = local.sql(rule.sql).dicts()
        assert len(rows) == 1, rule.name
        assert all(isinstance(v, str) and v for v in rows[0].values()), rule.name


def test_rule_values_match_a_count_made_without_sql(local):
    """Each rule's answer, worked out again from the CSV with plain Python."""
    icp = csv_rows("icp-service-transactions-2020-2025.csv")

    def latest(service):
        r = max((r for r in icp if r["service_en"] == service), key=lambda r: int(r["year"]))
        return {"year": r["year"], "transactions": r["transactions"]}
    assert local.sql(RULES[0].sql).dicts() == [latest("Renewal of ID Card")]
    assert local.sql(RULES[1].sql).dicts() == [latest("Renewal of Residence Permits")]

    tasheel = csv_rows("mohre-tasheel-service-centres.csv")
    assert local.sql(RULES[2].sql).dicts() == [
        {"centres": str(len(tasheel)), "emirates": str(len({r["emirate_en"] for r in tasheel}))}]

    ehs = csv_rows("ehs-medical-fitness-residency-visa-2022-2025.csv")
    year = max(int(r["year"]) for r in ehs)
    assert local.sql(RULES[3].sql).dicts() == [
        {"year": str(year), "exams": str(sum(int(r["total"]) for r in ehs if int(r["year"]) == year))}]


def test_complaints_rule_counts_each_emirate_once_and_skips_the_partial_year(local):
    rows = csv_rows("mohre-solved-complaints-private-sector-2009-2020.csv")
    by_city = defaultdict(lambda: defaultdict(int))
    for r in rows:
        by_city[r["Year"]][r["City_En"]] += int(r["SolvedComplaints_count"])
    # The duplication the rule works around is really in the file.
    assert by_city["2019"]["Al-Ain"] == by_city["2019"]["Abu Dhabi"] == by_city["2019"]["Delma"]
    assert by_city["2019"]["Kalbaa"] == by_city["2019"]["Sharjah"]
    emirates = ("Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Om-El-Quain", "Ras El-Khaema", "Fujiarah")
    want = sum(by_city["2019"][c] for c in emirates)
    assert local.sql(RULES[4].sql).dicts() == [{"year": "2019", "solved": str(want)}]
    assert want < sum(by_city["2019"].values())  # the old sum overcounted


@pytest.mark.parametrize("statement", [
    "DELETE FROM dalil.open_tasheel_centres",
    "UPDATE dalil.open_icp_transactions SET transactions = 1",
    "INSERT INTO dalil.open_icp_transactions (year) VALUES (1)",
    "DROP TABLE dalil.open_icp_transactions",
    "CREATE TABLE dalil.extra (a)",
    "ATTACH DATABASE 'elsewhere.db' AS other",
    "PRAGMA query_only = OFF",
    "SELECT load_extension('x')",
])
def test_local_refuses_anything_but_reading(local, statement):
    with pytest.raises(ds.DataSourceError):
        local.sql(statement)
    assert local.sql(RULES[0].sql).dicts()[0]["transactions"] == "5856085"


def test_local_honours_the_row_limit(local):
    assert len(local.sql("SELECT * FROM dalil.open_tasheel_centres", limit=5).rows) == 5


def test_local_refuses_a_file_that_differs_from_sources_csv(tmp_path):
    shutil.copytree(OPEN, tmp_path / "open")
    f = tmp_path / "open" / "mohre-tasheel-service-centres.csv"
    f.write_text(f.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ds.DataSourceError, match="SHA-256"):
        ds.LocalSource(tmp_path / "open")


def test_local_reads_quoted_multiline_fields(tmp_path):
    (tmp_path / "tables.csv").write_text("table,file\nopen_x,x.csv\n", encoding="utf-8")
    (tmp_path / "x.csv").write_text('id,note,n\n1,"two\nlines",\n2,plain,3\n', encoding="utf-8")
    src = ds.LocalSource(tmp_path)
    assert src.sql("SELECT id, note, n FROM dalil.open_x ORDER BY id").dicts() == [
        {"id": "1", "note": "two\nlines", "n": None}, {"id": "2", "note": "plain", "n": "3"}]


def test_table_files_refuse_a_path_outside_the_folder(tmp_path):
    (tmp_path / "tables.csv").write_text("table,file\nopen_x,../x.csv\n", encoding="utf-8")
    with pytest.raises(ds.DataSourceError, match="outside"):
        ds.table_files(tmp_path)


# --- parity with the server's Zetaris -----------------------------------------------


@pytest.mark.parametrize("rec", RECORDED["recorded"], ids=lambda r: r["sql"][-40:])
def test_local_returns_the_rows_zetaris_returned(local, rec):
    assert local.sql(rec["sql"]).dicts() == rec["rows"]


# --- choosing a source --------------------------------------------------------------


def env(tmp_path, **kw):
    """An environment with an empty Zetaris credentials file, plus kw."""
    empty = tmp_path / "zetaris.env"
    empty.write_text("# no login\n", encoding="utf-8")
    return {"DALIL_ZETARIS_ENV": str(empty), **kw}


def test_auto_uses_the_local_database_when_zetaris_is_not_configured(tmp_path):
    source, why = ds.resolve(env(tmp_path))
    assert source.kind == "local" and why == ""
    assert ds.status(source) == {"connected": True, "kind": "local", "label": "Open data: local database",
                                 "tables": sorted(source.row_counts)}


def test_auto_falls_back_to_local_when_zetaris_does_not_answer(tmp_path):
    dummy = "x" * 12  # a stand-in login; nothing listens on port 9
    e = env(tmp_path, ZETARIS_UI_URL="http://127.0.0.1:9", ZETARIS_ADMIN_EMAIL="a@b.c")
    e["ZETARIS_ADMIN_" + "PASSWORD"] = dummy
    source, why = ds.resolve(e, zetaris_timeout=2)
    assert source.kind == "local"
    assert "Zetaris unreachable" in why and "using the local database" in why
    assert dummy not in why


def test_zetaris_mode_never_falls_back(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="zetaris"))
    assert source is None and "Zetaris unreachable" in why


def test_local_and_off_modes(tmp_path):
    assert ds.resolve(env(tmp_path, DALIL_OPEN_DATA="local"))[0].kind == "local"
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="off"))
    assert source is None and "turned off" in why


def test_auto_with_no_data_files_is_neutral(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA_DIR=str(tmp_path / "nothing")))
    assert source is None and why == "Open data not configured; lookup skipped"
    assert ds.status(None) == {"connected": False, "kind": None, "label": "Open data: not configured", "tables": []}


def test_a_bad_mode_is_reported_not_raised(tmp_path):
    source, why = ds.resolve(env(tmp_path, DALIL_OPEN_DATA="sometimes"))
    assert source is None and "DALIL_OPEN_DATA must be one of" in why


def test_status_of_a_source_that_stops_answering():
    out = ds.status(FakeZetaris(fail=True))
    assert out["connected"] is False and out["kind"] == "zetaris" and out["tables"] == []


# --- end to end on the local database --------------------------------------------------


def local_pipeline(tmp_path, local):
    store = Store(tmp_path / "dalil.db")
    return Pipeline(store, FakeRetriever([hit(PASSAGES[0], 20.0), hit(PASSAGES[1], 15.0)]),
                    {p["passage_id"]: dict(p) for p in PASSAGES}, drafter=ExtractiveDrafter(),
                    zetaris_factory=lambda: local)


def test_pipeline_answers_from_the_local_database_and_the_checker_reruns_it(tmp_path, local):
    p = local_pipeline(tmp_path, local)
    rec = p.answer("How early can I renew my Emirates ID card?")
    data = [s for s in rec.check["sentences"] if any("sql" in c for c in s["citations"])]
    assert len(data) == 1 and data[0]["kept"]
    assert data[0]["text"] == "ICP open data counts 5856085 Emirates ID card renewals in 2025."
    assert all(c["ok"] and c["reason"] == "value returned by the query" for c in data[0]["citations"])
    assert rec.data[0]["source"] == "local"
    looked = next(e for e in p.store.get(rec.id)["audit"] if e["kind"] == "data_looked_up")
    assert looked["detail"]["source"] == "local"


def test_every_rule_passes_the_checker_on_the_local_database(local):
    from dalil.draft import data_claims
    from dalil.opendata import run_rules
    for rule in RULES:
        results = run_rules([rule], lambda sql: local.sql(sql, limit=200).dicts(), source="local")
        claims = data_claims(results)
        check = Verifier({}, data_lookup(local), allowed_sql=[rule.sql]).check_answer(claims)
        assert check.status == "pass", (rule.name, check.to_dict())


def test_health_and_citations_name_the_local_database(tmp_path, local):
    p = local_pipeline(tmp_path, local)
    tc = TestClient(create_app(p, {}, web_dir=None), base_url="http://127.0.0.1")
    health = tc.get("/api/health").json()
    assert health["zetaris"] is False
    assert health["data_source"]["connected"] and health["data_source"]["kind"] == "local"
    assert health["data_source"]["label"] == "Open data: local database"
    qid = tc.post("/api/questions", json={"question": "How early can I renew my Emirates ID card?"}).json()["id"]
    d = tc.get(f"/api/questions/{qid}").json()
    cits = [c for s in d["check"]["sentences"] for c in s["citations"] if c.get("sql")]
    assert cits and all(c["source_label"] == "queried from the local database" for c in cits)
    assert d["data"][0]["source_label"] == "queried from the local database"


def test_health_with_no_source_is_neutral(tmp_path):
    p = Pipeline(Store(tmp_path / "dalil.db"), FakeRetriever([]), {}, zetaris_factory=None)
    health = TestClient(create_app(p, {}, web_dir=None), base_url="http://127.0.0.1").get("/api/health").json()
    assert health["zetaris"] is False
    assert health["data_source"] == {"connected": False, "kind": None, "label": "Open data: not configured",
                                     "tables": []}


def test_old_records_without_a_source_are_shown_as_zetaris():
    from dalil.app import data_kind
    assert data_kind({"rows": [{"year": "2025"}]}) == "zetaris"
    assert data_kind({"rows": []}) == ""
    assert data_kind({"rows": [{"year": "2025"}], "source": "local"}) == "local"


def test_a_source_that_stops_answering_is_dropped_and_resolved_again(tmp_path, local):
    sources = [FakeZetaris(), local]
    p = Pipeline(Store(tmp_path / "dalil.db"), FakeRetriever([]), {}, zetaris_factory=lambda: sources.pop(0))
    assert isinstance(p.data_source()[0], FakeZetaris)
    p._zetaris.fail = True
    assert p.data_source_status()["kind"] == "local"
