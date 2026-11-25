"""One question, start to finish: screen, retrieve, look up open data, draft,
verify, queue for a person. Every step lands in the audit store.

    from dalil.pipeline import answer
    record = answer("What do I need to renew my Emirates ID?")

The status is "held" when the verifier holds the draft, else
"awaiting_review". Nothing is ever delivered here; a reviewer approves it
through the store (or the HTTP app).
"""

from __future__ import annotations

import sys

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from dalil import lang, safety
from dalil.draft import (FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL, Drafter, ExtractiveDrafter, caveat, data_claims,
                         refusal)
from dalil.opendata import DataResult, match_rules, run_rules
from dalil.retrieve import Hit, Searcher, make_retriever
from dalil.store import Store
from dalil.hmem import Guidance, guidance
from dalil.laws import is_scope_question, law_refs, scope_passages
from dalil.subjects import names_subject, subject_docs
from dalil.verify import Verifier, data_lookup, load_passages

def cited_passages(check: dict[str, Any] | None) -> list[str]:
    """Passage ids the kept sentences of a checked answer cite."""
    return sorted({c["passage_id"] for s in (check or {}).get("sentences", []) if s.get("kept")
                   for c in s.get("citations", []) if c.get("passage_id")})


def document_titles() -> dict[str, str]:
    """doc_id -> title from the manifest, or nothing if it cannot be read."""
    from dalil.corpus import doc_id, load_manifest
    try:
        return {doc_id(r["file"]): r.get("title", "") for r in load_manifest()}
    except Exception as e:
        print(f"warning: no document titles for the checker ({type(e).__name__})", file=sys.stderr)
        return {}


def manifest_laws() -> dict[str, Any]:
    """doc_id -> LawRef for every numbered law in the manifest, or nothing."""
    from dalil.corpus import load_manifest
    try:
        return law_refs(load_manifest())
    except Exception as e:
        print(f"warning: no law list for the checker ({type(e).__name__})", file=sys.stderr)
        return {}


# The only caveat and refusal sentences that may go out without a citation,
# in every language Dalil answers in.
FIXED_UNCITED = lang.all_fixed({
    WEAK_REFUSAL, FEE_CAVEAT, LAW_CAVEAT,
    safety.REFUSAL_TEXT, safety.REFUSAL_FOLLOW_UP, safety.PERSONAL_DATA_CAVEAT, safety.ASK_FOR_DETAIL,
})
# How many of the top hits are looked at for laws whose scope article a
# question about a law's reach needs, and how many scope passages are added.
SCOPE_FROM = 8
SCOPE_ADD = 3

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PASSAGES = REPO_ROOT / "build" / "passages.jsonl"
K = 8
ZETARIS_TIMEOUT = 30.0


@dataclass
class Record:
    id: str
    question: str  # redacted
    flags: list[str]
    status: str
    retrieval: list[dict[str, Any]] = field(default_factory=list)
    data: list[dict[str, Any]] = field(default_factory=list)
    draft: list[dict[str, Any]] = field(default_factory=list)
    check: dict[str, Any] = field(default_factory=dict)


def default_zetaris() -> Any:
    """A Zetaris client, or raise. Imported lazily so tests never touch it."""
    from dalil import zetaris as z

    return z.Client(timeout=ZETARIS_TIMEOUT)


class NoSource(RuntimeError):
    """Raised by default_data_source when no open-data source is available."""


def default_data_source() -> Any:
    """The open-data source DALIL_OPEN_DATA picks (dalil.datasource), or raise
    NoSource with the reason. Zetaris when configured and answering, else the
    local database, by default."""
    from dalil import datasource

    source, why = datasource.resolve(zetaris_timeout=ZETARIS_TIMEOUT)
    if source is None:
        raise NoSource(why)
    if why:  # auto mode fell back to the local database
        print(f"open data: {why}", file=sys.stderr)
    return source


class Pipeline:
    def __init__(self, store: Store, retriever: Searcher, passages: dict[str, dict[str, Any]],
                 drafter: Drafter | None = None,
                 zetaris_factory: Callable[[], Any] | None = default_data_source, k: int = K,
                 memory: Any = None, redraft: bool = True):
        self.store = store
        self.retriever = retriever
        self.passages = passages
        self.drafter = drafter or ExtractiveDrafter()
        # Builds the open-data source: anything with sql(statement, limit) whose
        # result has dicts(). The name is kept from when only Zetaris existed.
        self.zetaris_factory = zetaris_factory
        self.k = k
        self._zetaris: Any = None
        # Review memory (dalil.hmem.HMem or a stand-in with recall()); None turns it off.
        self.memory = memory
        # When the checker cuts every claim the model wrote, quote the sources
        # with the extractive drafter instead of holding an empty answer.
        self.redraft = redraft
        self.titles = document_titles()
        self.laws = manifest_laws()

    @classmethod
    def from_build(cls, store: Store | None = None, passages_path: Path | str = DEFAULT_PASSAGES,
                   **kwargs: Any) -> "Pipeline":
        return cls(store or Store(), make_retriever(passages_path), load_passages(passages_path), **kwargs)

    def data_source(self) -> tuple[Any, str]:
        """The open-data source and "", or None and why it is unavailable."""
        if self._zetaris is not None:
            return self._zetaris, ""
        if self.zetaris_factory is None:
            return None, "Open data not configured; lookup skipped"
        try:
            source = self.zetaris_factory()
            source.sql("SELECT 1")
        except NoSource as e:
            return None, str(e)
        except Exception as e:  # neither source puts credentials in its messages
            return None, f"Open data source unreachable; lookup skipped ({type(e).__name__}: {str(e)[:160]})"
        self._zetaris = source
        return source, ""

    # The old name, for callers written before the local database existed.
    zetaris = data_source

    def data_source_status(self) -> dict[str, Any]:
        """The health endpoint's data_source block. A cached source that stops
        answering is dropped, so the next question resolves again (in auto
        mode, Zetaris down means the local database takes over)."""
        from dalil import datasource

        source, _ = self.data_source()
        out = datasource.status(source)
        if source is not None and not out["connected"]:
            self._zetaris = None
            source, _ = self.data_source()
            out = datasource.status(source)
        return out

    def recall(self, qid: str, question: str) -> Guidance:
        """Reviews of similar questions, from the review memory, and log what was used."""
        if self.memory is None:
            return Guidance()
        recalled = self.memory.recall(question)
        if recalled is None:
            self.store.event(qid, "recalled", {"available": False})
            return Guidance()
        g = guidance(question, recalled.get("memories") or [])
        self.store.event(qid, "recalled", {
            "available": True, "similar_reviews": g.similar,
            "passages_left_out": sorted(g.avoid), "notes": len(g.notes),
        })
        return g

    def search(self, question: str, avoid: set[str]) -> list[Hit]:
        """Top k hits, leaving out `avoid` and filling up from further down the ranking.

        For a question about what a law covers, the scope article of each law
        in the results comes first, in the question's language, because it is
        the law's own answer and its wording is often too far from the
        question's to rank high.
        """
        hits = self.retriever.search(question, self.k + len(avoid))
        hits = [h for h in hits if h.passage_id not in avoid][: self.k]
        hits = self.on_subject(question, hits)
        if not is_scope_question(question) or not self.laws:
            return hits
        law_docs = list(dict.fromkeys(h.doc_id for h in hits[:SCOPE_FROM] if h.doc_id in self.laws))
        have = {h.passage_id for h in hits}
        question_lang = lang.language(question)
        # The same law in the question's language, when the results only hold another.
        keys = {self.laws[d].key for d in law_docs}
        law_docs += [d for d, ref in self.laws.items() if ref.key in keys and d not in law_docs]
        wanted = [d for d in law_docs if self._language(d) == question_lang] or law_docs
        extra = [pid for pid in scope_passages(wanted, self.passages)
                 if pid not in have and pid not in avoid][:SCOPE_ADD]
        return [self._hit(pid) for pid in extra] + hits

    @staticmethod
    def _all_claims_cut(check: dict[str, Any]) -> bool:
        claims = [s for s in check["sentences"] if s["kind"] == "claim"]
        return bool(claims) and not any(s["kept"] for s in claims)

    def on_subject(self, question: str, hits: list[Hit]) -> list[Hit]:
        """When the question names a service that has its own page, and that
        page is in the results, keep only passages that can be about it: that
        page, the laws, and passages that name the service themselves.

        "Can a Golden Visa holder sponsor their parents?" otherwise gets the
        rules for employees' families and for UAE citizens' parents, which
        read close to the question and do not apply to it."""
        titles, laws = getattr(self, "titles", {}), getattr(self, "laws", {})
        subjects = subject_docs(question, titles)
        if not subjects or not any(h.doc_id in subjects for h in hits):
            return hits
        names = [titles[d].split("|")[0] for d in subjects]
        return [h for h in hits if h.doc_id in subjects or h.doc_id in laws
                or names_subject(f"{titles.get(h.doc_id, '')} {h.heading or ''} {h.text}", names)]

    def _language(self, doc_id: str) -> str:
        for p in self.passages.values():
            if p["doc_id"] == doc_id:
                return p.get("language") or ""
        return ""

    def _hit(self, pid: str) -> Hit:
        p = self.passages[pid]
        return Hit(passage_id=pid, doc_id=p["doc_id"], page=int(p["page"]), score=0.0, heading=p.get("heading"),
                   text=p["text"], start=int(p["start"]), end=int(p["end"]), kind=p.get("kind") or "")

    def answer(self, question: str) -> Record:
        screened = safety.screen(question)
        # From here on only the redacted text exists.
        qid = self.store.create_question(screened.redacted_question, screened.flags)
        self.store.event(qid, "screened", {"flags": screened.flags})
        try:
            return self._answer(qid, screened)
        except Exception as e:
            self.store.event(qid, "failed", {"error": type(e).__name__, "message": str(e)[:300]})
            self.store.save_record(qid, [], [], [], {"status": "held", "answer": "", "sentences": []}, "held")
            self.store.event(qid, "queued", {"status": "held"})
            return Record(qid, screened.redacted_question, screened.flags, "held")

    def _answer(self, qid: str, screened: safety.ScreenResult) -> Record:
        q = screened.redacted_question
        hits: list[Hit] = []
        data: list[DataResult] = []
        client = None
        if screened.is_rule_evasion:
            self.store.event(qid, "retrieved", {"skipped": "rule_evasion", "hits": 0})
            draft = [refusal(safety.REFUSAL_TEXT), refusal(safety.REFUSAL_FOLLOW_UP)]
        elif screened.is_vague:
            # Too few key words to know the subject: ask, rather than guess.
            self.store.event(qid, "retrieved", {"skipped": "vague_question", "hits": 0})
            draft = [refusal(safety.ASK_FOR_DETAIL)]
            if screened.has_personal_data:
                draft.append(caveat(safety.PERSONAL_DATA_CAVEAT))
        else:
            past = self.recall(qid, q)
            hits = self.search(q, past.avoid)
            self.store.event(qid, "retrieved", {
                "k": self.k, "hits": len(hits),
                "top": [{"passage_id": h.passage_id, "score": round(h.score, 3)} for h in hits[:3]],
            })
            rules = match_rules(q)
            if rules:
                client, why = self.data_source()
                run = (lambda sql: client.sql(sql, limit=200).dicts()) if client else None
                kind = _kind(client)
                data = run_rules(rules, run, why, source=kind)
                if client is not None and all(d.note.startswith("lookup failed") for d in data):
                    self._zetaris = None  # resolve again next time; auto mode can fall back
                self.store.event(qid, "data_looked_up", {
                    "source": kind or None,
                    "tables": [d.table for d in data],
                    "rows": [len(d.rows) for d in data],
                    "notes": [d.note for d in data if d.note],
                })
            draft = self.drafter.draft(q, hits, data, notes=past.notes)
            if screened.has_personal_data:
                draft = [*draft, caveat(safety.PERSONAL_DATA_CAVEAT)]
        # Fixed refusals and caveats go out in the question's language.
        question_lang = lang.language(q)
        draft = [{**s, "text": lang.fixed(s["text"], question_lang)} if s.get("kind") in ("refusal", "caveat") else s
                 for s in draft]
        ran = "fixed refusal" if (screened.is_rule_evasion or screened.is_vague) else (
            getattr(self.drafter, "last_used", "") or type(self.drafter).__name__)
        error = "" if (screened.is_rule_evasion or screened.is_vague) else getattr(self.drafter, "last_error", "")
        self.store.event(qid, "drafted", {
            "drafter": ran, **({"fallback_reason": error} if error else {}),
            "sentences": len(draft),
            "kinds": [s.get("kind", "claim") for s in draft],
        })
        drafted = len(draft)
        draft = drop_repeats(draft, data)
        repeats = drafted - len(draft)
        drafted = len(draft)
        draft = drop_internal_names(draft, data)
        internal = drafted - len(draft)
        draft, attached = attach_data(draft, data)
        if client is None and any("sql" in c for s in draft for c in s.get("citations") or []):
            client, _ = self.data_source()
        subjects = {d: self.titles[d] for d in subject_docs(q, self.titles) if any(h.doc_id == d for h in hits)}
        verifier = Verifier(self.passages, data_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED,
                            allowed_passages=[h.passage_id for h in hits], titles=self.titles, laws=self.laws,
                            similarity=getattr(self.retriever, "similarities", None), question=q,
                            subjects=subjects,
                            data_facts={d.sql: " ".join(f for f in (d.fact, d.fact_ar) if f) for d in data if d.fact})
        check = verifier.check_answer(draft).to_dict()
        added = restate_data(draft, check, data, question_lang)
        if added:
            draft = added
            check = verifier.check_answer(draft).to_dict()
        if self.redraft and self._all_claims_cut(check) and not isinstance(self.drafter, ExtractiveDrafter):
            # The model's every claim failed the checker. Quote the sources
            # instead: the extractive drafter copies whole source sentences,
            # so whatever it keeps is the documents' own words.
            # Quote from the page the question is about, when it names one.
            redraft = ExtractiveDrafter().draft(q, [h for h in hits if h.doc_id in subjects] or hits, data)
            if screened.has_personal_data:
                redraft = [*redraft, caveat(safety.PERSONAL_DATA_CAVEAT)]
            redraft = [{**s, "text": lang.fixed(s["text"], question_lang)}
                       if s.get("kind") in ("refusal", "caveat") else s for s in redraft]
            second = verifier.check_answer(redraft).to_dict()
            if lang.language(" ".join(s["text"] for s in second["sentences"] if s["kept"])) == question_lang \
                    and any(s["kept"] and s["kind"] == "claim" for s in second["sentences"]) \
                    and second["status"] != "held":
                self.store.event(qid, "redrafted", {"drafter": "ExtractiveDrafter",
                                                    "reason": "the checker cut every claim the model wrote",
                                                    "cut_first": [s["reasons"] for s in check["sentences"]
                                                                  if not s["kept"]]})
                draft, check = redraft, second
        kept = sum(s["kept"] for s in check["sentences"])
        self.store.event(qid, "verified", {"status": check["status"], "kept": kept,
                                           "cut": len(check["sentences"]) - kept,
                                           **({"data_restated": sum(1 for s in draft if s.get("restated"))}
                                              if added else {}),
                                           **({"data_citations_attached": attached} if attached else {}),
                                           **({"repeats_dropped": repeats} if repeats else {}),
                                           **({"internal_names_dropped": internal} if internal else {})})
        status = "held" if check["status"] == "held" else "awaiting_review"
        retrieval = [asdict(h) for h in hits]
        data_dicts = [d.to_dict() for d in data]
        self.store.save_record(qid, retrieval, data_dicts, draft, check, status)
        self.store.event(qid, "queued", {"status": status})
        return Record(qid, q, screened.flags, status, retrieval, data_dicts, draft, check)


def _kind(source: Any) -> str:
    if source is None:
        return ""
    from dalil.datasource import kind_of

    return kind_of(source)


def _numbers(text: str) -> set[str]:
    from dalil.verify import numbers
    return numbers(text)


def drop_repeats(draft: list[dict[str, Any]], data: list[DataResult] = ()) -> list[dict[str, Any]]:
    """The draft without claims that repeat an earlier one.

    A small model sometimes writes the same sentence several times, or
    follows "EHS carried out 1424313 exams in 2025." with "The data value is
    1424313." A claim is a repeat when its text matches an earlier claim word
    for word, or when every number in it is an open-data value (not a year
    alone) that an earlier claim already stated. One copy says it."""
    values = {n for res in data if res.rows for k, v in res.rows[0].items()
              if v is not None and k.lower() != "year" for n in _numbers(str(v))}
    seen: set[str] = set()
    said: set[str] = set()
    out = []
    for s in draft:
        if s.get("kind", "claim") == "claim":
            key = " ".join(str(s.get("text", "")).split()).casefold()
            nums = _numbers(str(s.get("text", "")))
            if key in seen or (nums & values and nums <= said):
                continue
            seen.add(key)
            said |= nums
        out.append(s)
    return out


def drop_internal_names(draft: list[dict[str, Any]], data: list[DataResult]) -> list[dict[str, Any]]:
    """The draft without claims that name an open-data table, such as "The data
    is from the open_ehs_medical_fitness table." A table name is Dalil's own
    identifier and tells a resident nothing; the screen shows the source."""
    names = {n.lower() for res in data for n in (res.table, res.table.split(".")[-1])}
    if not names:
        return draft
    return [s for s in draft if s.get("kind", "claim") != "claim"
            or not any(n in str(s.get("text", "")).lower() for n in names)]


def attach_data(draft: list[dict[str, Any]], data: list[DataResult]) -> tuple[list[dict[str, Any]], int]:
    """Give a drafted claim the data citations for the lookup values it states.

    A small model often writes "The number of exams is 1424313." and cites a
    passage, or nothing that holds the number. When a claim states a value
    (other than a year) from the row of a lookup Dalil ran for this question,
    the citation to that column is added, with the year too if the claim
    states it. The checker then reruns the query and tests the wording as for
    any data citation, so this can only keep a sentence the data backs.
    Returns the new draft and how many citations were added.
    """
    added = 0
    out = []
    for s in draft:
        if s.get("kind", "claim") != "claim":
            out.append(s)
            continue
        stated = _numbers(s.get("text", ""))
        cits = list(s.get("citations") or [])
        have = {(c.get("sql"), c.get("column")) for c in cits if "sql" in c}
        new = []
        for res in data:
            if not res.rows:
                continue
            row = res.rows[0]
            cols = [k for k, v in row.items() if v not in (None, "") and _numbers(str(v))
                    and _numbers(str(v)) <= stated]
            if not any(k.lower() != "year" for k in cols):
                continue
            new += [{"table": res.table, "sql": res.sql, "column": k, "value": str(row[k])}
                    for k in cols if (res.sql, k) not in have]
        if new:
            added += len(new)
            s = {**s, "citations": cits + new, "data_attached": len(new)}
        out.append(s)
    return out, added


def _states(sentence: Mapping[str, Any], citation: Mapping[str, Any]) -> bool:
    """True when the sentence's text holds the data citation's value."""
    value = _numbers(str(citation.get("value") or ""))
    return bool(value) and value <= _numbers(sentence.get("text", ""))


def _drop_unstated_data(sentence: dict[str, Any]) -> dict[str, Any]:
    cits = sentence.get("citations") or []
    keep = [c for c in cits if "sql" not in c or _states(sentence, c)]
    if len(keep) == len(cits) or sentence.get("kind", "claim") != "claim":
        return sentence
    return {**sentence, "citations": keep, "unstated_data_dropped": len(cits) - len(keep)}


def restate_data(draft: list[dict[str, Any]], check: dict[str, Any], data: list[DataResult],
                 language: str = "en") -> list[dict[str, Any]] | None:
    """A new draft when the model tried to state an open-data result but no
    kept sentence states it; else None.

    A sentence carries a lookup only when it cites the query and states the
    cited value. A data citation on a sentence that does not state its value
    backs nothing, so it is taken off that sentence; the sentence still has to
    pass on its other citations.

    "Tried" means a drafted claim cites the lookup's query or writes one of the
    values in its row other than the year. The rule's own fixed sentence (dalil/opendata.py) then
    states the value, with a citation the checker reruns like any other. The
    model's cut sentence stays in the record for the reviewer to see. A lookup
    the model left alone is left out: it judged the data off the question.
    """
    sentences = check.get("sentences", [])
    out = [_drop_unstated_data(s) for s in draft]
    changed = out != draft
    extra = []
    for res in data:
        if not res.rows or not res.fact:
            continue
        carried = any(s["kept"] and any(c.get("ok") and c.get("sql") == res.sql and _states(s, c)
                                        for c in s.get("citations", []))
                      for s in sentences)
        if carried:
            continue
        # A year alone does not show intent: passage claims carry years too.
        values = _numbers(" ".join(str(v) for k, v in res.rows[0].items() if v is not None and k.lower() != "year"))
        tried = any(s.get("kind", "claim") == "claim" and (
            any(c.get("sql") == res.sql for c in s.get("citations") or []) or _numbers(s.get("text", "")) & values)
            for s in draft)
        if tried:
            # In the question's language: the Arabic sentence when there is one.
            said = DataResult(res.table, res.sql, res.rows, fact=res.fact_ar if language == lang.ARABIC and res.fact_ar
                              else res.fact)
            extra += [{**claim, "restated": True} for claim in data_claims([said])]
    if not extra:
        return out if changed else None
    # After the model's claims, before the fixed caveats and refusals.
    at = max((i + 1 for i, s in enumerate(out) if s.get("kind", "claim") == "claim"), default=0)
    return out[:at] + extra + out[at:]


_default: Pipeline | None = None


def answer(question: str) -> Record:
    """Answer through the default pipeline over build/passages.jsonl and build/dalil.db."""
    global _default
    if _default is None:
        _default = Pipeline.from_build()
    return _default.answer(question)
