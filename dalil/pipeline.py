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
from typing import Any, Callable

from dalil import lang, safety
from dalil.draft import FEE_CAVEAT, LAW_CAVEAT, WEAK_REFUSAL, Drafter, ExtractiveDrafter, caveat, refusal
from dalil.opendata import DataResult, match_rules, run_rules
from dalil.retrieve import Hit, Searcher, make_retriever
from dalil.store import Store
from dalil.hmem import Guidance, guidance
from dalil.laws import is_scope_question, law_refs, scope_passages
from dalil.verify import Verifier, load_passages, zetaris_lookup

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


class Pipeline:
    def __init__(self, store: Store, retriever: Searcher, passages: dict[str, dict[str, Any]],
                 drafter: Drafter | None = None,
                 zetaris_factory: Callable[[], Any] | None = default_zetaris, k: int = K,
                 memory: Any = None):
        self.store = store
        self.retriever = retriever
        self.passages = passages
        self.drafter = drafter or ExtractiveDrafter()
        self.zetaris_factory = zetaris_factory
        self.k = k
        self._zetaris: Any = None
        # Review memory (dalil.hmem.HMem or a stand-in with recall()); None turns it off.
        self.memory = memory
        self.titles = document_titles()
        self.laws = manifest_laws()

    @classmethod
    def from_build(cls, store: Store | None = None, passages_path: Path | str = DEFAULT_PASSAGES,
                   **kwargs: Any) -> "Pipeline":
        return cls(store or Store(), make_retriever(passages_path), load_passages(passages_path), **kwargs)

    def zetaris(self) -> tuple[Any, str]:
        """The Zetaris client and "", or None and why it is unavailable."""
        if self._zetaris is not None:
            return self._zetaris, ""
        if self.zetaris_factory is None:
            return None, "Zetaris not configured; lookup skipped"
        try:
            client = self.zetaris_factory()
            client.sql("SELECT 1")
        except Exception as e:  # ZetarisError messages carry no credentials
            return None, f"Zetaris unreachable; lookup skipped ({type(e).__name__}: {str(e)[:160]})"
        self._zetaris = client
        return client, ""

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
                client, why = self.zetaris()
                run = (lambda sql: client.sql(sql, limit=200).dicts()) if client else None
                data = run_rules(rules, run, why)
                self.store.event(qid, "data_looked_up", {
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
        if client is None and any("sql" in c for s in draft for c in s.get("citations") or []):
            client, _ = self.zetaris()
        verifier = Verifier(self.passages, zetaris_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED,
                            allowed_passages=[h.passage_id for h in hits], titles=self.titles, laws=self.laws,
                            similarity=getattr(self.retriever, "similarities", None))
        check = verifier.check_answer(draft).to_dict()
        kept = sum(s["kept"] for s in check["sentences"])
        self.store.event(qid, "verified", {"status": check["status"], "kept": kept,
                                           "cut": len(check["sentences"]) - kept})
        status = "held" if check["status"] == "held" else "awaiting_review"
        retrieval = [asdict(h) for h in hits]
        data_dicts = [d.to_dict() for d in data]
        self.store.save_record(qid, retrieval, data_dicts, draft, check, status)
        self.store.event(qid, "queued", {"status": status})
        return Record(qid, q, screened.flags, status, retrieval, data_dicts, draft, check)


_default: Pipeline | None = None


def answer(question: str) -> Record:
    """Answer through the default pipeline over build/passages.jsonl and build/dalil.db."""
    global _default
    if _default is None:
        _default = Pipeline.from_build()
    return _default.answer(question)
