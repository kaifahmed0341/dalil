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
from dalil.retrieve import Hit, Searcher, make_retriever, tokenize
from dalil.store import Store
from dalil.hmem import Guidance, guidance
from dalil.laws import is_scope_question, law_refs, scope_passages
from dalil.subjects import names_subject, subject_docs, title_terms

# Words for the person who holds a service, not for what is asked about it.
_ROLE_WORDS = frozenset(tokenize("holder holders owner owners applicant applicants resident residents person people"))
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
# Past reviews whose questions are translated for matching, per request, and
# how many translations are kept across requests.
MEMORY_TRANSLATIONS = 5
ENGLISH_CACHE_MAX = 2000

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
                 memory: Any = None, redraft: bool = True):
        self.store = store
        self.retriever = retriever
        self.passages = passages
        self.drafter = drafter or ExtractiveDrafter()
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

    def recall(self, qid: str, question: str, english: str = "") -> Guidance:
        """Reviews of similar questions, from the review memory, and log what was used."""
        if self.memory is None:
            return Guidance()
        recalled = self.memory.recall(question)
        if recalled is None:
            self.store.event(qid, "recalled", {"available": False})
            return Guidance()
        g = guidance(question, recalled.get("memories") or [], english=self._to_english(question, english))
        self.store.event(qid, "recalled", {
            "available": True, "similar_reviews": g.similar,
            "passages_left_out": sorted(g.avoid), "notes": len(g.notes),
        })
        return g

    def _to_english(self, question: str, english: str) -> Callable[[str], str] | None:
        """English for the review memory's question matching: this question's
        translation, English text unchanged, others translated once and cached."""
        translate = self.translator()
        if translate is None:
            return None
        cache: dict[str, str] = getattr(self, "_english_cache", {})
        if len(cache) > ENGLISH_CACHE_MAX:
            cache = {}
        self._english_cache = cache
        failed: set[str] = set()  # this request only: a later request tries again
        if english:
            cache[question] = english

        budget = [MEMORY_TRANSLATIONS]

        def to_english(text: str) -> str:
            """English for `text`, or "" when it cannot be had. A failure is
            cached too, and each request translates at most a few memories,
            so a slow model cannot hold a request for long."""
            if lang.language(text) == lang.ENGLISH:
                return text
            if text in cache:
                return cache[text]
            if text in failed or budget[0] <= 0:
                return ""
            budget[0] -= 1
            try:
                cache[text] = translate(text)
            except Exception:
                failed.add(text)
                return ""
            return cache[text]
        return to_english

    def translator(self) -> Callable[[str], str] | None:
        """The model's translate(), when the drafter has a model."""
        d = self.drafter
        for candidate in (d, getattr(d, "primary", None), getattr(getattr(d, "inner", None), "primary", None)):
            if candidate is not None and callable(getattr(candidate, "translate", None)):
                return candidate.translate
        return None

    def english_query(self, qid: str, question: str) -> str:
        """For a question that is not in English, its English translation, or ""."""
        if lang.language(question) == lang.ENGLISH:
            return ""
        translate = self.translator()
        if translate is None:
            return ""
        try:
            english = translate(question)
        except Exception as e:  # the model is down: search with the question alone
            self.store.event(qid, "translated", {"ok": False, "error": type(e).__name__})
            return ""
        english = english if lang.language(english) == lang.ENGLISH else ""
        self.store.event(qid, "translated", {"ok": bool(english), "english": english})
        return english

    def search(self, question: str, avoid: set[str], english: str = "") -> list[Hit]:
        """Top k hits, leaving out `avoid` and filling up from further down the ranking.

        For a question about what a law covers, the scope article of each law
        in the results comes first, in the question's language, because it is
        the law's own answer and its wording is often too far from the
        question's to rank high.
        """
        hits = self.retriever.search(question, self.k + len(avoid))
        if english:
            hits = fuse(self.retriever.search(english, self.k + len(avoid)), hits)
        hits = [h for h in hits if h.passage_id not in avoid][: self.k]
        hits = self.on_subject(english or question, hits)
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
        # What the question asks about the subject: its words, less the
        # subject's own name and words for the person holding it.
        asked = set(tokenize(question)) - set().union(*(title_terms(n) for n in names)) - _ROLE_WORDS \
            - safety.subject_generic()
        return [h for h in hits if h.doc_id in subjects or h.doc_id in laws
                or (names_subject(f"{titles.get(h.doc_id, '')} {h.heading or ''} {h.text}", names)
                    and set(tokenize(h.text)) & asked)]

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
        english = ""  # the question in English, when it was asked in another language
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
            english = self.english_query(qid, q)
            past = self.recall(qid, q, english)
            hits = self.search(q, past.avoid, english)
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
            draft = self.drafter.draft(q, hits, data, notes=past.notes, **({"english": english} if english else {}))
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
        subjects = {d: self.titles[d] for d in subject_docs(english or q, self.titles)
                    if any(h.doc_id == d for h in hits)}
        verifier = Verifier(self.passages, zetaris_lookup(client) if client else None,
                            allowed_sql=[d.sql for d in data], allowed_uncited=FIXED_UNCITED,
                            allowed_passages=[h.passage_id for h in hits], titles=self.titles, laws=self.laws,
                            similarity=getattr(self.retriever, "similarities", None), question=english or q,
                            language=question_lang,
                            subjects=subjects)
        check = verifier.check_answer(draft).to_dict()
        # Only for English questions: the extractive drafter matches words, and
        # most sources are English, so for another language it finds the few
        # documents in that language whether or not they answer.
        if self.redraft and question_lang == lang.ENGLISH and self._all_claims_cut(check) \
                and not isinstance(self.drafter, ExtractiveDrafter):
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
                                           "cut": len(check["sentences"]) - kept})
        status = "held" if check["status"] == "held" else "awaiting_review"
        retrieval = [asdict(h) for h in hits]
        data_dicts = [d.to_dict() for d in data]
        self.store.save_record(qid, retrieval, data_dicts, draft, check, status)
        self.store.event(qid, "queued", {"status": status})
        return Record(qid, q, screened.flags, status, retrieval, data_dicts, draft, check)


def fuse(first: list[Hit], second: list[Hit], rrf_k: int = 60) -> list[Hit]:
    """Two rankings in one by reciprocal rank, each passage keeping its best
    BM25 score, so the weak-retrieval gate sees the English query's matches."""
    score: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in (first, second):
        for rank, h in enumerate(ranking, start=1):
            score[h.passage_id] = score.get(h.passage_id, 0.0) + 1.0 / (rrf_k + rank)
            old = best.get(h.passage_id)
            best[h.passage_id] = h if old is None else Hit(**{**asdict(h), "score": max(h.score, old.score),
                                                              "dense": max(h.dense, old.dense)})
    return [best[p] for p in sorted(score, key=lambda p: -score[p])]


_default: Pipeline | None = None


def answer(question: str) -> Record:
    """Answer through the default pipeline over build/passages.jsonl and build/dalil.db."""
    global _default
    if _default is None:
        _default = Pipeline.from_build()
    return _default.answer(question)
