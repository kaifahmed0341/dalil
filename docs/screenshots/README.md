# Reviewer screen, before and after

Captured on 25 November 2026 with headless Chromium (Playwright) at 1920 by 1080 and 1440 by 900. Every screen is the real app answering real questions: Dalil running on a laptop, fully offline, NVIDIA Nemotron 3 Nano 4B drafting and Nemotron Embed 1B searching, 1,024 passages, and no open-data service. Each capture set started from an empty store. The reviewer name is a placeholder.

The after set was taken on this branch before it merged `feat/itu-final-update`, except `after/arabic-*.png`, `after/hard-scope-articles-1920.png` and `after/s4-refusal-1440.png`, taken after the merge: there the writer answers Arabic questions in Arabic and a law's scope article is added to the search results.

The answers differ between the before and after sets because the model drafted them again. Some answers are wrong in ways a reviewer should catch, such as "The sponsor must be the UAE citizen", which quotes a rule about UAE citizens' relatives. They are left as the model wrote them.

## Before and after pairs

| State | Before | After |
|---|---|---|
| Main screen, an answer with citations | `before/main-*.png` | `after/main-*.png` |
| Citation 1 opened on its page | `before/citation-*.png` | `after/citation-*.png` |
| A sentence cut by the checker, with its reason | `before/cut-*.png` | `after/cut-*.png` |
| Audit trail | `before/audit-*.png` | `after/audit-*.png` |
| An Arabic question and answer, citation opened | `before/arabic-*.png` | `after/arabic-*.png` |

`*` is `1920` or `1440`.

## New in the after set

- `after/live-sending-1920.png`: the moment after asking, before the screener has recorded anything.
- `after/live-writer-running-1920.png`, `after/live-writer-running-1440.png`: the stage strip while the writer runs. The screener and searcher are done, with their measured times; the writer's time counts up from the searcher's recorded finish.
- The seven ITU scenarios from `scripts/run_itu_scenarios.py`, asked through the screen:
  - S1 `s1-fee-1920.png`: a fee the sources do not state.
  - S2 `s2-id-1920.png`, `s2-passport-1920.png`: personal data replaced before saving.
  - S3 `s3-law-1920.png`: an answer from the 2007 tenancy law, with the note about later amendments.
  - S4 `s4-refusal-1920.png`: a request to get around the rules, refused, nothing searched.
  - S5 `s5-gratuity-1920.png`: the gratuity question. The scenario runner corrupts a number to show the checker cutting it; the screen cannot do that, so this shows the plain answer.
  - S6 `s6-vague-1920.png`, then `s6-full-1920.png` (approved and sent).
  - S7 `s7-resign-1920.png`, `s7-rejected-audit-1920.png` (rejected with a note, the decision kept in the review memory), then `s7-followup-1920.png`.
- `hard-result6-1920.png`: the data protection question, with search result 6 (the federal law's Article 2) opened.
- `hard-scope-articles-1920.png`: the same question after the merge. The answer quotes Article 2's exclusion, and the two scope articles added ahead of the ranked results are labelled as added.
- `s4-refusal-1440.png`: the refusal and its follow-up in one box.
- `audit-raw-1920.png`: the raw record behind its control. `record-details-1920.png`: the question code and status codes behind theirs.
- `arabic-2-1920.png`: an Arabic question whose answer came back in English, a known limit of the writer.
- `zoom125-1152.png`: a 1440 by 900 laptop at 125% browser zoom.
- `keyboard-focus-1920.png`: the focus ring while tabbing through the queue.
