# Working rules for this repo

## Search the literature before experimenting

When a technical question is hard — an unexplained failure mode, a choice between
methods, "is this number any good?" — **look it up before spending compute on it**.
Papers, journals, articles, vendor docs. Do not iterate through guesses at methods
and burn GPU hours rediscovering published results.

This rule exists because it was violated and then vindicated on 27 Aug 2026. Hours
went into diagnosing the building-underestimation failure and designing a β-NLL
reweighting fix, with a two-hour GPU run already committed to it. A single search
surfaced [HTC-DC Net](https://arxiv.org/abs/2309.16486), which named that failure as
the field's central known problem, showed loss reweighting is the wrong *class* of
fix, gave the method that works, and revealed that published DFC19 numbers are not
comparable to ours at all. See [docs/literature.md](docs/literature.md).

Practical notes:
- **Always read the evaluation protocol before comparing numbers.** Data splits are
  where incomparability hides — HTC-DC Net random-splits crops from two cities; we
  split by region, which is a strictly harder task.
- WebFetch's extractor usually fails on paper PDFs. `pymupdf` is in the venv:
  `page.get_text()` for prose, and for tables rendered as images,
  `page.get_pixmap(matrix=Matrix(4,4), clip=Rect(...))` then read the PNG.
- Set `PYTHONIOENCODING=utf-8` before any script that prints paper text — Windows
  consoles are cp1252 and Greek letters raise `UnicodeEncodeError`.

## Never quote a number without its protocol

Global RMSE alone is close to meaningless here: ground is ~74% of pixels and easy,
buildings are ~13% and carry over 90% of the squared error. Report per-class error
and share-of-squared-error alongside any headline figure. Oracle-affine numbers are
not deployable and must never be presented as results.

Development metrics come from `--split val`. The three evaluation protocols in use,
and the ~1.5 m offset between crop-wise and whole-tile scoring of the *same*
checkpoint, are tabulated in [docs/evaluation-protocol.md](docs/evaluation-protocol.md).
The test split has already been spent twice; do not score it again until the final
reported number.

## Never change GPU settings while a CUDA job is running

Applying MSI Afterburner settings resets the driver and invalidates live CUDA
contexts. It has already killed one run. Pause training first.
