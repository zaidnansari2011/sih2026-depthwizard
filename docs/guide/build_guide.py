# -*- coding: utf-8 -*-
"""Merge the deck annotation + the Q&A into one print-ready HTML."""
import re, pathlib

HERE = pathlib.Path(__file__).parent
DECK = (HERE / "deck-annotated.html").read_text(encoding="utf-8")
QA = (HERE / "depthwizard-qa.html").read_text(encoding="utf-8")


def split(doc):
    m = re.search(r"<style>(.*?)</style>", doc, re.S)
    return m.group(1), doc[m.end():]


deck_css, deck_body = split(DECK)
qa_css, qa_body = split(QA)


def strip_wrap(body):
    body = body.strip()
    body = re.sub(r'^<div class="wrap">', "", body).strip()
    if body.endswith("</div>"):
        body = body[: -len("</div>")].rstrip()
    return body


def drop(body, pat):
    return re.sub(pat, "", body, flags=re.S)


deck_body = strip_wrap(deck_body)
qa_body = strip_wrap(qa_body)

deck_body = drop(deck_body, r'<nav class="jump">.*?</nav>')
qa_body = drop(qa_body, r'<nav class="jump">.*?</nav>')
deck_body = drop(deck_body, r'<header class="top">.*?</header>')
qa_body = drop(qa_body, r'<header class="top">.*?</header>')

# ---- the "seven things to fix" box: rewrite as "already fixed" -------------
fixbox = re.search(r'<div class="fixes" id="fix">.*?</ol>\s*</div>', deck_body, re.S)
FIXBOX = fixbox.group(0) if fixbox else ""
deck_body = deck_body.replace(FIXBOX, "")
FIXBOX = (
    FIXBOX.replace('<div class="fixes" id="fix">', '<div class="fixed" id="fix">')
    .replace("<h2>Seven things to fix</h2>", "<h2>Seven errors found and corrected</h2>")
    .replace(
        "These are things a judge could catch.",
        "All seven are already fixed in the deck you are presenting. They are listed "
        "here because a judge may be holding an older copy, and because knowing what "
        "was wrong is how you avoid saying it out loud.",
    )
)

# ---- the cheat sheet and the glossary out of the QA doc --------------------
cheat = re.search(r'<div class="cheat".*?</ol>\s*</div>', qa_body, re.S)
CHEAT = cheat.group(0) if cheat else ""
qa_body = qa_body.replace(CHEAT, "")

GLOSS = re.search(r'<section id="gloss">.*?</section>', qa_body, re.S).group(0)
# the part heading below already names it; drop the doc's own duplicate heading
GLOSS = drop(GLOSS, r'\s*<h2 class="sec">Section 10</h2>')
GLOSS = drop(GLOSS, r'\s*<h3 class="sect">Glossary</h3>')
FOOT = re.search(r"<footer>.*?</footer>", qa_body, re.S).group(0)

# ---- carve the 43 Q&A blocks out by number --------------------------------
lines = qa_body.splitlines()
blocks, i = {}, 0
while i < len(lines):
    if lines[i].strip() == '<div class="qa">':
        j = i + 1
        while not lines[j].startswith("  </div>"):
            j += 1
        chunk = "\n".join(lines[i: j + 1])
        num = int(re.search(r'class="qn">Q(\d+)<', chunk).group(1))
        blocks[num] = chunk
        i = j + 1
    else:
        i += 1
assert len(blocks) == 43, len(blocks)

TIERS = [
    (
        "Before the questions start",
        "How to open, how to run the demo, and what to do if the internet fails. "
        "These are not judge questions. They are your own script.",
        [1, 2, 3],
    ),
    (
        "Level 1 &mdash; Easy",
        "Almost every judge opens with one of these. If you cannot answer them "
        "without thinking, nothing further matters.",
        [4, 6, 7, 8, 16, 17, 32, 33],
    ),
    (
        "Level 2 &mdash; Moderate",
        "Technical, but expected. A judge who has read the problem statement "
        "properly will ask most of these.",
        [5, 9, 10, 11, 13, 14, 18, 19, 20, 34, 36],
    ),
    (
        "Level 3 &mdash; Hard",
        "These go looking for the weak spot. Answer them by naming the weakness "
        "first, then saying what you did about it. Never let a judge find it "
        "before you have said it.",
        [12, 15, 21, 22, 23, 24, 26, 27, 29, 30, 35, 37],
    ),
    (
        "Level 4 &mdash; Hardest",
        "Questions designed to make you defend the whole idea. Every one of them "
        "has an honest answer, and the honest answer is stronger than a dodge.",
        [25, 28, 31, 38, 39, 40, 41, 42, 43],
    ),
]
assert sorted(n for _, _, ns in TIERS for n in ns) == list(range(1, 44))

out, seq = [], 0
for k, (title, note, nums) in enumerate(TIERS, 1):
    out.append('<section class="tier">')
    out.append('  <h2 class="sec">Part 2 &middot; %d of %d</h2>' % (k, len(TIERS)))
    out.append('  <h3 class="sect">%s</h3>' % title)
    out.append('  <p class="sec-note">%s</p>' % note)
    for n in nums:
        seq += 1
        out.append(
            re.sub(r'class="qn">Q\d+<', 'class="qn">Q%02d<' % seq, blocks[n], count=1)
        )
    out.append("</section>")
QA_SECTIONS = "\n".join(out)

PRINT_CSS = """
:root{color-scheme:light}
@page{size:A4;margin:16mm 15mm 14mm}
body{background:#fff;font-size:11.2px;line-height:1.62}
.wrap{max-width:none;padding:0}
h1{font-size:33px}
.shead h3,h3.sect{font-size:19px}
.el h4{font-size:14px}
.qt{font-size:14.5px}
.qn{font-size:10.5px}
table{font-size:10.4px;min-width:0}
th,td{padding:6px 9px}
.ti,.gi{font-size:10.8px}
.ti dt,.gi dt{font-size:10.5px}

.onslide{font-size:10.6px}
.say p,.trap p,.flag p{font-size:11.1px}
.el p,.el ul,.el ol,.a>p,.a ul,.a ol,.say p,.trap p,.flag p,.sec-note,
.standfirst,.onslide,.shead p,.fixed p.lede{max-width:none}
section.slide,section.tier{margin-bottom:24px}
.cover{height:246mm;display:flex;flex-direction:column;justify-content:flex-start;
  page-break-after:always;border-bottom:none;padding:0}
.cover .rule{height:4px;background:var(--ink);margin:0 0 30px}
.cover .eyebrow{font-size:11px}
.cover h1{font-size:46px;margin:0 0 16px;line-height:1.06}
.cover .standfirst{font-size:14.5px;max-width:none;color:var(--ink-2)}
.cover .cmeta{margin-top:auto;padding-top:22px;border-top:1px solid var(--rule);
  display:grid;grid-template-columns:repeat(2,1fr);gap:16px 26px;
  font:400 11px/1.5 var(--sans);color:var(--ink-2)}
.cover .cmeta b{display:block;color:var(--ink-3);font:600 9.5px/1 var(--mono);
  letter-spacing:.12em;text-transform:uppercase;margin-bottom:5px}
.parthead{page-break-before:always;border-bottom:3px solid var(--ink);
  padding-bottom:12px;margin:0 0 24px}
.parthead .pn{font:600 10.5px/1 var(--mono);letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.parthead h2{font:600 30px/1.1 var(--serif);margin:0 0 8px;letter-spacing:-.01em}
.parthead p{margin:0;color:var(--ink-2);font-size:11.6px;max-width:none}
.toc{display:grid;margin:0 0 8px}
.toc .row{display:grid;grid-template-columns:34px 1fr;gap:14px;
  padding:7px 0;border-bottom:1px solid var(--rule-soft);font-size:12px}
.toc .row span:first-child{font:600 10.5px/1.5 var(--mono);color:var(--accent)}
.toc .grp{font:600 10px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3);margin:20px 0 4px}
.toc .grp:first-child{margin-top:0}
.fixed{background:var(--accent-soft);border:1px solid var(--accent);
  padding:18px 20px;margin:0 0 22px}
.fixed h2{font:600 11.5px/1 var(--mono);letter-spacing:.12em;text-transform:uppercase;
  color:var(--accent-ink);margin:0 0 9px}
.fixed p.lede{margin:0 0 13px;color:var(--ink-2);font-size:11.3px}
.fixed ol{margin:0;padding-left:17px;display:grid;gap:8px}
.fixed li{font-size:11.3px;line-height:1.55}
.cheat{margin:0 0 22px;padding:18px 20px}
.cheat h2{font-size:11.5px;margin-bottom:14px}
.cheat li{font-size:11.5px;line-height:1.55}
.parthead,.qa,.el,.gi,.ti,tr,.say,.trap,.flag,.onslide,
.cheat li,.fixed li,.fixes li,.toc .row{break-inside:avoid;page-break-inside:avoid}
/* one deck slide per page makes this usable as a lookup table during Q&A;
   the first one rides along under the Part 1 divider rather than wasting a page */
section.slide{break-before:page;page-break-before:always}
.parthead + section.slide{break-before:auto;page-break-before:auto}
.shead,.parthead,h3.sect,h2.sec,.el h4{break-after:avoid;page-break-after:avoid}
.qa{padding-top:14px;margin-top:15px}
.a{padding-left:42px}
.tw{overflow:visible}
footer{margin-top:24px;font-size:10.5px}
footer p{max-width:none}
a{color:var(--accent-ink);text-decoration:none}

/* Chrome's paged renderer collapses a CSS grid that spans a page break, which
   flattens every cell to one word per line. Any grid that is a tall vertical
   stack becomes normal flow here; only rows that never fragment stay flex. */
.cheat ol,.fixed ol,.fixes ol,.terms,.gloss,.toc{display:block}
.cheat li{display:block;margin:0 0 10px;padding-left:38px;text-indent:-38px}
.fixed li,.fixes li{display:list-item;margin:0 0 9px}
.cheat li::before{display:inline-block;width:20px;padding-right:9px;margin-right:9px;
  text-align:right;border-right:1px solid var(--rule)}
.ti,.gi,.toc .row,.q{display:flex;gap:14px;align-items:baseline}
.ti dt{flex:0 0 148px}
.gi dt{flex:0 0 158px}
.ti dd,.gi dd,.toc .row span:last-child,.qt{flex:1 1 auto;min-width:0}
.toc .row span:first-child{flex:0 0 34px}
.qn{flex:0 0 38px}
"""

HTML = """<!doctype html>
<html lang="en" data-theme="light"><head><meta charset="utf-8">
<title>DepthWizard Deck Guide and Defence Notes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,400&display=swap">
<style>%(deck_css)s</style>
<style>%(qa_css)s</style>
<style>%(print_css)s</style>
</head><body><div class="wrap">

<header class="top cover">
  <div class="rule"></div>
  <div class="eyebrow">Smart India Hackathon 2026 &middot; SIH26175 &middot; Team Dev Up &middot; Team ID 47</div>
  <h1>DepthWizard<br>Deck Guide &amp; Defence Notes</h1>
  <p class="standfirst">Part 1 explains the presentation one slide at a time, and defines
  every technical word in plain English. Part 2 is forty-three questions a judge could ask,
  ordered from the easiest to the hardest, each with an answer you can say out loud.</p>
  <div class="cmeta">
    <div><b>Problem statement</b>SIH26175 &mdash; DepthWizard: Single-View Height Estimation and 3D Flythrough</div>
    <div><b>Organisation</b>ISRO &mdash; Space Applications Centre</div>
    <div><b>Theme &amp; category</b>Disaster Management &middot; Software</div>
    <div><b>Live site</b>project5.zaidansari.tech</div>
    <div><b>Deck</b>SIH2026-DevUp-SIH26175-DepthWizard-C &middot; 6 slides</div>
    <div><b>Compiled</b>6 September 2026</div>
  </div>
</header>

<section>
  <div class="parthead" style="page-break-before:auto">
    <div class="pn">Contents</div>
    <h2>What is in this document</h2>
    <p>Read Part 1 once, so you know what sits on each slide. Read Part 2 twice.</p>
  </div>
  <div class="toc">
    <div class="grp">Front matter</div>
    <div class="row"><span>&mdash;</span><span>Seven errors found and corrected</span></div>
    <div class="row"><span>&mdash;</span><span>The five things to remember</span></div>
    <div class="grp">Part 1 &mdash; The deck, slide by slide</div>
    <div class="row"><span>1</span><span>Title slide</span></div>
    <div class="row"><span>2</span><span>Problem statement</span></div>
    <div class="row"><span>3</span><span>Technical approach</span></div>
    <div class="row"><span>4</span><span>Feasibility and viability</span></div>
    <div class="row"><span>5</span><span>Impact and benefits</span></div>
    <div class="row"><span>6</span><span>Research and references</span></div>
    <div class="grp">Part 2 &mdash; Questions, easiest to hardest</div>
    <div class="row"><span>Q01</span><span>Before the questions start &mdash; your own script <b>(3)</b></span></div>
    <div class="row"><span>Q04</span><span>Level 1 &mdash; Easy <b>(8)</b></span></div>
    <div class="row"><span>Q12</span><span>Level 2 &mdash; Moderate <b>(11)</b></span></div>
    <div class="row"><span>Q23</span><span>Level 3 &mdash; Hard <b>(12)</b></span></div>
    <div class="row"><span>Q35</span><span>Level 4 &mdash; Hardest <b>(9)</b></span></div>
    <div class="grp">Back matter</div>
    <div class="row"><span>&mdash;</span><span>Glossary &mdash; every term, in plain English</span></div>
  </div>
</section>

%(fixbox)s

%(cheat)s

<div class="parthead">
  <div class="pn">Part 1</div>
  <h2>The deck, slide by slide</h2>
  <p>Every element on every slide: what it means, and the sentence to say when a judge
  points at it. If a judge points at anything on screen, the answer is in here.</p>
</div>

%(deck)s

<div class="parthead">
  <div class="pn">Part 2</div>
  <h2>Questions, easiest to hardest</h2>
  <p>Forty-three questions in four levels of difficulty. Level 1 is what everyone asks.
  Level 4 is what a judge asks when they want to find out whether you really built it.</p>
</div>

%(qa)s

<div class="parthead">
  <div class="pn">Reference</div>
  <h2>Glossary</h2>
  <p>Every abbreviation and technical term used in the deck or in the answers above.</p>
</div>

%(gloss)s

%(foot)s

</div></body></html>
""" % dict(
    deck_css=deck_css,
    qa_css=qa_css,
    print_css=PRINT_CSS,
    fixbox=FIXBOX,
    cheat=CHEAT,
    deck=deck_body,
    qa=QA_SECTIONS,
    gloss=GLOSS,
    foot=FOOT,
)

out_html = HERE / "depthwizard-guide.html"
out_html.write_text(HTML, encoding="utf-8")
print("wrote", out_html, len(HTML), "bytes;", len(blocks), "questions")
