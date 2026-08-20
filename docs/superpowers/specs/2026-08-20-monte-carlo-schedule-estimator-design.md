# Monte Carlo Project Schedule Estimator — Design

Date: 2026-08-20
Status: Approved for planning

## 1. Purpose

Project managers estimate each task three ways — optimistic, realistic,
pessimistic — and then report a single number to management. That single
number is almost always the sum of the realistic estimates, and it is
almost always wrong: it carries roughly a 50% chance of being met, and
nobody says so out loud.

This tool takes the same three-point estimates from an Excel file, runs
10,000 Monte Carlo simulations, and reports the schedule at three
confidence levels — P50, P85, P95 — together with two charts a manager
can paste straight into a deck.

Success criteria:

- A PM with an Excel file and no Python knowledge gets a chart in under
  two minutes.
- Column names in the source file do not have to match anything.
- The same file always produces the same numbers.

## 2. Scope

In scope:

- Excel upload (`.xlsx`, `.xls`) through a local Streamlit web app.
- Automatic column detection with manual override.
- Input validation with a readable problem report.
- 10,000-iteration Monte Carlo simulation using a triangular
  distribution per task.
- Optional parallel work streams.
- Optional correlation between task durations.
- P50 / P85 / P95 in the source file's own unit and in calendar dates.
- Deterministic baseline for comparison.
- Optional target date, with the probability of meeting it.
- Histogram and S-curve charts, downloadable as 300 dpi PNG.
- Summary export to Excel.

Out of scope for this version:

- Dependency graphs and critical path (CPM).
- Resource levelling, cost, or budget simulation.
- Sensitivity / tornado analysis (candidate for phase 2).
- Persistence, database, authentication, multi-user state.
- Deployment beyond `streamlit run app.py` on a laptop.

## 3. Modelling decisions

### 3.1 Why a triangular distribution

Three-point estimates map onto it directly: optimistic is the lower
bound, pessimistic the upper bound, realistic the mode. It requires no
extra parameter the PM would have to invent, and its bounds are hard,
which matches how people actually reason about a "best case". PERT-beta
is the common alternative; it is smoother but weights the mode more
heavily, which understates the tail. For a tool whose whole point is to
expose the tail, triangular is the honest default.

### 3.2 Why not a full critical path

Project duration is set by the longest path through the network, and at
points where paths converge, a delay on any incoming path delays the
project — the merge bias. A full CPM simulation models this exactly, but
it requires a predecessor column, topological sorting, cycle detection,
and a schedule the PM has actually maintained. That is a different, much
larger tool.

The compromise: an optional `Stream` column. Tasks sharing a stream run
in sequence; streams run in parallel with each other.

    total = max( sum(stream_1), sum(stream_2), ... )

With no `Stream` column mapped, the model degrades to a plain sum, which
is the correct answer for a sequential project and an honest, clearly
stated approximation otherwise.

### 3.3 Why correlation is configurable

Simulation tools default to treating tasks as independent. Real projects
are not: one understaffed team, one slow client, or one optimistic
estimator affects many tasks at once. Assuming independence lets the
errors cancel out and produces a distribution that is too narrow — the
tool then understates risk while looking rigorous. Practitioners
typically apply 0.3–0.7 to tasks sharing a common risk driver.

The tool exposes a single correlation slider (default 0.3) applied
uniformly, implemented as a one-factor Gaussian copula:

    Z_i = sqrt(rho) * Z_common + sqrt(1 - rho) * Z_i
    U_i = Phi(Z_i)

`Z_common` is drawn once per iteration and shared by every task; `Z_i` is
drawn per task per iteration. This produces exactly the requested
correlation, and rho = 0 reduces to full independence. The UI states
plainly what the slider means, and the report records the value used.

### 3.4 Why the seed is fixed

The same file must produce the same numbers every time. A manager who
sees P85 = 142 days on Monday and 144 on Tuesday stops trusting the
tool. The seed is fixed at a constant and exposed as an advanced input
so a user can deliberately test stability.

## 4. Architecture

Thin UI, thick core. Streamlit knows nothing about the mathematics; the
core knows nothing about Streamlit. Every core function takes plain data
and returns plain data, so all of it is testable without a browser.

    montecarlo/
      app.py                 Streamlit UI: widgets and orchestration only
      core/
        loader.py            Excel bytes -> DataFrame
        mapping.py           header -> role auto-detection
        validate.py          DataFrame + mapping -> list of issues
        simulate.py          triangular sampling, correlation, aggregation
        stats.py             percentiles, target probability, baseline
        charts.py            matplotlib figures
        dates.py             duration -> calendar date
      tests/                 pytest over core; the UI is not unit-tested
      sample_data/
        tasks_sample.xlsx    deliberately messy headers, for the demo
      requirements.txt
      README.md

Note: the module is `dates.py`, not `calendar.py`, to avoid shadowing the
standard library.

Data flow is linear, with no feedback:

    upload -> load -> map -> validate -> simulate -> stats -> charts -> export

Each stage takes the previous stage's output and nothing else. A stage
can be replaced without touching its neighbours: swapping Streamlit for a
CLI touches only `app.py`; adding a lognormal distribution touches only
`simulate.py`.

## 5. Components

### 5.1 loader.py

    load_workbook(file) -> dict[str, DataFrame]
    read_sheet(file, sheet_name) -> DataFrame

Reads via pandas/openpyxl. If the workbook has several sheets, the UI
offers a sheet picker; otherwise the only sheet is used. Fully blank rows
and columns are dropped on read.

### 5.2 mapping.py

    RoleGuess = namedtuple("RoleGuess", "column confidence")
    guess_mapping(columns) -> dict[str, RoleGuess]

Five roles: `task`, `optimistic`, `realistic`, `pessimistic`, `stream`.
Only `stream` is optional.

Detection is three passes, stopping at the first hit:

1. Normalise each header — lowercase, strip, collapse whitespace, drop
   punctuation.
2. Exact match against a synonym dictionary per role. The dictionary
   covers English and Russian: `optimistic / best / best case / min /
   minimum / low / lo / o / оптимистичная / лучший`, and so on for each
   role.
3. Fuzzy match via `difflib.get_close_matches` at a 0.75 cutoff, to
   survive typos and suffixes such as `Optimistic (days)`.

Each role reports a confidence: `exact`, `fuzzy`, or `none`.

The guess never decides silently. It pre-fills five dropdowns in the UI,
and low-confidence guesses are flagged so the user's eye goes to them.

### 5.3 validate.py

    Issue = namedtuple("Issue", "severity row column message")
    validate(df, mapping) -> list[Issue]

Errors block simulation; warnings do not. Checks:

- Error: a duration cell is empty or non-numeric.
- Error: a duration is negative.
- Error: after coercion, no valid rows remain.
- Warning: `optimistic > realistic` or `realistic > pessimistic`. The UI
  offers to sort the three values per row, showing which rows change.
- Warning: `optimistic == pessimistic` (a task with zero uncertainty).
- Warning: a blank task name; the row is kept and labelled by index.
- Warning: duplicate task names.

Issues are returned as data, never raised. The UI renders them as a
table, so the user fixes the file once rather than discovering problems
one exception at a time.

### 5.4 simulate.py

    simulate(o, m, p, streams, rho, n=10_000, seed=SEED) -> ndarray

Takes three float arrays of length T, an optional stream label array,
correlation, iteration count, and seed. Returns an array of N project
totals.

Steps, all vectorised over an (N, T) matrix — no Python loop over
iterations:

1. Draw `Z_common` of shape (N, 1) and `Z` of shape (N, T) from a
   standard normal.
2. Mix them per section 3.3 and convert to uniforms with the normal CDF.
3. Invert the triangular CDF elementwise:

       Fc = (m - o) / (p - o)
       U <  Fc:  x = o + sqrt(U * (p - o) * (m - o))
       U >= Fc:  x = p - sqrt((1 - U) * (p - o) * (p - m))

   The degenerate case `o == p` returns `o` directly, guarding the
   division.
4. Aggregate: with no streams, sum across tasks. With streams, sum within
   each stream and take the row-wise maximum across streams.

`np.random.default_rng(seed)` is used, not the legacy global random
state.

### 5.5 stats.py

    percentiles(totals, levels=(50, 85, 95)) -> dict[int, float]
    deterministic_baseline(m, streams) -> float
    probability_of(totals, target) -> float

`deterministic_baseline` applies the same aggregation rule to the
realistic estimates alone. It is the number the PM would have reported
without this tool, and showing it next to P85 is the point of the whole
exercise.

### 5.6 dates.py

    to_working_days(duration, unit, days_per_week) -> float
    to_date(duration_days, start_date, days_per_week) -> date

Estimates in the Excel file may be in days or weeks; the unit is a
setting, not something the tool guesses, because guessing it wrong is
silent and catastrophic. `to_working_days` converts weeks to days using
the same `days_per_week` the calendar uses, so the two settings cannot
drift apart.

`days_per_week` is 5 or 7. For 5, calendar days are advanced skipping
Saturdays and Sundays. Public holidays are not modelled; the UI says so.

### 5.7 charts.py

    histogram(totals, pctls, baseline, ...) -> Figure
    s_curve(totals, pctls, ...) -> Figure

Both are matplotlib figures returned to the caller, never written to disk
by the core. Presentation defaults: 300 dpi, wide aspect, large type,
minimal chrome, no gridline clutter, direct labels on the P-lines rather
than a legend the reader must decode. The S-curve annotates each marker
with both duration and calendar date.

## 6. User interface

Single page, top to bottom, matching the data flow. Later sections stay
hidden until the earlier ones succeed, so there is never a control on
screen that cannot yet do anything.

1. **Upload** — file picker; sheet picker if the workbook has several.
2. **Map columns** — five dropdowns, pre-filled, low-confidence ones
   flagged. A preview of the first rows sits underneath so the user can
   see the mapping is right.
3. **Review issues** — shown only when validation returns something.
   Errors block the run; warnings offer a fix.
4. **Settings** (sidebar) — estimate unit (days or weeks), iterations,
   correlation, start date, days per week, optional target date, seed.
5. **Results** — P50/P85/P95 as large figures with dates, the
   deterministic baseline beside them, and the target probability if a
   target was given.
6. **Charts** — two tabs, histogram and S-curve, each with a download
   button.
7. **Export** — one Excel file: inputs, mapping, settings, percentiles.

## 7. Error handling

The rule is that the user always learns what to do next, and the app
never shows a traceback.

- Unreadable or corrupt file: a plain message naming the supported
  formats.
- Missing required role after mapping: the Run button stays disabled with
  the reason shown.
- Validation errors: the table from 5.3; the run is blocked.
- A run large enough to be slow (over 100,000 iterations): a spinner and
  a note that the default 10,000 is enough for stable percentiles.
- Anything unexpected: caught at the `app.py` boundary and shown as a
  short message with the detail behind an expander.

## 8. Testing

Pytest over `core`. The UI is exercised by hand.

- `mapping`: realistic header sets — English, Russian, with units in
  parentheses, with typos, with no stream column; and a set that must
  fail to match so a wrong guess is not silently accepted.
- `validate`: text in a numeric column, negative values, `o > p`, blank
  rows, blank names, duplicates, an all-invalid file.
- `simulate`: sampled mean and variance against the closed-form
  triangular moments; the degenerate `o == m == p` case returns a
  constant; a hand-computed two-stream example aggregates correctly;
  rho = 0.6 produces a wider spread than rho = 0 on the same inputs; the
  same seed returns identical arrays.
- `stats`: percentiles against a known array; the baseline against a
  hand-computed sum and a hand-computed two-stream max.
- `dates`: a 5-day week crossing a weekend and a month boundary.

## 9. Constraints

- Python 3.9 — no PEP 604 (`X | Y`) annotations, no `match`. Use
  `typing.Optional` and `typing.Dict`.
- Dependencies: streamlit, pandas, numpy, scipy, matplotlib, openpyxl.
  Nothing else without a reason.
- Runs entirely locally. No data leaves the machine.

## 10. Phase 2 candidates

Recorded so they are not smuggled into phase 1: sensitivity / tornado
analysis showing which task drives the spread; per-task or per-group
correlation instead of one global value; PERT-beta as a selectable
distribution; a CLI wrapper over the same core; direct PowerPoint export.
