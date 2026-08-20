# Monte Carlo Schedule Estimator

Turns three-point task estimates into a date you can defend.

A plan that adds up every "realistic" estimate produces a date with roughly
a coin-flip chance of being met — often much worse. This tool simulates the
project 10,000 times and reports the duration you would hit in 50%, 85% and
95% of those runs, with charts built for a management deck.

It is hosted on https://monte-carlo-project-schedule-estimator.streamlit.app/

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/streamlit run app.py
```

Then open the address it prints and upload your Excel file. A sample is in
`sample_data/tasks_sample.xlsx`.

## Your Excel file

Four columns are required. The names do not have to match — the tool
recognises common variants in English and Russian, and you confirm its
guess on screen.

| Role | Recognised names include |
|------|--------------------------|
| Task name | Task, Activity, Work package, Задача |
| Optimistic | Optimistic, Best case, Min, Оптимистичная |
| Realistic | Realistic, Most likely, Expected, Реалистичная |
| Pessimistic | Pessimistic, Worst case, Max, Пессимистичная |

If your sheet has title rows or a logo above the real table, expand
**"Sheet preview and header row"** and tell it which row actually holds the
column names.

Rows whose task name reads like a spreadsheet total — "Total", "Grand
Total", "Итого", "Всего", "Сумма" — are recognised and skipped
automatically; the app tells you which rows it ignored.

## Settings that matter

- **Estimates are in** — days, weeks, or hours. Hours convert using the
  "Hours per working day" setting (default 8).
- **Correlation** — how much tasks slip together. 0 assumes every task is
  independent, which makes the forecast look far more precise than it is.
  The default of 0.3 is a conservative middle.
- **Working week** — 5 or 7 days. Public holidays are not modelled.
- **Seed** — fixed, so the same file always gives the same numbers.

## Tests

```bash
.venv/bin/pytest
```

## Design documents

- Spec: `docs/superpowers/specs/2026-08-20-monte-carlo-schedule-estimator-design.md`
- Plan: `docs/superpowers/plans/2026-08-20-monte-carlo-schedule-estimator.md`
- One-pager: `docs/one-pager.html`
