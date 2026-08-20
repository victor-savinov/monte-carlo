"""Monte Carlo schedule estimator — the screen.

This file holds every widget and no arithmetic. Anything that computes
belongs in montecarlo/core and has tests.
"""
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from montecarlo.core import charts, dates, loader, mapping, simulate, stats
from montecarlo.core.export import summary_workbook
from montecarlo.core.validate import ERROR, prepare, strip_total_rows, validate

st.set_page_config(page_title="Schedule Estimator", layout="wide",
                   page_icon="🗓️")

ROLE_LABELS = {
    "task": "Task name",
    "optimistic": "Optimistic",
    "realistic": "Realistic",
    "pessimistic": "Pessimistic",
}
NONE_OPTION = "— none —"

# The same palette, type stack and status vocabulary as montecarlo.core.charts —
# the screen and the charts it produces are meant to read as one instrument.
BOARD = "#10151D"
BOARD_LINE = "#26303C"
PAPER = "#EDEFEF"
SURFACE = "#FFFFFF"
INK = "#10151D"
INK_SOFT = "#5B6773"
BOARDING = "#D98A2B"       # brand accent — masthead/button chrome, not a risk color
RECOMMENDED = "#4ECBA5"    # matches charts.GREEN_PEAK — the one tier worth acting on

# Plain-language read of each tier instead of a green/amber/red "how scared
# should I be" scale: P50 and P95 share the same neutral gray (both sit
# outside the range worth defending, just in opposite directions), and only
# P85 gets an accent color and a badge.
LEVEL_STYLE = {
    50: ("High risk of missing the deadline", INK_SOFT),
    85: ("Reliable buffer", RECOMMENDED),
    95: ("Overly conservative estimate", INK_SOFT),
}


def inject_style():
    """Load the type system and restyle Streamlit's own chrome.

    Everything here is presentational; it never reads or writes app state.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Archivo+Expanded:wght@800;900&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Public Sans', -apple-system, sans-serif;
        }}
        .stApp {{ background-color: {paper}; }}
        h1, h2, h3 {{ font-family: 'Public Sans', sans-serif; color: {ink}; }}
        code, .stCode, [data-testid="stMetricValue"] {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace !important;
        }}

        /* ---- masthead: the board casing ---- */
        .board-masthead {{
            background: {board};
            border-radius: 4px;
            padding: 28px 32px 24px;
            margin: 0 0 28px;
            border-bottom: 3px solid {boarding};
        }}
        .board-masthead__wordmark {{
            font-family: 'Archivo Expanded', sans-serif;
            font-weight: 900;
            font-size: 2.1rem;
            letter-spacing: 0.02em;
            color: #F4F5F3;
            text-transform: uppercase;
            line-height: 1.05;
            margin: 0;
        }}
        .board-masthead__wordmark span {{ color: {boarding}; }}
        .board-masthead__status {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            color: #9FB0BC;
            margin-top: 10px;
            text-transform: uppercase;
        }}

        /* ---- sidebar: the instrument's control panel ---- */
        [data-testid="stSidebar"] {{
            background-color: {board};
        }}
        [data-testid="stSidebar"] * {{ color: #E4E9EC; }}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            font-family: 'Archivo Expanded', sans-serif;
            text-transform: uppercase;
            font-size: 0.95rem;
            letter-spacing: 0.05em;
            color: #F4F5F3;
        }}
        [data-testid="stSidebar"] label p {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-size: 0.76rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: #B7C2CA;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"],
        [data-testid="stSidebar"] input {{
            background-color: #1A212B !important;
            border-color: {boardline} !important;
            color: #F4F5F3;
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
        }}
        [data-testid="stSidebar"] input::placeholder {{ color: #6E7B85; }}
        [data-testid="stSidebar"] svg {{ fill: #B7C2CA; }}
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            border-color: {boardline};
            background-color: #161C25;
        }}

        /* ---- primary action ---- */
        .stButton button[kind="primary"] {{
            background-color: {boarding};
            color: {board};
            font-family: 'Public Sans', sans-serif;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            border: none;
            border-radius: 3px;
        }}
        .stButton button[kind="primary"]:hover {{
            background-color: #C27A20;
            color: {board};
        }}

        /* ---- tabs ---- */
        .stTabs [data-baseweb="tab"] {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            text-transform: uppercase;
            font-size: 0.82rem;
            letter-spacing: 0.05em;
            color: {inksoft};
        }}
        .stTabs [aria-selected="true"] {{
            color: {ink} !important;
            border-bottom-color: {boarding} !important;
        }}

        /* ---- alerts ---- */
        [data-testid="stAlert"] {{
            border-radius: 3px;
            font-family: 'Public Sans', sans-serif;
        }}

        /* ---- board rows: the signature result display ---- */
        .board-row {{
            display: grid;
            grid-template-columns: 1fr auto auto;
            align-items: center;
            gap: 18px;
            background: {board};
            padding: 16px 22px;
            border-bottom: 1px solid {boardline};
        }}
        .board-row:first-child {{ border-radius: 4px 4px 0 0; }}
        .board-row:last-child {{ border-radius: 0 0 4px 4px; border-bottom: none; }}
        .board-row__status {{
            font-family: 'Archivo Expanded', sans-serif;
            font-weight: 800;
            font-size: 0.92rem;
            letter-spacing: 0.02em;
        }}
        .board-row__level {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-size: 0.78rem;
            color: #9FB0BC;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 3px;
        }}
        .board-row__badge {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(78, 203, 165, 0.16);
            color: {recommended};
            white-space: nowrap;
        }}
        .board-row__value {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-weight: 700;
            font-size: 1.7rem;
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .board-row__date {{
            font-family: 'IBM Plex Mono', 'PT Mono', monospace;
            font-size: 0.85rem;
            color: #9FB0BC;
            text-align: right;
            margin-top: 2px;
            letter-spacing: 0.03em;
        }}
        </style>
        """.format(board=BOARD, boardline=BOARD_LINE, paper=PAPER, ink=INK,
                  inksoft=INK_SOFT, boarding=BOARDING, recommended=RECOMMENDED),
        unsafe_allow_html=True,
    )


def render_masthead():
    """The dark title band that opens the page — the product's own casing.

    Built as one unbroken line: a blank line anywhere inside HTML handed to
    st.markdown makes Streamlit's renderer close the HTML block early and
    print whatever follows as literal text, so multi-line indented
    templates are a trap here.
    """
    st.markdown(
        '<div class="board-masthead">'
        '<p class="board-masthead__wordmark">MONTE CARLO <span>&middot;</span> SCHEDULE</p>'
        '<p class="board-masthead__status">THREE-POINT ESTIMATES IN '
        '&middot; A DATE YOU CAN DEFEND OUT</p>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_board_rows(pctls, date_labels, unit_label):
    """The three confidence tiers, styled as rows on a departures board.

    Each row spells out what its P-level actually means — "P85" alone
    tells a reader nothing — and only the recommended tier (P85) carries
    a badge, so the eye lands on the one number worth taking to a manager
    instead of three equally-weighted options.

    See render_masthead's note: every row is assembled on one line so
    joining several of them can never introduce a blank-line break.
    """
    rows = []
    for level in sorted(pctls):
        status, color = LEVEL_STYLE.get(
            level, ("P{0}".format(level), INK_SOFT))
        badge = ('<div class="board-row__badge">Recommended</div>'
                 if level == 85 else '<div></div>')
        rows.append(
            '<div class="board-row" style="border-left: 4px solid {color}">'
            '<div><div class="board-row__status" style="color: {color}">'
            '{status}</div>'
            '<div class="board-row__level">P{level} &middot; {level}% likely'
            '</div></div>'
            '{badge}'
            '<div><div class="board-row__value" style="color: {color}">'
            '{value:.0f} <span style="font-size:0.95rem">{unit}</span></div>'
            '<div class="board-row__date">{date}</div></div>'
            '</div>'.format(color=color, status=status, level=level,
                            badge=badge, value=pctls[level],
                            unit=unit_label.upper(),
                            date=date_labels[level].upper())
        )
    st.markdown('<div>' + "".join(rows) + '</div>', unsafe_allow_html=True)


def sidebar_settings():
    """Collect every run setting. Returns a plain dict."""
    st.sidebar.header("Settings")
    unit = st.sidebar.selectbox("Estimates are in", dates.UNITS, index=0)
    hours_per_day = dates.DEFAULT_HOURS_PER_DAY
    if unit == dates.HOURS:
        hours_per_day = st.sidebar.number_input(
            "Hours per working day", 1.0, 24.0, dates.DEFAULT_HOURS_PER_DAY, 0.5)
    days_per_week = st.sidebar.selectbox("Working week", [5, 7], index=0,
                                         format_func=lambda d: "{0} days".format(d))
    start = st.sidebar.date_input("Start date", value=date.today())
    rho = st.sidebar.slider(
        "Correlation between tasks", 0.0, 0.9, 0.3, 0.05,
        help="How much tasks slip together. 0 means fully independent, which "
             "makes any forecast look more precise than it is.",
    )
    target = st.sidebar.date_input("Target date (optional)", value=None)
    with st.sidebar.expander("Advanced"):
        iterations = st.number_input("Iterations", 1000, 200_000,
                                     simulate.DEFAULT_ITERATIONS, 1000)
        seed = st.number_input("Random seed", 0, 10**9, simulate.DEFAULT_SEED)
    return {
        "unit": unit, "hours_per_day": hours_per_day,
        "days_per_week": days_per_week, "start": start,
        "rho": rho, "target": target, "iterations": int(iterations),
        "seed": int(seed),
    }


def mapping_controls(columns):
    """Draw one dropdown per role, pre-filled with the automatic guess."""
    guesses = mapping.guess_mapping(columns)
    chosen = {}
    grid = st.columns(len(mapping.ROLES))
    for column_box, role in zip(grid, mapping.ROLES):
        options = [NONE_OPTION] + list(columns)
        guess = guesses[role]
        index = options.index(guess.column) if guess.column in options else 0
        label = ROLE_LABELS[role]
        if guess.confidence == "fuzzy":
            label += " ⚠"
        selection = column_box.selectbox(label, options, index=index,
                                         key="map_" + role)
        chosen[role] = None if selection == NONE_OPTION else selection
    if any(g.confidence == "fuzzy" for g in guesses.values()):
        st.caption("⚠ marks a column matched by similarity. Please confirm it.")
    return chosen


def show_issues(issues):
    """Render the validation report. Returns True when the run is blocked."""
    if not issues:
        return False
    table = pd.DataFrame(
        [{"Severity": i.severity, "Row": i.row or "", "Column": i.column or "",
          "Problem": i.message} for i in issues]
    )
    blocking = any(issue.severity == ERROR for issue in issues)
    if blocking:
        st.error("{0} problem(s) must be fixed before running.".format(
            sum(1 for i in issues if i.severity == ERROR)))
    else:
        st.warning("{0} warning(s). You can still run.".format(len(issues)))
    st.dataframe(table, use_container_width=True, hide_index=True)
    return blocking


def show_results(totals, settings, baseline_days, unit_label, prepared):
    """Percentiles, dates, the plan comparison and the two charts."""
    pctls = stats.percentiles(totals)
    date_labels = {
        level: dates.to_date(value, settings["start"],
                             settings["days_per_week"]).strftime("%d %b %Y")
        for level, value in pctls.items()
    }

    render_board_rows(pctls, date_labels, unit_label)

    plan_probability = stats.probability_of(totals, baseline_days)
    plan_date = dates.to_date(baseline_days, settings["start"],
                              settings["days_per_week"])
    st.info(
        "Your plan of {0:.0f} {1} lands on {2} — that's close to a coin "
        "flip, succeeding in {3:.0f}% of runs. {4:.0f} {1} (landing on {5}) "
        "gives a reliable buffer at 85% likely, without tipping into an "
        "overly conservative estimate.".format(
            baseline_days, unit_label, plan_date.strftime("%d %b %Y"),
            plan_probability, pctls[85], date_labels[85])
    )

    if settings["target"]:
        target_days = _days_until(settings["target"], settings)
        st.info("Your target of {0} succeeds in {1:.0f}% of runs.".format(
            settings["target"].strftime("%d %b %Y"),
            stats.probability_of(totals, target_days)))

    distribution, curve = st.tabs(["Distribution", "S-curve"])
    with distribution:
        figure = charts.histogram(totals, pctls, baseline=baseline_days,
                                  baseline_probability=plan_probability,
                                  unit_label=unit_label)
        st.pyplot(figure)
        st.download_button("Download PNG", charts.figure_to_png_bytes(figure),
                           "distribution.png", "image/png")
    with curve:
        figure = charts.s_curve(totals, pctls, date_labels=date_labels,
                                unit_label=unit_label)
        st.pyplot(figure)
        st.download_button("Download PNG", charts.figure_to_png_bytes(figure),
                           "s_curve.png", "image/png")

    st.download_button(
        "Export summary .xlsx",
        summary_workbook(
            prepared, pctls, date_labels,
            {"unit": settings["unit"],
             "days_per_week": settings["days_per_week"],
             "start": settings["start"].strftime("%d %b %Y"),
             "correlation": settings["rho"],
             "iterations": settings["iterations"],
             "seed": settings["seed"]},
            baseline_days, plan_probability,
        ),
        "schedule_summary.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _days_until(target, settings):
    """Working days between the start date and a target date."""
    day = settings["start"]
    counted = 0
    while day <= target:
        if settings["days_per_week"] == 7 or day.weekday() < 5:
            counted += 1
        day += timedelta(days=1)
    return counted


inject_style()
render_masthead()

settings = sidebar_settings()
uploaded = st.file_uploader("Excel file with task estimates",
                            type=["xlsx", "xls"])

if uploaded is None:
    st.info("Upload a file to begin. A sample lives in sample_data/.")
    st.stop()

try:
    uploaded.seek(0)
    names = loader.sheet_names(uploaded)
    sheet = names[0] if len(names) == 1 else st.selectbox("Sheet", names)
    uploaded.seek(0)
    raw = loader.raw_preview(uploaded, sheet)
except loader.LoaderError as error:
    st.error("That file could not be read as a workbook. Supported formats "
             "are .xlsx and .xls.")
    with st.expander("Details"):
        st.code(str(error))
    st.stop()

detected_header_row = mapping.detect_header_row(raw.values.tolist()) + 1

with st.expander("Sheet preview and header row",
                 expanded=(detected_header_row != 1)):
    st.caption("If your file has title rows, a logo, or a merged section "
               "title above the table, point this at the row that actually "
               "holds the column names (min / real / max, and similar).")
    if detected_header_row != 1:
        st.caption("Detected the header at row {0} — change it if that's "
                   "wrong.".format(detected_header_row))
    st.dataframe(raw, use_container_width=True, hide_index=True)
    header_row_display = st.number_input(
        "Row number with the column headers", min_value=1,
        max_value=max(len(raw), 1), value=detected_header_row)

try:
    uploaded.seek(0)
    df = loader.read_sheet(uploaded, sheet, header_row=header_row_display - 1)
except loader.LoaderError as error:
    st.error("That file could not be read as a workbook. Supported formats "
             "are .xlsx and .xls.")
    with st.expander("Details"):
        st.code(str(error))
    st.stop()

st.subheader("Map the columns")
chosen = mapping_controls(list(df.columns))

df, dropped_totals = strip_total_rows(df, chosen)
if dropped_totals:
    st.info("Ignored {0} total/summary row(s): {1}".format(
        len(dropped_totals), ", ".join(dropped_totals)))

st.dataframe(df.head(6), use_container_width=True, hide_index=True)

issues = validate(df, chosen)
blocked = show_issues(issues)
sort_estimates = False
if any("out of order" in issue.message for issue in issues):
    sort_estimates = st.checkbox(
        "Sort the three estimates on out-of-order rows", value=True)

if blocked:
    st.stop()

if st.button("Run {0:,} simulations".format(settings["iterations"]),
             type="primary"):
    try:
        with st.spinner("Simulating…"):
            data = prepare(df, chosen, sort_three_point=sort_estimates)
            to_days = dates.to_working_days
            unit_args = (settings["unit"], settings["days_per_week"], settings["hours_per_day"])
            o = [to_days(v, *unit_args) for v in data.o]
            m = [to_days(v, *unit_args) for v in data.m]
            p = [to_days(v, *unit_args) for v in data.p]
            totals = simulate.simulate(o, m, p, streams=data.streams,
                                       rho=settings["rho"],
                                       n_iterations=settings["iterations"],
                                       seed=settings["seed"])
            baseline = stats.deterministic_baseline(m, data.streams)
        show_results(totals, settings, baseline, "working days", data)
    except Exception as error:  # the screen is the last line of defence
        st.error("The simulation could not finish. Check the column mapping "
                 "and the estimates, then try again.")
        with st.expander("Details"):
            st.exception(error)
