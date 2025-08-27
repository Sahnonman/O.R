# app.py
# Streamlit app for Alrugaib monthly optimization (Company vs 3PL)
# - Uses ONLY pandas + PuLP (CBC). No Gurobi required.
# - UI to upload Excel, pick month, choose integer/continuous trips
# - Builds and solves an LP/ILP with PuLP/CBC
# - Outputs Summary, Per Destination, Per Vehicle (summary/detail), Cost Breakdown
# - Daily Distribution (avg/day) tables + Narrative text
# - Exports a multi-sheet Excel report

import io
from typing import Dict, List

import pandas as pd
import streamlit as st

# ==== solver (PuLP/CBC) ====
try:
    import pulp
    from pulp import (
        LpProblem, LpMinimize, LpVariable, lpSum, LpStatus,
        LpInteger, LpContinuous, PULP_CBC_CMD
    )
except Exception as e:
    raise RuntimeError("PuLP is required. Install with: pip install pulp") from e


# ---------------------------
# Helpers
# ---------------------------
def _norm(s):
    return "" if s is None else "".join(str(s).lower().split())

def _clean_name(x):
    if pd.isna(x):
        return None
    return str(x).strip().replace(" ", "").replace("_", "-").upper()

def _resolve(xls_obj: pd.ExcelFile, keyword: str):
    key = _norm(keyword)
    for s in xls_obj.sheet_names:
        if _norm(s).find(key) != -1:
            return s
    return None

def _read(xls_or_path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xls_or_path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False, regex=True)]
    return df

def _month_val(row: pd.Series, m: int):
    val = row.get(str(m), pd.NA)
    if pd.isna(val):
        val = row.get(m, pd.NA)
    return pd.to_numeric(val, errors="coerce")

def _col_by_substring(df: pd.DataFrame, key: str):
    key_l = key.lower()
    for c in df.columns:
        if key_l in str(c).lower():
            return c
    return None


# ---------------------------
# Model builder/solver (PuLP)
# ---------------------------
def build_and_solve(
    duration_df: pd.DataFrame,
    tlimit_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    min_demand_df: pd.DataFrame,
    month_num: int,
    use_integer: bool
):
    # --- Normalize/clean ---
    duration_df = duration_df.copy()
    duration_df["Destination"] = duration_df["Destination"].apply(_clean_name)
    dur_col = _col_by_substring(duration_df, "duration")
    if dur_col is None:
        raise ValueError("Duration sheet must include a 'Duration (hr)' column.")
    duration_df[dur_col] = pd.to_numeric(duration_df[dur_col], errors="coerce").fillna(0.0)

    costs_df = costs_df.copy()
    costs_df["Destination"] = costs_df["Destination"].apply(_clean_name)
    a_col = _col_by_substring(costs_df, "alrugaib") or _col_by_substring(costs_df, "in-house") or _col_by_substring(costs_df, "company")
    if a_col is None:
        raise ValueError("Costs sheet must include 'Alrugaib Vehicles' (or Company/In-house) column.")
    costs_df[a_col] = pd.to_numeric(costs_df[a_col], errors="coerce").fillna(0.0)

    threepl_col = None
    for k in ["3PL", "threepl", "3 pl"]:
        c = _col_by_substring(costs_df, k)
        if c is not None:
            threepl_col = c
            break
    if threepl_col is None:
        raise ValueError("Costs sheet must include a '3PL' column.")
    costs_df["_3PL_raw"] = pd.to_numeric(costs_df[threepl_col], errors="coerce")

    monthly_df = monthly_df.copy()
    monthly_df["Destination"] = monthly_df["Destination"].apply(_clean_name)

    min_demand_df = min_demand_df.copy()
    md_col = None
    for c in min_demand_df.columns:
        if _norm(c).startswith("destination/month"):
            md_col = c
            break
    if md_col is None:
        md_col = _col_by_substring(min_demand_df, "destination")
    min_demand_df[md_col] = min_demand_df[md_col].apply(_clean_name)

    # --- DEST & maps ---
    DEST = duration_df["Destination"].dropna().unique().tolist()
    duration_hr = dict(zip(duration_df["Destination"], duration_df[dur_col]))

    # --- Time limit row ---
    month_col = _col_by_substring(tlimit_df, "month")
    days_col  = _col_by_substring(tlimit_df, "days")
    hours_col = _col_by_substring(tlimit_df, "hour")
    if month_col is None or days_col is None or hours_col is None:
        raise ValueError("Time Limit sheet must have columns: Month, Days, Hours' limit.")
    row_m = tlimit_df.loc[tlimit_df[month_col] == month_num]
    if row_m.empty:
        raise ValueError(f"Month {month_num} not found in Time Limit sheet.")
    row_m = row_m.iloc[0]
    DAYS_M = int(pd.to_numeric(row_m[days_col], errors="coerce"))
    HOURS_LIMIT = float(pd.to_numeric(row_m[hours_col], errors="coerce"))

    # --- Vehicles ---
    veh_col = vehicles_df.columns[0]
    vdf = vehicles_df.copy()
    vdf[veh_col] = vdf[veh_col].astype(str).str.strip()

    vehicle_count = None
    for c in vdf.columns:
        if str(c).strip().isdigit():
            vehicle_count = int(str(c).strip()); break
    is_3pl = vdf[veh_col].str.lower().eq("3pl")
    explicit_names = [v for v in vdf.loc[~is_3pl, veh_col] if v and not v.isdigit()]
    if explicit_names:
        KA = list(dict.fromkeys(explicit_names))
    elif vehicle_count:
        KA = [f"Veh_{i+1}" for i in range(vehicle_count)]
    else:
        KA = []

    # --- Costs maps & NO_3PL ---
    NO_3PL_DESTS = set(costs_df.loc[costs_df["_3PL_raw"].isna(), "Destination"])
    costA = dict(zip(costs_df["Destination"], costs_df[a_col]))
    cost3 = {d: (None if pd.isna(v) else float(v))
             for d, v in zip(costs_df["Destination"], costs_df["_3PL_raw"])}

    cost_dest = set(costs_df["Destination"])
    for d in DEST:
        if d not in cost_dest:
            NO_3PL_DESTS.add(d)
            costA.setdefault(d, 0.0)
            if d not in cost3:
                cost3[d] = None

    # --- Demand maps aligned to DEST ---
    monthly_demand = {r["Destination"]: float(_month_val(r, month_num) or 0.0)
                      for _, r in monthly_df.iterrows()}
    monthly_demand = {d: float(monthly_demand.get(d, 0.0)) for d in DEST}

    # --- Min monthly demand ---
    min_month_by_dest = {}
    for _, r in min_demand_df.iterrows():
        dest = r[md_col]
        val = _month_val(r, month_num)
        if not pd.isna(val):
            min_month_by_dest[dest] = float(val)
    min_month_by_dest = {d: float(min_month_by_dest.get(d, 0.0)) for d in DEST}

    # --- Build PuLP model ---
    cat = LpInteger if use_integer else LpContinuous
    prob = LpProblem("AlrugaibMonthly_NoDays", LpMinimize)

    zA = {}
    if len(KA) > 0:
        for d in DEST:
            zA[d] = {}
            for k in KA:
                zA[d][k] = LpVariable(f"zA__{d}__{k}", lowBound=0, cat=cat)
    z3 = {d: LpVariable(f"z3PL__{d}", lowBound=0, cat=cat) for d in DEST}

    # Objective
    obj_terms = []
    if len(KA) > 0:
        for d in DEST:
            for k in KA:
                obj_terms.append(float(costA.get(d, 0.0)) * zA[d][k])
    for d in DEST:
        c3 = 0.0 if cost3.get(d) is None else float(cost3[d])
        obj_terms.append(c3 * z3[d])
    prob += lpSum(obj_terms)

    # Hours per vehicle
    if len(KA) > 0:
        for k in KA:
            prob += lpSum(float(duration_hr.get(d, 0.0)) * zA[d][k] for d in DEST) <= HOURS_LIMIT

    # Demand + min-demand
    for d in DEST:
        lhs = (lpSum(zA[d][k] for k in KA) if len(KA) > 0 else 0) + z3[d]
        prob += lhs >= monthly_demand.get(d, 0.0)

    for d, v in min_month_by_dest.items():
        if v > 0:
            lhs = (lpSum(zA[d][k] for k in KA) if len(KA) > 0 else 0) + z3[d]
            prob += lhs >= v

    # Forbid 3PL
    for d in NO_3PL_DESTS:
        if d in DEST:
            prob += z3[d] == 0

    prob.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus[prob.status]

    zA_out = {d: {k: (zA[d][k].value() if len(KA) > 0 else 0.0) for k in KA} for d in DEST}
    z3_out = {d: z3[d].value() for d in DEST}
    objective = float(pulp.value(prob.objective)) if status == "Optimal" else float("nan")

    outputs = _build_outputs(
        DEST, KA, duration_hr, DAYS_M, HOURS_LIMIT,
        monthly_demand, min_month_by_dest, costA, cost3, zA_out, z3_out, objective
    )

    diagnostics = {
        "missing_in_costs": sorted(list(set(DEST) - cost_dest))[:20],
        "missing_in_monthly": sorted(list(set(DEST) - set(monthly_df["Destination"])))[:20],
        "extra_in_monthly": sorted(list(set(monthly_df["Destination"]) - set(DEST)))[:20],
    }

    meta = dict(
        month=month_num,
        days=DAYS_M,
        hours_limit=HOURS_LIMIT,
        vehicles=len(KA),
        solver="cbc",
        status=status
    )
    return outputs, diagnostics, meta


def _build_outputs(
    DEST, KA, duration_hr, DAYS_M, HOURS_LIMIT,
    monthly_demand, min_month_by_dest, costA, cost3, zA, z3, objective
):
    # Per Destination
    rows_dest = []
    for d in DEST:
        trips_A = sum(zA[d].values()) if len(KA) > 0 else 0.0
        trips_3 = z3[d]
        cA = float(costA.get(d, 0.0))
        c3 = 0.0 if cost3.get(d) is None else float(cost3.get(d, 0.0))
        dur = float(duration_hr.get(d, 0.0))
        rows_dest.append(dict(
            Destination=d,
            **{"Duration (hr)": dur},
            **{"Demand (monthly)": float(monthly_demand.get(d, 0.0))},
            **{"Min. monthly": float(min_month_by_dest.get(d, 0.0))},
            **{"In-house trips": trips_A},
            **{"3PL trips": trips_3},
            **{"In-house cost": cA * trips_A},
            **{"3PL cost": c3 * trips_3},
            **{"Total cost (dest)": cA * trips_A + c3 * trips_3},
        ))
    df_dest = pd.DataFrame(rows_dest).sort_values("Destination")

    # Per Vehicle
    rows_veh_sum, rows_veh_det = [], []
    if len(KA) > 0:
        for k in KA:
            hrs_used, trips_tot = 0.0, 0.0
            for d in DEST:
                v = zA[d][k]
                if v and v > 1e-6:
                    rows_veh_det.append(dict(
                        Vehicle=k,
                        Destination=d,
                        Trips=v,
                        Hours=float(duration_hr.get(d, 0.0)) * v,
                        **{"Cost (in-house)": float(costA.get(d, 0.0)) * v}
                    ))
                hrs_used += float(duration_hr.get(d, 0.0)) * v
                trips_tot += v
            util = (hrs_used / HOURS_LIMIT) if HOURS_LIMIT > 0 else 0.0
            rows_veh_sum.append(dict(
                Vehicle=k,
                **{"Trips (total)": trips_tot},
                **{"Hours used": hrs_used},
                **{"Hours limit": HOURS_LIMIT},
                **{"Utilization": util},
            ))
    df_veh_sum = pd.DataFrame(rows_veh_sum).sort_values("Vehicle") if rows_veh_sum else pd.DataFrame(
        columns=["Vehicle", "Trips (total)", "Hours used", "Hours limit", "Utilization"])
    df_veh_det = pd.DataFrame(rows_veh_det).sort_values(["Vehicle", "Destination"]) if rows_veh_det else pd.DataFrame(
        columns=["Vehicle", "Destination", "Trips", "Hours", "Cost (in-house)"])

    # Summary
    total_cost_inhouse = df_dest["In-house cost"].sum()
    total_cost_3pl = df_dest["3PL cost"].sum()
    total_cost = total_cost_inhouse + total_cost_3pl
    trips_A_total = df_dest["In-house trips"].sum()
    trips_3_total = df_dest["3PL trips"].sum()

    df_summary = pd.DataFrame([dict(
        **{"In-house trips (total)": trips_A_total},
        **{"3PL trips (total)": trips_3_total},
        **{"In-house cost (total)": total_cost_inhouse},
        **{"3PL cost (total)": total_cost_3pl},
        **{"Total cost": total_cost},
        **{"Objective": objective},
    )])

    # Cost Breakdown (segregated lines)
    df_cost_breakdown = pd.DataFrame([
        {"Line": "Company (In-house)", "Trips": trips_A_total, "Cost": total_cost_inhouse},
        {"Line": "3PL", "Trips": trips_3_total, "Cost": total_cost_3pl},
    ])

    # Daily Distribution (avg/day)
    rows_daily = []
    if not df_veh_det.empty:
        for _, r in df_veh_det.iterrows():
            day_trips = r["Trips"] / max(DAYS_M, 1)
            day_hours = r["Hours"] / max(DAYS_M, 1)
            rows_daily.append({
                "Vehicle": r["Vehicle"],
                "Destination": r["Destination"],
                "Avg Trips / Day": day_trips,
                "Avg Hours / Day": day_hours
            })
    df_daily_per_vehicle = pd.DataFrame(rows_daily).sort_values(["Vehicle", "Destination"]) if rows_daily else \
        pd.DataFrame(columns=["Vehicle", "Destination", "Avg Trips / Day", "Avg Hours / Day"])

    rows_daily_sum = []
    for d in DEST:
        a_trips_day = (df_dest.loc[df_dest["Destination"] == d, "In-house trips"].sum()) / max(DAYS_M, 1)
        p_trips_day = (df_dest.loc[df_dest["Destination"] == d, "3PL trips"].sum()) / max(DAYS_M, 1)
        dur = float(duration_hr.get(d, 0.0))
        rows_daily_sum.append({
            "Destination": d,
            "Company Trips/Day": a_trips_day,
            "3PL Trips/Day": p_trips_day,
            "Company Hours/Day": a_trips_day * dur,
            "3PL Hours/Day": p_trips_day * dur
        })
    df_daily_summary = pd.DataFrame(rows_daily_sum).sort_values("Destination")

    # Narrative
    util_avg = (df_veh_sum["Utilization"].mean() if not df_veh_sum.empty else 0.0)
    trips_day_total = (trips_A_total + trips_3_total) / max(DAYS_M, 1)
    trips_day_company = trips_A_total / max(DAYS_M, 1)
    trips_day_3pl = trips_3_total / max(DAYS_M, 1)

    narrative = f"""
Daily Fleet Distribution – ({DAYS_M} days)

We plan an average of {trips_day_total:.1f} trips/day across all destinations.
Of these, {trips_day_company:.1f} trips/day are covered by company vehicles and {trips_day_3pl:.1f} trips/day by 3PL.

Average vehicle utilization (hours used vs. monthly limit) is {util_avg:.1%}.
Use the 'Daily Summary (avg/day)' to see trips/day and hours/day by destination split between Company and 3PL,
and 'Daily Distribution (avg/day)' for per-vehicle details.

Note: Daily values are proportional averages derived from the monthly plan (not a strict calendar).
"""

    return dict(
        df_summary=df_summary,
        df_dest=df_dest,
        df_veh_sum=df_veh_sum,
        df_veh_det=df_veh_det,
        df_cost_breakdown=df_cost_breakdown,
        df_daily_per_vehicle=df_daily_per_vehicle,
        df_daily_summary=df_daily_summary,
        narrative=narrative.strip()
    )


def export_excel(outputs: Dict, KA: List[str]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        outputs["df_summary"].to_excel(writer, sheet_name="Summary", index=False)
        outputs["df_dest"].to_excel(writer, sheet_name="Per Destination", index=False)
        outputs["df_veh_sum"].to_excel(writer, sheet_name="Per Vehicle (summary)", index=False)
        outputs["df_veh_det"].to_excel(writer, sheet_name="Per Vehicle (detail)", index=False)
        outputs["df_cost_breakdown"].to_excel(writer, sheet_name="Cost Breakdown", index=False)
        outputs["df_daily_per_vehicle"].to_excel(writer, sheet_name="Daily Distribution (avg/day)", index=False)
        outputs["df_daily_summary"].to_excel(writer, sheet_name="Daily Summary (avg/day)", index=False)
        pd.DataFrame({"Narrative": [outputs["narrative"]]}).to_excel(writer, sheet_name="Narrative", index=False)

        # one sheet per vehicle
        df_veh_det = outputs["df_veh_det"]
        if len(KA) > 0 and not df_veh_det.empty:
            for k in KA:
                sub = df_veh_det[df_veh_det["Vehicle"] == k].copy()
                if sub.empty:
                    sub = pd.DataFrame([{"Vehicle": k, "Destination": "-", "Trips": 0, "Hours": 0, "Cost (in-house)": 0}])
                sub = sub.sort_values(["Trips", "Destination"], ascending=[False, True])
                safe_name = str(k)[:31]
                sub.to_excel(writer, sheet_name=safe_name, index=False)
    buf.seek(0)
    return buf.read()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Alrugaib Fleet Optimizer (pandas + PuLP)", layout="wide")
st.title("🚚 Alrugaib Monthly Fleet Optimizer — pandas + PuLP (no Gurobi)")

with st.sidebar:
    st.header("1) Data")
    up = st.file_uploader("Upload Excel (Trips and Demand-Data.xlsx)", type=["xlsx"])
    st.caption("Required sheets: Time Limit, Monthly Demand (Exp), Minimum Demand, Costs, Vehicles, Duration.")

    st.header("2) Settings")
    month_num = st.selectbox("Month", list(range(1, 13)), index=6, help="Choose month number (1..12)")
    use_integer = st.toggle("Integer trips", value=True, help="If off, allows fractional trips")

    run = st.button("Optimize", type="primary")

if up and run:
    try:
        xls = pd.ExcelFile(up)
        # Resolve sheet names (fuzzy)
        duration_name = _resolve(xls, "Duration")
        tlimit_name = _resolve(xls, "Time Limit")
        vehicles_name = _resolve(xls, "Vehicles")
        costs_name = _resolve(xls, "Costs")
        monthly_exp_name = _resolve(xls, "Monthly Demand (Exp)") or _resolve(xls, "Monthly Demand")
        min_demand_name = _resolve(xls, "Minimum Demand")

        missing = [n for n in [duration_name, tlimit_name, vehicles_name, costs_name, monthly_exp_name, min_demand_name] if n is None]
        if missing:
            st.error("Could not resolve all required sheets. Ensure names contain: Duration, Time Limit, Vehicles, Costs, Monthly Demand (Exp), Minimum Demand.")
            st.stop()

        duration_df = _read(xls, duration_name)
        tlimit_df   = _read(xls, tlimit_name)
        vehicles_df = _read(xls, vehicles_name)
        costs_df    = _read(xls, costs_name)
        monthly_df  = _read(xls, monthly_exp_name)
        min_demand_df = _read(xls, min_demand_name)

        outputs, diagnostics, meta = build_and_solve(
            duration_df, tlimit_df, vehicles_df, costs_df, monthly_df, min_demand_df,
            month_num, use_integer
        )

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        tot = float(outputs["df_summary"]["Total cost"].iloc[0])
        in_c = float(outputs["df_summary"]["In-house cost (total)"].iloc[0])
        p_c  = float(outputs["df_summary"]["3PL cost (total)"].iloc[0])
        a_tr = float(outputs["df_summary"]["In-house trips (total)"].iloc[0])
        p_tr = float(outputs["df_summary"]["3PL trips (total)"].iloc[0])

        c1.metric("Solver", "cbc")
        c2.metric("Status", meta["status"])
        c3.metric("Total Cost (SAR)", f"{tot:,.0f}")
        c4.metric("Vehicles", meta["vehicles"])
        st.caption(f"Month {meta['month']} • Days: {meta['days']} • Hours/vehicle limit: {meta['hours_limit']:.1f}")

        tabs = st.tabs(["Summary", "Cost Breakdown", "Per Destination", "Per Vehicle", "Daily (avg/day)", "Narrative", "Diagnostics", "Export"])

        with tabs[0]:
            st.subheader("Summary")
            st.dataframe(outputs["df_summary"], use_container_width=True)

        with tabs[1]:
            st.subheader("Cost Breakdown (Segregated Lines)")
            st.dataframe(outputs["df_cost_breakdown"], use_container_width=True)

        with tabs[2]:
            st.subheader("Per Destination")
            st.dataframe(outputs["df_dest"], use_container_width=True)

        with tabs[3]:
            st.subheader("Per Vehicle (summary)")
            st.dataframe(outputs["df_veh_sum"], use_container_width=True)
            st.divider()
            st.subheader("Per Vehicle (detail)")
            st.dataframe(outputs["df_veh_det"], use_container_width=True)

        with tabs[4]:
            st.subheader("Daily Distribution (avg/day) — per vehicle")
            st.dataframe(outputs["df_daily_per_vehicle"], use_container_width=True)
            st.divider()
            st.subheader("Daily Summary (avg/day) — per destination")
            st.dataframe(outputs["df_daily_summary"], use_container_width=True)

        with tabs[5]:
            st.subheader("Daily Fleet Distribution — Description")
            st.write(outputs["narrative"])

        with tabs[6]:
            st.subheader("Diagnostics (first 20 only)")
            c1, c2, c3 = st.columns(3)
            c1.write("In Duration but NOT in Costs")
            c1.code(", ".join(diagnostics["missing_in_costs"]) or "-")
            c2.write("In Duration but NOT in Monthly")
            c2.code(", ".join(diagnostics["missing_in_monthly"]) or "-")
            c3.write("In Monthly but NOT in Duration")
            c3.code(", ".join(diagnostics["extra_in_monthly"]) or "-")

        with tabs[7]:
            st.subheader("Export Report")
            KA = outputs["df_veh_sum"]["Vehicle"].tolist() if not outputs["df_veh_sum"].empty else []
            xl_bytes = export_excel(outputs, KA)
            st.download_button(
                label="⬇️ Download Excel Report",
                data=xl_bytes,
                file_name="Alrugaib_Monthly_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload your Excel, choose settings in the sidebar, then click **Optimize**.")
