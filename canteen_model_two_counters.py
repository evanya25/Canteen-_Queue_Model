"""
Canteen pre-ordering model (trace-driven discrete-event simulation).

USAGE
  python canteen_model.py                    # reads walkin_observations.csv (YOUR real data)
  python canteen_model.py --make-test-data   # writes SYNTHETIC_walkin_observations.csv
                                             # ONLY to test that the code runs. Never submit it.

INPUT CSV columns (see your template):
  Session_ID, Customer_ID, Arrival_Time, Service_Start, Service_End,
  Left_Without_Service, Left_Time, Queue_Length_At_Arrival, Counters_Open
  (times as HH:MM:SS; Service_* blank for leavers; Left_Time blank for served)

OUTPUT (folder outputs/): results_by_session.csv, results_summary.csv,
  validation_baseline.csv, fig_wait_vs_alpha.png

SCENARIO INPUTS below are ASSUMPTIONS, not observations. Replace with
measured values (pilot) or report them as ranges.
"""
import sys, os, heapq
import numpy as np, pandas as pd
from scipy import stats

# ---------------- CONFIG (scenario variables) ----------------
ALPHAS = [0.2, 0.4, 0.6, 0.8]
F_PREP = 0.50        # ASSUMED preparation share of service time (replace if measured)
H_MIN = 0.50         # ASSUMED handover time, minutes (replace if measured)
LEAD_MIN = 20.0      # ASSUMED order lead time before planned pickup, minutes
GAP_SD = 2.0         # SD of (actual arrival - planned pickup), minutes
REPS = 100
SEED = 12345
HERE = os.path.dirname(os.path.abspath(__file__))   # folder where this script is saved
INFILE = os.path.join(HERE, "SYNTHETIC_TWO_COUNTER_walkin_observations.csv")  # change to your real CSV name later
OUT = os.path.join(HERE, "outputs")

# ---------------- optional synthetic generator ----------------
def make_test_data():
    rng = np.random.default_rng(1)
    rows = []
    for sid in ["TEST_A", "TEST_B"]:
        t, cid = 12 * 3600.0, 0
        while t < 13 * 3600:
            t += rng.exponential(18)
            cid += 1
            s = rng.lognormal(np.log(60), 0.5)
            rows.append([sid, cid, t, s])
    df = pd.DataFrame(rows, columns=["Session_ID", "Customer_ID", "a", "s"])
    out = []
    for sid, g in df.groupby("Session_ID"):
        free = [0.0] * 3
        for _, r in g.iterrows():
            k = int(np.argmin(free)); st = max(r.a, free[k]); free[k] = st + r.s
            f = lambda x: f"{int(x//3600):02d}:{int(x%3600//60):02d}:{int(x%60):02d}"
            out.append([sid, int(r.Customer_ID), f(r.a), f(st), f(st + r.s), 0, "", 0, 3])
    pd.DataFrame(out, columns=["Session_ID", "Customer_ID", "Arrival_Time", "Service_Start",
        "Service_End", "Left_Without_Service", "Left_Time", "Queue_Length_At_Arrival",
        "Counters_Open"]).to_csv("SYNTHETIC_walkin_observations.csv", index=False)
    print("Wrote SYNTHETIC_walkin_observations.csv (artificial; for code testing only)")

# ---------------- data loading and cleaning ----------------
def secs(x):
    if pd.isna(x) or str(x).strip() == "": return np.nan
    h, m, s = str(x).split(":"); return int(h) * 3600 + int(m) * 60 + int(s)

def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "Session_ID" not in df.columns and "Day" in df.columns:   # accept "Day" as the session column
        df = df.rename(columns={"Day": "Session_ID"})
    for c in ["Arrival_Time", "Service_Start", "Service_End", "Left_Time"]:
        df[c] = df[c].map(secs)
    bad = pd.Series(False, index=df.index)
    served = df.Left_Without_Service == 0
    bad |= df.Arrival_Time.isna()
    bad |= served & (df.Service_Start.isna() | df.Service_End.isna())
    bad |= served & (df.Service_Start < df.Arrival_Time)
    bad |= served & (df.Service_End - df.Service_Start < 5)
    bad |= ~served & (df.Left_Time.isna() | (df.Left_Time < df.Arrival_Time))
    bad |= df.duplicated(["Session_ID", "Customer_ID"])
    os.makedirs(OUT, exist_ok=True)
    df[bad].to_csv(f"{OUT}/rejected_rows.csv", index=False)
    print(f"Rows read {len(df)}, rejected {bad.sum()}, kept {(~bad).sum()}")
    df = df[~bad].copy()
    df["a"] = (df.Arrival_Time - df.groupby("Session_ID").Arrival_Time.transform("min")) / 60
    df["s"] = (df.Service_End - df.Service_Start) / 60
    df["served"] = (df.Left_Without_Service == 0).astype(int)
    df["theta"] = np.where(df.served == 1, np.inf, (df.Left_Time - df.Arrival_Time) / 60)
    df["wait_obs"] = np.where(df.served == 1, (df.Service_Start - df.Arrival_Time) / 60, np.nan)
    return df.sort_values(["Session_ID", "a"]).reset_index(drop=True)

# ---------------- simulation engine (FCFS, pools of servers) ----------------
def run(jobs, pools):
    """jobs: dict id -> (release, duration, pool, patience, extra_release_fn or None)
    FCFS by release time; each pool has its own list of free times."""
    free = {p: [0.0] * n for p, n in pools.items()}
    heap = [(r, i) for i, (r, d, p, th, nxt) in jobs.items()]
    heapq.heapify(heap)
    res, busy = {}, {p: 0.0 for p in pools}
    while heap:
        r, i = heapq.heappop(heap)
        _, d, p, th, nxt = jobs[i]
        if not free[p]:                      # pool with no staff: no queueing server (shelf)
            start = r
        else:
            k = int(np.argmin(free[p])); start = max(r, free[p][k])
            if start - r > th:               # walk-away
                res[i] = (None, r + th, True); continue
            free[p][k] = start + d; busy[p] += d
        res[i] = (start, start + d, False)
        if nxt is not None:                  # prep finished -> release pickup job
            nxt(i, start + d, heap)
    return res, busy

def simulate(g, c, alpha, arrangement, u, lead, gap, s_leaver):
    a = g.a.values; s = g.s.values.copy(); th = g.theta.values
    s = np.where(g.served.values == 1, s, s_leaver)
    n = len(g); pre = (u < alpha) if alpha > 0 else np.zeros(n, bool)
    if arrangement == "shared": pools = {"w": c, "k": 0}
    elif arrangement == "shelf": pools = {"w": c - 1, "k": 1, "h": 0}
    else: pools = {"w": c - 2, "k": 1, "h": 1}
    prep_pool = "w" if arrangement == "shared" else "k"
    pick_pool = "w" if arrangement == "shared" else ("h" if arrangement == "pickup" else "h")
    jobs, extra = {}, {}
    def make_pickup(i, R, heap):
        rel = max(R, a[i]); jobs[("pk", i)] = (rel, H_MIN, pick_pool, np.inf, None)
        heapq.heappush(heap, (rel, ("pk", i)))
    for i in range(n):
        if pre[i]:
            o = a[i] - gap[i] - lead[i]
            jobs[("pp", i)] = (o, F_PREP * s[i], prep_pool, np.inf, make_pickup)
        else:
            jobs[("w", i)] = (a[i], s[i], "w", th[i], None)
    # fix key ordering: heap tuples need comparable ids
    jobs = {str(k): v for k, v in jobs.items()}
    def make_pickup2(i_key, R, heap):
        i = int(i_key.split(", ")[1].rstrip(")"))
        rel = max(R, a[i]); k = f"('pk', {i})"
        jobs[k] = (rel, H_MIN, pick_pool, np.inf, None); heapq.heappush(heap, (rel, k))
    jobs = {k: (v[0], v[1], v[2], v[3], make_pickup2 if v[4] else None) for k, v in jobs.items()}
    res, busy = run(jobs, pools)
    waits, w_walk, w_pick, left, tsys, spells = [], [], [], 0, [], 0.0
    end_max = 0.0
    for i in range(n):
        if pre[i]:
            st, en, _ = res[f"('pk', {i})"]
            w = st - a[i]; w_pick.append(w); waits.append(w); tsys.append(en - a[i]); spells += w
        else:
            st, en, ab = res[f"('w', {i})"]
            if ab: left += 1; spells += en - a[i]
            else:
                w = st - a[i]; w_walk.append(w); waits.append(w); tsys.append(en - a[i]); spells += w
        end_max = max(end_max, en)
    T = max(end_max - a.min(), 1e-9)
    work = sum(busy.values())
    return dict(avg_wait=np.mean(waits), wait_walkin=np.mean(w_walk) if w_walk else np.nan,
        wait_preorder=np.mean(w_pick) if w_pick else np.nan, avg_queue=spells / T,
        time_in_system=np.mean(tsys), utilisation=work / (c * T), abandon=left / n)

def ci95(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2: return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))

def main():
    df = load(INFILE)
    rng = np.random.default_rng(SEED)
    rows, valid = [], []
    for sid, g in df.groupby("Session_ID"):
        g = g.reset_index(drop=True); c = int(g.Counters_Open.iloc[0]); n = len(g)
        served_s = g.s[g.served == 1].values
        valid.append(dict(Session_ID=sid, n=n, counters=c, obs_mean_wait=g.wait_obs.mean(),
            obs_abandon=1 - g.served.mean(), arrival_rate_per_min=n / (g.a.max() or 1),
            mean_service_min=served_s.mean(), cv_service=served_s.std(ddof=1) / served_s.mean()))
        acc = {}
        for r in range(REPS):
            u = rng.random(n); lead = rng.uniform(0.75, 1.25, n) * LEAD_MIN
            gap = rng.normal(0, GAP_SD, n); sl = rng.choice(served_s, n)
            scen = [("A", 0.0, "shared")] + [("B_" + arr, al, arr) for arr in
                    (["shared", "shelf"] + (["pickup"] if c >= 3 else [])) for al in ALPHAS]
            for name, al, arr in scen:
                if arr != "shared" and c < 2: continue
                m = simulate(g, c, al, arr, u, lead, gap, sl)
                acc.setdefault((name.replace("B_", "B "), al), []).append(m)
        for (name, al), ms in acc.items():
            d = pd.DataFrame(ms).mean().to_dict()
            rows.append(dict(Session_ID=sid, system=name, alpha=al, **d))
        va = [m["avg_wait"] for m in acc[("A", 0.0)]]
        valid[-1]["sim_A_mean_wait"] = np.mean(va)
        valid[-1]["sim_A_abandon"] = np.mean([m["abandon"] for m in acc[("A", 0.0)]])
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/results_by_session.csv", index=False)
    pd.DataFrame(valid).to_csv(f"{OUT}/validation_baseline.csv", index=False)
    cols = ["avg_wait", "wait_walkin", "wait_preorder", "avg_queue", "time_in_system",
            "utilisation", "abandon"]
    S = R.groupby(["system", "alpha"])[cols].agg(["mean", ci95]).round(3)
    S.to_csv(f"{OUT}/results_summary.csv")
    A = R[R.system == "A"].set_index("Session_ID").avg_wait
    pc = []
    for (sy, al), g in R[R.system != "A"].groupby(["system", "alpha"]):
        d = (A - g.set_index("Session_ID").avg_wait).dropna()
        pc.append(dict(system=sy, alpha=al, mean_A_minus_B=d.mean(), ci95=ci95(d),
                       reduction_pct=100 * d.mean() / A.mean(), sessions=len(d)))
    pd.DataFrame(pc).round(3).to_csv(f"{OUT}/paired_comparison.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4.5))
        for sy, g in R[R.system != "A"].groupby("system"):
            m = g.groupby("alpha").avg_wait.mean(); plt.plot(m.index * 100, m.values, "o-", label=sy)
        plt.axhline(A.mean(), ls="--", c="k", label="System A (existing)")
        plt.xlabel("Pre-order share α (%)"); plt.ylabel("Average physical wait (min)")
        plt.title("Simulated waiting vs adoption (scenario assumptions apply)")
        plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/fig_wait_vs_alpha.png", dpi=150)
    except Exception as e: print("plot skipped:", e)
    print(pd.DataFrame(valid).round(3).to_string(index=False))
    print(pd.DataFrame(pc).round(3).to_string(index=False))
    print(f"\nF_PREP={F_PREP} and H_MIN={H_MIN} are ASSUMED values unless you measured them.")

if __name__ == "__main__":
    if "--make-test-data" in sys.argv: make_test_data()
    else: main()
