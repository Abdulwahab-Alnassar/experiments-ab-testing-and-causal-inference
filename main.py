"""Simulate and evaluate a randomized book-preview experiment (Python 3.12)."""
from hashlib import sha256
from math import ceil, sqrt
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def assign_variant(user_id):
    key = f"book-search-preview-v1:{user_id}".encode("utf-8")
    bucket = int.from_bytes(sha256(key).digest()[:8], "big")
    return "B" if bucket < 2**63 else "A"


def main():
    root = Path(__file__).resolve().parent
    dataset = root / "data"
    charts = root / "outputs" / "charts"
    results = root / "outputs" / "results"
    for directory in (dataset, charts, results):
        directory.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams["svg.hashsalt"] = "book-preview"
    test_ids = [f"demo-{i}" for i in range(20)]
    first_visit = {uid: assign_variant(uid) for uid in test_ids}
    second_visit = {uid: assign_variant(uid) for uid in reversed(test_ids)}
    assert first_visit == second_visit
    assert set(first_visit.values()) == {"A", "B"}
    print("نجح اختبار التخصيص: بقيت الحسابات العشرون في نسخها عند إعادة الزيارة.")

    CONFIG = {
        "baseline_save_rate": 0.18,
        "mde_abs": 0.02,
        "daily_eligible_accounts": 800,
        "treatment_share": 0.50,
        "alpha": 0.05,
        "power": 0.80,
        "follow_up_hours": 24,
    }

    p_a = CONFIG["baseline_save_rate"]
    p_b = p_a + CONFIG["mde_abs"]
    daily = CONFIG["daily_eligible_accounts"]
    assert CONFIG["treatment_share"] == 0.5  # الصيغة لمجموعتين متساويتين
    normal = NormalDist()
    z_sum = normal.inv_cdf(1 - CONFIG["alpha"] / 2) + normal.inv_cdf(CONFIG["power"])
    variance = p_a * (1 - p_a) + p_b * (1 - p_b)
    n_min = ceil(z_sum**2 * variance / CONFIG["mde_abs"]**2)
    days_min = ceil(2 * n_min / daily)
    weeks = ceil(days_min / 7)
    days = 7 * weeks

    print(f"الحد الأدنى: {n_min:,} حسابًا لكل نسخة، أي {2 * n_min:,} إجمالًا.")
    print(f"التسجيل: {days_min} يومًا قبل التقريب، و{days} يومًا ({weeks} أسابيع) بعده.")
    for label, period in [("مدة الخطة", days), ("أسبوع واحد", 7)]:
        per_group = period * daily * CONFIG["treatment_share"]
        mde = z_sum * sqrt(2 * p_a * (1 - p_a) / per_group)
        print(f"{label}: {per_group:,.0f} حساب لكل نسخة، وأصغر أثر تقريبي {100 * mde:.2f} نقطة مئوية.")
    print(f"موعد التحليل: بعد {days} يوم تسجيل و{CONFIG['follow_up_hours']} ساعة متابعة لآخر حساب.")

    rng = np.random.default_rng(2026)
    size = days * CONFIG["daily_eligible_accounts"]
    ids = [f"reader-{i}" for i in range(size)]
    variant = np.array([assign_variant(uid) for uid in ids])
    treated = variant == "B"
    day = np.repeat(np.arange(1, days + 1), CONFIG["daily_eligible_accounts"])
    # نشاط سابق ذو مستويين متساويي الاحتمال، ومتوسط خط أساس متوقع 18%.
    active = rng.binomial(1, 0.5, size)
    prior = rng.poisson(1 + 6 * active)
    n_requests = rng.integers(1, 5, size)
    owner = np.repeat(np.arange(size), n_requests)
    failed = rng.random(len(owner)) < 0.006
    latency = rng.lognormal(np.log(260), 0.35, len(owner)) + 30 * treated[owner]
    can_view = np.bincount(owner, weights=~failed, minlength=size) > 0
    save_prob = CONFIG["baseline_save_rate"] + 0.18 * (active - 0.5) + 0.02 * treated
    saved = can_view & (rng.random(size) < save_prob)
    books = saved * (1 + rng.poisson(0.35, size))
    opened = saved | (can_view & (rng.random(size) < 0.36 + 0.12 * active))
    data = pd.DataFrame(dict(account=ids, group=variant, day=day, prior=prior,
                             saved=saved.astype(int), books=books, opened=opened.astype(int)))
    print(f"{len(data):,} accounts | {len(owner):,} search requests | {days} enrollment days")

    # 6. فحوص الثقة: اكتمال 24 ساعة لكل حساب افتراض في المحاكاة.
    # في المنتج نتحقق من مرور 24 ساعة بين أول بحث مؤهل واستخراج البيانات، ثم نفحص وقوع الحفظ داخل النافذة.
    assert data.account.is_unique and not data.isna().any().any()
    assert np.array_equal(data.group, [assign_variant(uid) for uid in data.account])
    assert np.array_equal(data.saved, (data.books > 0).astype(int))
    counts = data.group.value_counts().reindex(["A", "B"])
    srm_p = stats.chisquare(counts, f_exp=[size / 2, size / 2]).pvalue
    print(f"A={counts.A:,}, B={counts.B:,}, SRM p={srm_p:.4f}")
    if srm_p < 0.001 or counts.min() < n_min:
        raise ValueError("Review allocation and sample size before interpreting effects.")

    # 7.1 و7.2: الفرق وفترة الثقة، CUPED، وتصحيح المقاييس الثانوية
    def compare(values, scale=1):
        a, b = (np.asarray(values)[variant == arm] * scale for arm in ["A", "B"])
        diff = b.mean() - a.mean()
        se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
        return pd.Series(dict(A=a.mean(), B=b.mean(), diff=diff, lo=diff-1.96*se,
                              hi=diff+1.96*se, SE=se, p=2*stats.norm.sf(abs(diff/se))))

    theta = data.saved.cov(data.prior) / data.prior.var()
    adjusted = data.saved - theta * (data.prior - data.prior.mean())
    primary = pd.DataFrame({"Raw": compare(data.saved, 100),
                            "CUPED": compare(adjusted, 100)}).T
    print(primary.round(4).to_string())
    print(f"Variance reduction: {1-(primary.SE.CUPED/primary.SE.Raw)**2:.1%}")
    secondary = pd.DataFrame({"Opened (%)": compare(data.opened, 100),
                              "Books/account": compare(data.books)}).T
    secondary["p_BH"] = stats.false_discovery_control(secondary.p, method="bh")
    secondary["significant_BH"] = secondary.p_BH < CONFIG["alpha"]
    print(secondary.round(4).to_string())

    # 7.3: إعادة أخذ عينات الحسابات مع جميع طلباتها داخل كل ذراع
    boot = np.random.default_rng(2027)
    def guardrails(arm, draws=1000):
        accounts = np.flatnonzero(variant == arm)
        mask = variant[owner] == arm
        local_owner = np.searchsorted(accounts, owner[mask])
        failures = np.bincount(local_owner, weights=failed[mask], minlength=len(accounts))
        requests = n_requests[accounts]
        good = ~failed[mask]
        order = np.argsort(latency[mask][good])
        times = latency[mask][good][order]
        user = local_owner[good][order]

        def metrics(weights):
            ranks = weights[user].cumsum()
            p95 = times[np.searchsorted(ranks, np.ceil(0.95 * ranks[-1]))]
            error_pct = 100 * (weights @ failures) / (weights @ requests)
            return [p95, error_pct]

        observed = metrics(np.ones(len(accounts), dtype=int))
        samples = []
        for _ in range(draws):
            sampled = boot.integers(len(accounts), size=len(accounts))
            samples.append(metrics(np.bincount(sampled, minlength=len(accounts))))
        return np.array(observed), np.array(samples)

    ga, sa = guardrails("A")
    gb, sb = guardrails("B")
    low, high = np.quantile(sb-sa, [0.025, 0.975], axis=0)
    guards = pd.DataFrame(dict(A=ga, B=gb, diff=gb-ga, lo=low, hi=high,
                               limit=[150, 0.2]), index=["p95 latency (ms)", "Failed requests (%)"])
    guards["pass"] = guards.hi < guards.limit
    print(guards.round(3).to_string())

    # 7.4 أثر المقتطف حسب أسبوع التسجيل
    week = (data.day - 1) // 7 + 1
    weekly = data.groupby([week, "group"]).saved.mean().unstack()
    print("Save-rate difference by enrollment week (percentage points):")
    print((100 * (weekly.B - weekly.A)).round(2).to_string())
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(weekly.index, 100 * (weekly.B - weekly.A), marker="o", label="Weekly B - A")
    ax.axhline(primary.loc["Raw", "diff"], ls="--", label="Overall raw effect")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set(xlabel="Enrollment week", ylabel="Save-rate difference (pp)", xticks=[1, 2, 3])
    ax.set_title("Preview effect by enrollment week")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts / "weekly_effect.svg", metadata={"Date": None})
    plt.close(fig)
    effect = primary.loc["CUPED"]
    launch = effect["diff"] >= 100*CONFIG["mde_abs"] and effect.lo > 0 and guards["pass"].all()
    print("Simulation decision:", "Launch" if launch else "Hold rollout")
    data.to_csv(dataset / "dataset.csv", index=False)
    pd.DataFrame({"account": data.account.to_numpy()[owner], "failed": failed.astype(int), "latency_ms": latency}).to_csv(dataset / "requests.csv", index=False)
    summary = pd.concat([primary, secondary, guards], axis=0)
    summary.to_csv(results / "metrics.csv", index_label="metric")
    weekly.assign(difference_pp=100 * (weekly.B - weekly.A)).to_csv(results / "weekly.csv", index_label="week")


if __name__ == "__main__":
    main()
