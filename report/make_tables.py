"""Deliverable 2: turn report/stats/*.json into numbers2.tex (macros) and tables/*.tex.

Sections whose module has not been run yet (error analysis, bias check) get "pending"
placeholders instead of numbers, so the report never shows invented results.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "stats"
T = HERE / "tables"
T.mkdir(exist_ok=True)
PENDING = r"\emph{pending}"
TASK_NAME = {"weak": "Task A (weak labels)", "human": "Task B (human labels)"}
SHORT = {"INFORMATION-TECHNOLOGY": "IT", "BUSINESS-DEVELOPMENT": "Business Dev.",
         "PUBLIC-RELATIONS": "PR", "DIGITAL-MEDIA": "Digital Media", "BPO": "BPO", "HR": "HR"}


def load(name: str) -> dict:
    p = S / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def f3(x) -> str:
    return f"{x:.3f}"


def lr_str(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def esc(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table(name: str, spec: str, header: str, rows: list[str], caption: str, label: str,
          size: str = r"\small") -> None:
    body = "\n".join(rows)
    (T / f"{name}.tex").write_text(
        f"\\begin{{table}}[H]\n\\centering{size}\n\\begin{{tabularx}}{{\\linewidth}}{{{spec}}}\n"
        f"\\toprule\n{header}\\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabularx}}\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{table}}\n")


def main() -> None:
    m: dict[str, str] = {}
    bi, ev, rt = load("bi_encoder_tuning"), load("evaluation"), load("retrieval")
    mn = {t: load(f"matchnet_{t}") for t in ("weak", "human")}

    # ---- bi-encoder
    pre, best = bi["pretrained_val"], bi["best"]
    m.update(BiPreAucHard=f3(pre["auc_hard"]), BiPrePthree=f3(pre["p_at_3"]), BiPreAuc=f3(pre["auc"]),
             BiAucHard=f3(best["auc_hard"]), BiPthree=f3(best["p_at_3"]), BiAuc=f3(best["auc"]),
             BiBestEpoch=str(best["epoch"]), BiEpochs=str(bi["epochs"]),
             BiTriplets=f"{bi['train_triplets_per_epoch']:,}", BiRuns=str(len(bi["grid"])))
    bc = next(r for r in bi["grid"] if f"lr{r['lr']:g}_bs{r['batch_size']}_len{r['max_seq_len']}" == best["config"])
    m.update(BiBestLr=lr_str(bc["lr"]), BiBestBs=str(bc["batch_size"]), BiBestLen=str(bc["max_seq_len"]))
    rows = []
    for r in bi["grid"]:
        cells = [lr_str(r["lr"]), str(r["batch_size"]), str(r["max_seq_len"]), str(r["best_epoch"]),
                 f3(r["val_auc"]), f3(r["val_auc_hard"]), f3(r["val_p_at_3"]), f"{r['minutes']:.0f}"]
        if r is bc:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        rows.append(" & ".join(cells) + r"\\")
    rows.insert(0, r"\multicolumn{4}{l}{pretrained MiniLM, no fine-tuning} & " +
                f"{f3(pre['auc'])} & {f3(pre['auc_hard'])} & {f3(pre['p_at_3'])} & --\\\\")
    table("tab_bi_grid", "@{}c c c c r r r r@{}",
          r"Learning rate & Batch & Max len & Best ep. & Val AUC & Val AUC (hard) & Val P@3 & Min",
          rows, "Bi-encoder hyperparameter search (validation split). Selection criterion: AUC of "
          "positives vs.\\ hard negatives. The selected run is in bold.", "tab:bigrid")

    # ---- MatchNet
    for t, tag in (("weak", "A"), ("human", "B")):
        b = mn[t]["best"]
        m.update({f"Mn{tag}Lr": lr_str(b["lr"]), f"Mn{tag}Bs": str(b["batch_size"]),
                  f"Mn{tag}Drop": str(b["dropout"]), f"Mn{tag}Hidden": str(b["hidden"]),
                  f"Mn{tag}Epoch": str(b["best_epoch"]), f"Mn{tag}ValAuc": f3(b["val_auc"]),
                  f"Mn{tag}Params": f"{mn[t]['n_params']:,}",
                  f"Mn{tag}AblPre": f3(mn[t]["ablations_val_auc"]["pretrained_embeddings"]),
                  f"Mn{tag}AblNoFeat": f3(mn[t]["ablations_val_auc"]["no_ats_features"])})
    m["MnRuns"] = str(len(mn["weak"]["grid"]))
    rows = []
    for hp in ("lr", "batch_size", "dropout", "hidden"):
        vals = sorted({r[hp] for r in mn["weak"]["grid"]})
        for v in vals:
            cells = [{"lr": "Learning rate", "batch_size": "Batch size", "dropout": "Dropout",
                      "hidden": "Hidden size $H$"}[hp] if v == vals[0] else "",
                     lr_str(v) if hp == "lr" else str(v)]
            for t in ("weak", "human"):
                g = [r["val_auc"] for r in mn[t]["grid"] if r[hp] == v]
                sel = mn[t]["best"][hp] == v
                txt = f"{sum(g) / len(g):.3f} / {max(g):.3f}"
                cells.append(rf"\textbf{{{txt}}}" if sel else txt)
            rows.append(" & ".join(cells) + r"\\")
        rows.append(r"\addlinespace")
    rows = rows[:-1]
    table("tab_mn_grid", "@{}l c c c@{}",
          r"Hyperparameter & Value & Task A: mean / max val AUC & Task B: mean / max val AUC",
          rows, f"MatchNet grid search: all {m['MnRuns']} combinations of the values below were "
          "trained per task; each cell averages (and maximizes) validation AUC over the runs with that "
          "value. Bold marks the value of the selected configuration. The full grid is in the appendix.",
          "tab:mngrid")
    rows = []
    for ra, rb in zip(mn["weak"]["grid"], mn["human"]["grid"]):
        rows.append(f"{lr_str(ra['lr'])} & {ra['batch_size']} & {ra['dropout']} & {ra['hidden']} & "
                    f"{f3(ra['val_auc'])} & {ra['best_epoch']} & {f3(rb['val_auc'])} & {rb['best_epoch']}\\\\")
    table("tab_mn_full", "@{}c c c c r r r r@{}",
          r"LR & Batch & Dropout & $H$ & A: val AUC & A: epoch & B: val AUC & B: epoch",
          rows, "Full MatchNet grid (validation AUC and early-stopped epoch per task).", "tab:mnfull",
          size=r"\footnotesize")

    # ---- test results
    order = ["ATS score (rules)", "TF-IDF cosine", "MiniLM cosine (pretrained)",
             "Bi-encoder cosine (fine-tuned)", "Logistic regression", "Gradient boosting", "MatchNet"]
    for t, tag in (("weak", "A"), ("human", "B")):
        r = ev[t]["models"]
        rows = []
        for name in order:
            x = r[name]
            cells = [esc(name), f3(x["precision"]), f3(x["recall"]), f3(x["f1"]), f3(x["accuracy"]),
                     f"{f3(x['auc_roc'])} [{x['auc_ci'][0]:.2f}, {x['auc_ci'][1]:.2f}]", f3(x["pr_auc"])]
            if name == "MatchNet":
                cells = [rf"\textbf{{{c}}}" for c in cells]
            rows.append(" & ".join(cells) + r"\\")
        table(f"tab_results_{t}", "@{}l r r r r r r@{}",
              r"Model & Precision & Recall & F1 & Accuracy & AUC-ROC [95\% CI] & PR-AUC",
              rows, f"{TASK_NAME[t]}: test-set results (n = {ev[t]['test_size']:,}, "
              f"{100 * ev[t]['test_positive_share']:.0f}\\% positive). Thresholds were chosen on the "
              "validation split by maximum F1.", f"tab:res{t}")
        mnr = r["MatchNet"]
        m.update({f"Res{tag}F": f3(mnr["f1"]), f"Res{tag}P": f3(mnr["precision"]),
                  f"Res{tag}R": f3(mnr["recall"]), f"Res{tag}Auc": f3(mnr["auc_roc"]),
                  f"Res{tag}AucLo": f"{mnr['auc_ci'][0]:.2f}", f"Res{tag}AucHi": f"{mnr['auc_ci'][1]:.2f}",
                  f"Res{tag}AtsAuc": f3(r["ATS score (rules)"]["auc_roc"]),
                  f"Res{tag}AtsF": f3(r["ATS score (rules)"]["f1"]),
                  f"Res{tag}PreAuc": f3(r["MiniLM cosine (pretrained)"]["auc_roc"]),
                  f"Res{tag}BiAuc": f3(r["Bi-encoder cosine (fine-tuned)"]["auc_roc"]),
                  f"Res{tag}GbAuc": f3(r["Gradient boosting"]["auc_roc"]),
                  f"Res{tag}LrAuc": f3(r["Logistic regression"]["auc_roc"]),
                  f"Res{tag}N": f"{ev[t]['test_size']:,}",
                  f"Res{tag}PosShare": f"{100 * ev[t]['test_positive_share']:.0f}"})
        cm = ev[t]["matchnet_confusion"]
        m.update({f"Res{tag}FP": f"{cm[0][1]:,}", f"Res{tag}FN": f"{cm[1][0]:,}"})
    pt = ev["weak"].get("matchnet_accuracy_by_pair_type", {})
    m.update(AccPos=f3(pt.get("positive", 0)), AccHard=f3(pt.get("hard_negative", 0)),
             AccRand=f3(pt.get("random_negative", 0)))
    fit = ev["human"].get("matchnet_mean_score_by_fit", {})
    m.update(FitNo=f3(fit.get("No Fit", 0)), FitPot=f3(fit.get("Potential Fit", 0)),
             FitGood=f3(fit.get("Good Fit", 0)), AucGoodNo=f3(ev["human"].get("matchnet_auc_good_vs_no", 0)))

    # ---- retrieval
    p3 = rt["precision_at_3"]
    rows = [f"{esc(k)} & {v:.3f}\\\\" for k, v in p3.items()]
    rows.append(f"Random ranking (expected) & {rt['random_baseline']:.3f}\\\\")
    table("tab_retrieval", "@{}L r@{}", r"Ranker & Precision@3",
          rows, f"Top-3 recommendation on the test split: {rt['test_resumes']} resumes, each ranked "
          f"against all {rt['test_postings']:,} test postings with a known occupation. A recommended "
          "posting counts as relevant if it belongs to the resume's occupation.", "tab:retrieval")
    keys = list(p3)
    m.update(PthreeTfidf=f3(p3[keys[0]]), PthreePre=f3(p3[keys[1]]), PthreeFt=f3(p3[keys[2]]),
             PthreeRerank=f3(p3[keys[3]]), PthreeRandom=f3(rt["random_baseline"]))
    cats = rt["precision_at_3_by_category"]
    rr = sorted(cats.items(), key=lambda kv: kv[1][keys[3]])
    m["PthreeWorst"] = ", ".join(f"{SHORT.get(c, c.title())} ({v[keys[3]]:.2f})" for c, v in rr[:3])
    m["PthreeBest"] = ", ".join(f"{SHORT.get(c, c.title())} ({v[keys[3]]:.2f})" for c, v in rr[-3:][::-1])

    # ---- error analysis (Muaaz) and bias check (Hanan)
    ea = load("error_analysis")
    if ea:
        w = ea["weak"]
        fm = w["feature_means"]

        pretty = {"title_match": "title match", "skill_cov": "skill coverage"}

        def feat_gap(hi: str, lo: str, feats: tuple[str, ...]) -> str:
            bits = [f"{pretty.get(f, f)} ({fm[hi][f]:.2f} vs.\\ {fm[lo][f]:.2f})" for f in feats]
            return " and ".join(bits)

        m.update(ErrWorst=", ".join(SHORT.get(c, c.title()) for c in w["worst_categories"]),
                 ErrBest=", ".join(SHORT.get(c, c.title()) for c in w["best_categories"]),
                 ErrHard=f3(w["error_rate_by_pair_type"]["hard_negative"]),
                 ErrPos=f3(w["error_rate_by_pair_type"]["positive"]),
                 ErrRand=f3(w["error_rate_by_pair_type"]["random_negative"]),
                 ErrPot=f3(ea["human"]["error_rate_by_fit"]["Potential Fit"]),
                 ErrNoFit=f3(ea["human"]["error_rate_by_fit"]["No Fit"]),
                 ErrGood=f3(ea["human"]["error_rate_by_fit"]["Good Fit"]), ErrStatus="done",
                 ErrFeatFp=feat_gap("FP", "TN", ("title_match", "skill_cov")),
                 ErrFeatFn=feat_gap("FN", "TP", ("skill_cov", "title_match")))
        rows = [f"{esc(SHORT.get(c, c.title()))} & {v['precision']:.3f} & {v['recall']:.3f} & "
                f"{v['f1']:.3f} & {v['support']}\\\\"
                for c, v in sorted(w["per_category"].items(), key=lambda kv: kv[1]["f1"])]
        table("tab_error_category", "@{}L r r r r@{}", r"Resume category & Precision & Recall & F1 & Pairs",
              rows, "MatchNet per resume category on the Task A test split (worst first).", "tab:errcat",
              size=r"\footnotesize")
    else:
        for k in ("ErrWorst", "ErrBest", "ErrHard", "ErrPos", "ErrRand", "ErrPot", "ErrNoFit",
                  "ErrGood", "ErrFeatFp", "ErrFeatFn"):
            m[k] = PENDING
        m["ErrStatus"] = "pending"
        table("tab_error_category", "@{}L r r r r@{}", r"Resume category & Precision & Recall & F1 & Pairs",
              [r"\multicolumn{5}{l}{\emph{Pending: run} \code{src/error\_analysis.py}}\\"],
              "MatchNet per resume category on the Task A test split.", "tab:errcat")

    bias = load("bias_check")
    if bias:
        rows = []
        for k, v in bias["tests"].items():
            ptxt = f"{v['p_value']:.3f}" if v["p_value"] >= 0.001 else r"$<0.001$"
            rows.append(
                f"{esc(k.replace('_', ' '))} & {v['mean_delta']:+.4f} & "
                f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] & {v['mean_abs_delta']:.4f} & "
                f"{ptxt} & {100 * v['flip_rate']:.1f}\\%\\\\")
        m.update(BiasN=str(bias["n_pairs"]), BiasConclusion=esc(bias.get("conclusion", "")))
    else:
        rows = [r"\multicolumn{6}{l}{\emph{Pending: run} \code{src/bias\_check.py}}\\"]
        m.update(BiasN="300", BiasConclusion=PENDING)
    table("tab_bias", "@{}L r r r r r@{}",
          r"Counterfactual test & Mean $\Delta$ & 95\% CI & Mean $|\Delta|$ & $p$ & Decisions flipped",
          rows, "Bias check: change in MatchNet score when only one attribute of the resume is "
          "swapped (variant B minus variant A).", "tab:bias")

    # ---- contribution table
    assigned = {
        "Muhammad Sufyan": r"\code{src/models/}, \code{train\_*.py}, \code{evaluate.py}",
        "Muaaz Fahad": r"\code{src/error\_analysis.py}",
        "Hanan Ishaque": r"\code{src/bias\_check.py}",
    }
    con = load("contributions")
    rows = []
    for name, c in con.items():
        mods = ", ".join(r"\code{" + esc(x.replace("data_scripts/", "").replace("src/", "")) + "}"
                         for x in c["modules"][:4]) or "---"
        rows.append(f"{esc(name)} & {assigned.get(name, '---')} & {c['commits']} & "
                    f"+{c['lines_added']:,} / $-${c['lines_removed']:,} & {mods}\\\\")
    table("tab_contrib", "@{}l L r r L@{}",
          r"Member & Assigned module & Commits & Lines & Main files in git log",
          rows, "Contribution table: assigned Deliverable~2 modules, plus Git history from "
          "\\code{report/contributions.py} (generated files such as figures, PDFs and JSON "
          "statistics are excluded).", "tab:contrib")

    lines = ["% auto-generated by make_tables.py -- do not edit"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}\\xspace}}" for k, v in m.items()]
    (HERE / "numbers2.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(m)} macros to numbers2.tex and {len(list(T.glob('*.tex')))} tables")


if __name__ == "__main__":
    main()
