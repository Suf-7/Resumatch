"""Turn report/stats/*.json (written by data_scripts/) into LaTeX macros in numbers.tex,
so every number in the report is reproducible from the pipeline."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = HERE / "stats"


def load(name: str) -> dict:
    p = S / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fmt(x) -> str:
    if isinstance(x, float):
        if abs(x) < 1:
            return f"{x:.2f}"
        return f"{x:,.2f}".rstrip("0").rstrip(".")
    if isinstance(x, int):
        return f"{x:,}"
    return str(x)


def main() -> None:
    cr, cj, ft, eda = load("clean_resumes"), load("clean_jobs"), load("features"), load("eda")
    esco, rz, dl = load("esco"), load("scrape_rozee"), load("download_manifest")
    lr, lj = cr.get("loaded_by_source", {}), cj.get("loaded_by_source", {})
    fr, fj = cr.get("final_by_source", {}), cj.get("final_by_source", {})
    m = {
        # resumes
        "ResRawMain": lr.get("kaggle_resume_dataset", 0),
        "ResRawUpd": lr.get("kaggle_updated_resume_dataset", 0),
        "ResRaw": cr.get("loaded", 0),
        "ResEmpty": cr.get("loaded", 0) - cr.get("after_drop_empty", 0),
        "ResMojibake": cr.get("mojibake_repaired", 0),
        "ResNonEn": cr.get("after_drop_empty", 0) - cr.get("after_language_filter", 0),
        "ResExactDup": cr.get("after_language_filter", 0) - cr.get("after_exact_dedup", 0),
        "ResExactDupUpd": cr.get("exact_duplicates_by_source", {}).get("kaggle_updated_resume_dataset", 0),
        "ResNearDup": cr.get("near_duplicates", 0),
        "ResShort": cr.get("flagged_short", 0),
        "ResLong": cr.get("flagged_long", 0),
        "ResIQR": cr.get("iqr_flagged", 0),
        "ResJunk": cr.get("dropped_junk", 0),
        "ResPlow": int(cr.get("length_bounds", {}).get("p_low", 0)),
        "ResPhigh": int(cr.get("length_bounds", {}).get("p_high", 0)),
        "ResPII": cr.get("resumes_with_pii_masked", 0),
        "ResPIItokens": cr.get("pii_tokens_inserted", 0),
        "ResFinal": cr.get("final", 0),
        "ResFinalMain": fr.get("kaggle_resume_dataset", 0),
        "ResFinalUpd": fr.get("kaggle_updated_resume_dataset", 0),
        "ResCats": cr.get("categories", 0),
        # jobs
        "JobRawLI": lj.get("linkedin_kaggle", 0),
        "JobRawRZ": lj.get("rozee_pk_scraped", 0),
        "JobRaw": cj.get("loaded", 0),
        "JobEmpty": cj.get("loaded", 0) - cj.get("after_drop_empty", 0),
        "JobNonEn": cj.get("non_english_dropped", 0),
        "JobExactDup": cj.get("exact_duplicates", 0),
        "JobNearDup": cj.get("near_duplicates", 0),
        "JobShort": cj.get("flagged_short", 0),
        "JobLong": cj.get("flagged_long", 0),
        "JobIQR": cj.get("iqr_flagged", 0),
        "JobJunk": cj.get("dropped_junk", 0),
        "JobPlow": int(cj.get("length_bounds", {}).get("p_low", 0)),
        "JobPhigh": int(cj.get("length_bounds", {}).get("p_high", 0)),
        "JobPII": cj.get("postings_with_pii_masked", 0),
        "JobFinal": cj.get("final", 0),
        "JobFinalLI": fj.get("linkedin_kaggle", 0),
        "JobFinalRZ": fj.get("rozee_pk_scraped", 0),
        "JobSalaryShare": round(100 * cj.get("salary_available_share", 0), 1),
        "JobSenUnknown": cj.get("seniority", {}).get("unknown", 0),
        "JobSenMissPct": round(100 * cj.get("missing_share_raw", {}).get("seniority", 0), 1),
        # external labelled pairs
        "FitPairsTrain": dl.get("fit_pairs_train", {}).get("rows", 0),
        "FitPairsTest": dl.get("fit_pairs_test", {}).get("rows", 0),
        # ESCO + scraping
        "EscoSkills": esco.get("skills", 0),
        "EscoAlt": esco.get("alt_labels", 0),
        "RozeeQueries": rz.get("queries_covered_total", rz.get("queries", 0)),
        "RozeeQueriesDefined": rz.get("queries_defined", 0),
        "RozeeRequests": rz.get("requests_logged", 0),
        "RozeeBlocked": rz.get("requests_blocked_403_429", 0),
        "RozeeDelay": rz.get("delay_s", 0),
        # features
        "SkillVocab": ft.get("skill_vocab_size", 0),
        "TfidfVocab": ft.get("tfidf_vocab_size", 0),
        "MeanSkillsRes": ft.get("mean_skills_resume", 0),
        "MeanSkillsJob": ft.get("mean_skills_job", 0),
        "YearsCovRes": round(100 * ft.get("years_exp_coverage_resumes", 0), 1),
        "YearsCovJob": round(100 * ft.get("years_required_coverage_jobs", 0), 1),
        "JobCatCov": round(100 * ft.get("job_category_coverage", 0), 1),
        "ResTrain": ft.get("resume_splits", {}).get("train", 0),
        "ResVal": ft.get("resume_splits", {}).get("val", 0),
        "ResTest": ft.get("resume_splits", {}).get("test", 0),
        "JobTrain": ft.get("job_splits", {}).get("train", 0),
        "JobVal": ft.get("job_splits", {}).get("val", 0),
        "JobTest": ft.get("job_splits", {}).get("test", 0),
        "Pairs": ft.get("pairs", 0),
        "PairsPos": ft.get("pairs_by_type", {}).get("positive", 0),
        "PairsHard": ft.get("pairs_by_type", {}).get("hard_negative", 0),
        "PairsRand": ft.get("pairs_by_type", {}).get("random_negative", 0),
        "PairsTrain": ft.get("pairs_by_split", {}).get("train", 0),
        "PairsVal": ft.get("pairs_by_split", {}).get("val", 0),
        "PairsTest": ft.get("pairs_by_split", {}).get("test", 0),
        "CosPos": ft.get("mean_cosine_by_type", {}).get("positive", 0),
        "CosHard": ft.get("mean_cosine_by_type", {}).get("hard_negative", 0),
        "CosRand": ft.get("mean_cosine_by_type", {}).get("random_negative", 0),
        "RefYear": str(ft.get("resume_reference_year", {}).get("kaggle_resume_dataset", "")),
        # EDA
        "ImbalanceRatio": eda.get("resume_imbalance_ratio", 0),
        "MedWordsRes": eda.get("median_words_resume", 0),
        "MedWordsJob": eda.get("median_words_job", 0),
        "SkillsUniqJobs": eda.get("skills_unique_jobs", 0),
        "SkillsUniqRes": eda.get("skills_unique_resumes", 0),
        "SkillsOverlap": eda.get("skills_overlap", 0),
        "SkillsJobOnly": eda.get("skills_unique_jobs", 0) - eda.get("skills_overlap", 0),
        "SkillsJaccard": eda.get("skills_jaccard", 0),
        "TopHundredMissing": eda.get("top100_job_skills_missing_in_most_resumes", 0),
        "AucTfidf": eda.get("auc_pos_vs_hard_tfidf_cosine", 0),
        "AucEmb": eda.get("auc_pos_vs_hard_emb_cosine", 0),
        "EmbPos": eda.get("mean_emb_cosine_by_type", {}).get("positive", 0),
        "EmbHard": eda.get("mean_emb_cosine_by_type", {}).get("hard_negative", 0),
        "CorrYearsSalary": eda.get("corr_jobs", {}).get("years req.", {}).get("salary/yr", 0),
        "CorrWordsSkillsJob": eda.get("corr_jobs", {}).get("words", {}).get("skills", 0),
        "CorrWordsSkillsRes": eda.get("corr_resume", {}).get("words", {}).get("skills", 0),
    }
    live, ats = load("live_pool"), load("ats_baseline")
    open_n = live.get("open", 0)
    m.update({
        "LiveAsOf": live.get("as_of", "")[:10],
        "LiveOpen": open_n,
        "LiveExpired": live.get("expired", 0),
        "LiveWeek": live.get("closing_within_7_days", 0),
        "LiveMedDays": live.get("median_days_left") or 0,
        "LiveIT": live.get("open_by_category", {}).get("INFORMATION-TECHNOLOGY", 0),
        "LiveITPct": round(100 * live.get("open_by_category", {}).get("INFORMATION-TECHNOLOGY", 0)
                           / max(open_n, 1)),
        "LiveYearsPct": round(100 * (live.get("open_with_years_required") or 0), 1),
        "AtsPos": ats.get("mean_score_by_type", {}).get("positive", 0),
        "AtsHard": ats.get("mean_score_by_type", {}).get("hard_negative", 0),
        "AtsRand": ats.get("mean_score_by_type", {}).get("random_negative", 0),
        "AtsAucHard": ats.get("auc_pos_vs_hard", 0),
        "AtsAucRand": ats.get("auc_pos_vs_random", 0),
        "AtsTestPairs": ats.get("test_pairs", 0),
        "RozeeRobotsBlocked": "yes" if rz.get("robots_txt_blocked") else "no",
    })
    g = eda.get("rozee_gender_preference", {})
    m["RozeeMale"], m["RozeeFemale"] = g.get("Male", 0), g.get("Female", 0)
    m["RozeeGendered"] = g.get("Male", 0) + g.get("Female", 0)
    top = list(eda.get("rozee_top_skills", {}))[:5]
    m["RozeeTopSkills"] = ", ".join(s.replace(" (computer programming)", "") for s in top)
    short = {"INFORMATION-TECHNOLOGY": "IT"}
    if eda.get("resume_category_min"):
        (k, v), = eda["resume_category_min"].items()
        m["ResCatMin"], m["ResCatMinN"] = short.get(k, k), v
    if eda.get("resume_category_max"):
        (k, v), = eda["resume_category_max"].items()
        m["ResCatMax"], m["ResCatMaxN"] = short.get(k, k), v
    lines = ["% auto-generated by make_numbers.py -- do not edit"]
    for k, v in m.items():
        lines.append(f"\\newcommand{{\\{k}}}{{{fmt(v)}\\xspace}}")
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(m)} macros to numbers.tex")


if __name__ == "__main__":
    main()
