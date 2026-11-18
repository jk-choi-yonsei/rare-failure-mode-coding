"""Tables of the redesign results (notes/prereg-redesign-2026-11-06.md) from the committed metrics.

Reads results/redesign/{maude,sdr}_metrics.json and results/maude2026nat/{natural,downstream}_metrics.json
and prints Markdown tables for the results note and LaTeX rows for the paper's selector table.
Nothing is recomputed. Usage: python redesign_tables.py [md|tex]
"""
import json
import sys

from common import ROOT

RES = ROOT / "results"
LLMS = ("opus55", "qwen38_27b", "gemma4_31b")
NAME = {"opus55": "Opus 5.5", "qwen38_27b": "Qwen3.8-27B", "gemma4_31b": "Gemma 4 31B"}
DS = {"maude": "Device", "sdr": "Aircraft"}
TESTS = [("D1", "D1_selector_minus_best_single", "best_single"), ("D2", "D2_selector_minus_cascade", "cascade"),
         ("D3", "D3_selector_minus_frequency_rule", "llm_rare"), ("D4", "D4_selector_minus_majority_vote", "mv_enc"),
         ("D7", "D7_retrieval_minus_zeroshot", None)]


def load(corpus):
    p = RES / "redesign" / f"{corpus}_metrics.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cell(c, digits=3):
    if c is None:
        return "not run"
    mark = "holds" if c["holds"] else "fails"
    return (f"{c['item']['point']:+.{digits}f} item [{c['item']['ci95'][0]:+.3f}, {c['item']['ci95'][1]:+.3f}] "
            f"code [{c['code']['ci95'][0]:+.3f}, {c['code']['ci95'][1]:+.3f}] {mark}")


def md():
    for corpus in ("maude", "sdr"):
        R = load(corpus)
        if R is None:
            print(f"\n({corpus}: no metrics yet)")
            continue
        print(f"\n## {DS[corpus]} reports (selector encoder {R['selector_encoder']}, git {R['git_head_at_score']})")
        for year in ("2026", "2025"):
            Y = R["years"][year]
            print(f"\n### {year} ({'test year' if year == '2026' else 'development year'})")
            print("\nVerdicts: " + ", ".join(f"{k} {v}" for k, v in Y["verdicts"].items() if not k.endswith("_by_llm")))
            print("\n| LLM | Ours | best single (name) | majority vote | frequency rule | cascade | encoder | prompted | retrieval | oracle | choice enc/zs/rag |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for llm in LLMS:
                S = Y["selectors"].get(llm, {})
                if "macro_f1" not in S:
                    print(f"| {NAME[llm]} | not run | | | | | | | | | |")
                    continue
                m = S["macro_f1"]
                f = S["frozen"]
                cs = f["choice_share"]
                print(f"| {NAME[llm]} | {m['selector']['all']:.3f} | {m['best_single']['all']:.3f} ({f['best_single_arm_2024']}) | "
                      f"{m['mv_enc']['all']:.3f} | {m['llm_rare']['all']:.3f} | {m['cascade']['all']:.3f} | {m['enc']['all']:.3f} | "
                      f"{m['zs']['all']:.3f} | {m['rag']['all']:.3f} | {m['oracle']['all']:.3f} | "
                      f"{cs['enc']:.2f}/{cs['zs']:.2f}/{cs['rag']:.2f} |")
            print("\n| LLM | test | difference, item and code 95% intervals |")
            print("|---|---|---|")
            for llm in LLMS:
                S = Y["selectors"].get(llm, {})
                for short, key, _ in TESTS:
                    print(f"| {NAME[llm]} | {short} | {cell(S.get('tests', {}).get(key))} |")
                if "D8_leak_free" in S:
                    d8 = S["D8_leak_free"]
                    print(f"| {NAME[llm]} | D8: D1 on {d8['n']} reports without own term | "
                          f"{cell(d8['tests']['D1_selector_minus_best_single'])} |")
            print("\n| test | cell (N / bin / LLM) | difference, item and code 95% intervals |")
            print("|---|---|---|")
            for t in ("D5", "D6"):
                for k, c in Y["boundary"][t].items():
                    print(f"| {t} | {k} | {cell(c)} |")
            if "boundary_leak_free" in Y:
                for k, c in Y["boundary_leak_free"]["D5"].items():
                    print(f"| D8: D5 without own term | {k} | {cell(c)} |")
            if Y["boundary"].get("missing"):
                print(f"\nMissing inputs: {Y['boundary']['missing']}")
            print("\nMacro-F1 by training size and bin (base encoder, seed mean; prompted LLMs):")
            print("\n| N/bin | codes | reports | encoder | " + " | ".join(NAME[l] for l in LLMS) + " |")
            print("|---|---|---|---|" + "---|" * len(LLMS))
            for k, v in Y["boundary"]["macro_f1"].items():
                enc = f"{v['encoder_base']:.3f}" if "encoder_base" in v else "-"
                print(f"| {k} | {v['n_codes']} | {v['n_items']} | {enc} | " +
                      " | ".join(f"{v[l]:.3f}" if v.get(l) is not None else "-" for l in LLMS) + " |")
        if corpus == "maude" and "E_2026" in R:
            E = R["E_2026"]
            print(f"\n### E: never-seen modes, 2026 ({E['n_novel']} reports of {E['novel_codes']} hidden codes; "
                  f"groups {E['groups']}; missing {E['missing']})")
            print("\n| score / known group | AUROC | item 95% | code 95% | better than chance |")
            print("|---|---|---|---|---|")
            for k, v in E["auroc"].items():
                print(f"| {k} | {v['auroc']:.3f} | {v['item_ci95']} | {v['code_ci95']} | {'yes' if v['better_than_chance'] else 'no'} |")
            print("\n2024 practice (direction check): " + ", ".join(f"{k} {v['auroc']:.3f}" for k, v in R["E_2024_practice"]["auroc"].items()))
    nat = RES / "maude2026nat" / "natural_metrics.json"
    if nat.exists():
        d = json.loads(nat.read_text(encoding="utf-8"))
        print(f"\n## F: device reports of 2026 in natural proportions (Opus 5.5)\n\nVerdicts: {d['verdicts']}")
        print("\n| arm | accuracy (uniform) | macro-F1 (uniform) | recall 1-5 | recall 6-20 | placed 1-5 |")
        print("|---|---|---|---|---|---|")
        for k in d["uniform"]:
            if not isinstance(d["uniform"][k], dict) or "micro_f1" not in d["uniform"][k]:
                continue
            r15, r620 = d["rare"]["rare_1-5"], d["rare"]["rare_6-20"]
            print(f"| {k} | {d['uniform'][k]['micro_f1']:.3f} | {d['uniform'][k]['macro_f1']:.3f} | {r15['recall'][k]:.3f} | "
                  f"{r620['recall'][k]:.3f} | {r15['surfaced_weighted_whole_year'][k]:.0f} of {r15['n']} |")
        print("\nContrasts:", json.dumps(d["contrasts"], indent=0)[:3000])
    ds = RES / "maude2026nat" / "downstream_metrics.json"
    if ds.exists():
        d = json.loads(ds.read_text(encoding="utf-8"))
        print(f"\n## F downstream, 2026 (population {d['population']})")
        print("\n| arm | modes visible | correct | false placements [95%] | precision [95%] | review queue | rank corr all / rare |")
        print("|---|---|---|---|---|---|---|")
        for k, a in d["arms"].items():
            print(f"| {k} | {a['D1_visible_rare_modes']} of {a['rare_modes_present']} | {a['rare_reports_right']} of {a['rare_reports_total']} | "
                  f"{a['D2_false_rare_placements']:.0f} {a['D2_ci95']} | {a['D3_precision_rare']:.3f} {a['D3_ci95']} | "
                  f"{a['rare_placements_total']:.0f} | {a['D5_spearman_all']:.3f} / {a['D5_spearman_1-5']:.3f} |")
        print("\nP2:", d["P2_selector_minus_encoder_false_rare"])


def tex():
    """Rows of the paper's selector table: Ours, then each comparison as difference with the code-level interval."""
    for corpus in ("maude", "sdr"):
        R = load(corpus)
        if R is None:
            continue
        for year in ("2026", "2025"):
            Y = R["years"][year]
            for i, llm in enumerate(LLMS):
                S = Y["selectors"].get(llm, {})
                if "tests" not in S:
                    continue
                m = S["macro_f1"]
                cols = []
                for short, key, _ in (TESTS[0], TESTS[3], TESTS[2], TESTS[1]):  # paper order: best single, vote, rule, cascade
                    c = S["tests"][key]
                    p = f"{c['item']['point']:+.3f}"
                    p = f"\\textbf{{{p}}}" if c["holds"] else p
                    cols.append(f"\\makecell{{{p}\\\\{{[{c['code']['ci95'][0]:.3f}, {c['code']['ci95'][1]:.3f}]}}}}")
                lead = f"{DS[corpus]} {year}" if i == 0 else ""
                print(f"{lead} & {NAME[llm]} & {m['selector']['all']:.3f} & " + " & ".join(cols) +
                      f" & {m['oracle']['all']:.3f} \\\\")
            print("\\addlinespace")


if __name__ == "__main__":
    (tex if "tex" in sys.argv[1:] else md)()
