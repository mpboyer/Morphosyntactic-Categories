import pandas
import numpy as np
import argparse
import itertools
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "ud-treebanks-v2.14"
MODE = ""
VECTOR_DIR = f"{MODE}_Case_RelDep_Matches" if MODE else "Case_RelDep_Matches"
SAVE_DIR = f"{MODE}_Case_Proximities" if MODE else "Case_Proximities"


def figurifier(grammar_feature):
    results = r"\renewcommand{\arraystretch}{1.1}" + "\n"
    proximities = pandas.ExcelFile(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.xlsx")
    results += r"\begin{table}[H]" + "\n\t" + r"\centering" + "\n\t" + r"\begin{NiceTabular}{" + r"c" * (
        len(proximities.sheet_names)) + "}\n\t\t"
    results += r"Proximity with: "
    for s in proximities.sheet_names:
        if s != "Sheet":
            concurrent_case = s.split("_")[2]
            results += f"& {concurrent_case} "
    results += r"\\" + "\n"

    value_dict = {
        "Median": {},
        "Mean": {},
        "NLow": {},
        "NHigh": {},
        "First Quartile": {},
        "Third Quartile": {},
    }
    for s in proximities.sheet_names:
        if s != "Sheet":
            ws = proximities.parse(s)
            ws = ws[ws.columns[1:]].to_numpy()[574:, :574]
            value_dict["Median"][s] = round(np.nanmedian(ws), 5)
            value_dict["First Quartile"][s] = round(np.nanquantile(ws, 0.25), 5)
            value_dict["Third Quartile"][s] = round(np.nanquantile(ws, 0.75), 5)
            value_dict["Mean"][s] = round(np.nanmean(ws), 5)
            value_dict["NLow"][s] = round(np.count_nonzero((ws > 0) & (ws < .2)), 5)
            value_dict["NHigh"][s] = round(np.count_nonzero((ws < 1) & (ws > .8)), 5)

    for stat in value_dict:
        results += f"\t\t{stat} "
        for g in value_dict[stat]:
            results += f"& {value_dict[stat][g]} "
        results += r"\\" + "\n"

    results += "\t" + r"\CodeAfter" + "\n\t\t"
    results += r"\begin{tikzpicture}" + "\n\t\t\t"
    results += r"\foreach \i in {1,...," + f"{len(value_dict) + 2}" + r"}" + "\n\t\t\t\t"
    results += r"{\draw[draw=vulm] (1|-\i) -- (" + f"{len(proximities.sheet_names) + 1}|-" + r"\i);}" + "\n\t\t\t"
    results += r"\draw[draw=vulm] (2|-1)--(2|-" + f"{len(value_dict) + 2});"
    results += r"\end{tikzpicture}" + "\n\t"
    results += r"\end{NiceTabular}" + "\n\t"
    results += r"\caption{Proximities for " + f"{grammar_feature[0]}={grammar_feature[1]}" + "}\n"
    results += r"\end{table}"
    with open(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.tex", 'w') as f:
        f.write(results)


gf = ["Abl", "Acc", "Dat", "Gen", "Ins", "Loc", "Nom", "Voc"]


def csv_figurifier_angles():
    preamble = r"\renewcommand{\arraystretch}{1.1}" + "\n" + r"\begin{table}[H]" + "\n\t" + r"\centering" + "\n\t" + r"\resizebox{\textwidth}{!}{\begin{NiceTabular}{" + r"c" * (
        len(gf) + 1) + "}\n\t\t"
    preamble += r"Proximity with: "
    results = dict([(c, {
        "First Quartile": {},
        "Median": {},
        "Third Quartile": {},
        "Mean": {},
    }) for c in gf])

    for g in gf:
        preamble += f"& {g} "
    preamble += r"\\" + "\n"

    for case1, case2 in tqdm(itertools.product(gf, gf), colour="#7d1dd3", total=64):
        ws = pandas.read_csv(f"{SAVE_DIR}/Angles/DuoProximity_for_{case1}_and_{case2}.csv").values[:, 1:]
        results[case1]["Median"][case2] = round(np.nanmedian(ws), 5)
        results[case1]["First Quartile"][case2] = round(np.nanquantile(ws, 0.25), 5)
        results[case1]["Third Quartile"][case2] = round(np.nanquantile(ws, 0.75), 5)
        results[case1]["Mean"][case2] = round(np.nanmean(ws), 5)

    print(results['Nom'])
    print(results['Voc'])

    for case in tqdm(results, colour="#f7e500"):
        value_dict = results[case]
        result_string = preamble
        for stat in value_dict:
            result_string += f"\t\t{stat} "
            for g in value_dict[stat]:
                result_string += f"& {value_dict[stat][g]:.3f} "
            result_string += r"\\" + "\n"
        result_string += "\t" + r"\CodeAfter" + "\n\t\t"
        result_string += r"\begin{tikzpicture}" + "\n\t\t\t"
        result_string += r"\foreach \i in {1,...," + f"{len(value_dict) + 2}" + r"}" + "\n\t\t\t\t"
        result_string += r"{\draw[draw=vulm] (1|-\i) -- (" + f"{len(gf) + 2}|-" + r"\i);}" + "\n\t\t\t"
        result_string += r"\draw[draw=vulm] (2|-1)--(2|-" + f"{len(value_dict) + 2});"
        result_string += r"\end{tikzpicture}" + "\n\t"
        result_string += r"\end{NiceTabular}}" + "\n\t"
        result_string += r"\caption{Proximities for " + f"Case={case}" + "}\n"
        result_string += r"\end{table}"
        save_path = f"Figures/Visualisations/Angles_Case={case}_Proximity_{MODE}.tex" if MODE else f"Figures/Visualisations/Angles_Case={case}_Proximity.tex"
        with open(save_path, 'w') as f:
            f.write(result_string)


csv_figurifier_angles()


def csv_figurifier_distances():
    preamble = r"\renewcommand{\arraystretch}{1.1}" + "\n" + r"\begin{table}[H]" + "\n\t" + r"\centering" + "\n\t" + r"\resizebox{\textwidth}{!}{\begin{NiceTabular}{" + r"c" * (
        len(gf) + 1) + "}\n\t\t"
    preamble += r"Proximity with: "
    results = dict([(c, {
        "First Quartile": {},
        "Median": {},
        "Third Quartile": {},
        "Mean": {},
    }) for c in gf])

    for g in gf:
        preamble += f"& {g} "
    preamble += r"\\" + "\n"

    for case1, case2 in tqdm(itertools.product(gf, gf), colour="#7d1dd3", total=64):
        ws = pandas.read_csv(f"{SAVE_DIR}/Distances/Distances_{case1}_{case2}.csv").values[:, 1:]
        results[case1]["Median"][case2] = round(np.nanmedian(ws), 5)
        results[case1]["First Quartile"][case2] = round(np.nanquantile(ws, 0.25), 5)
        results[case1]["Third Quartile"][case2] = round(np.nanquantile(ws, 0.75), 5)
        results[case1]["Mean"][case2] = round(np.nanmean(ws), 5)

    for case in tqdm(results, colour="#f7e500"):
        value_dict = results[case]
        result_string = preamble
        for stat in value_dict:
            result_string += f"\t\t{stat} "
            for g in value_dict[stat]:
                result_string += f"& {value_dict[stat][g]:.3f} "
            result_string += r"\\" + "\n"
        result_string += "\t" + r"\CodeAfter" + "\n\t\t"
        result_string += r"\begin{tikzpicture}" + "\n\t\t\t"
        result_string += r"\foreach \i in {1,...," + f"{len(value_dict) + 2}" + r"}" + "\n\t\t\t\t"
        result_string += r"{\draw[draw=vulm] (1|-\i) -- (" + f"{len(gf) + 2}|-" + r"\i);}" + "\n\t\t\t"
        result_string += r"\draw[draw=vulm] (2|-1)--(2|-" + f"{len(value_dict) + 2});"
        result_string += r"\end{tikzpicture}" + "\n\t"
        result_string += r"\end{NiceTabular}}" + "\n\t"
        result_string += r"\caption{Proximities for " + f"Case={case}" + "}\n"
        result_string += r"\end{table}"
        save_path = f"Figures/Visualisations/Distances_Case={case}_Proximity_{MODE}.tex" if MODE else f"Figures/Visualisations/Distances_Case={case}_Proximity.tex"
        with open(save_path, 'w') as f:
            f.write(result_string)


csv_figurifier_distances()
