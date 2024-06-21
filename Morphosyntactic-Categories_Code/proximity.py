import math

import openpyxl
import openpyxl.styles
import pandas
from tqdm import tqdm, trange
import numpy as np
import numpy.linalg as npl
import re
import zipfile
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("-f1", "--feature1")
parser.add_argument("-v1", "--value1")
parser.add_argument("-f2", "--feature2", required=False)
parser.add_argument("-v2", "--value2", required=False)
args = parser.parse_args()

indent = 0


def dot_product(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])


def manhattan_normalizer(vec):
    c = sum(vec)
    if c == 0:
        return vec
    return list(map(lambda n: n / c, vec))


def square_normalizer(vec):
    c = dot_product(vec, vec)
    if c == 0:
        return vec
    return list(map(lambda n: n / c, vec))


def is_name_in_sheets(sheetname, filename):
    wb = openpyxl.load_workbook(filename)
    if sheetname in wb.sheetnames:
        return True
    wb.close()
    return False


def tabulize(grammar_feature):
    if is_name_in_sheets(f"Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}", "Proximity.xlsx"):
        return
    print(f"Tabulizing for {grammar_feature[0]}={grammar_feature[1]}")
    database = pandas.read_excel(
        "RelDep_Matches.xlsx",
        sheet_name=f"RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}"
    )
    database.fillna(0, inplace=True)
    # max_row = len(database["Treebank"])
    for col in (pbar := tqdm(database, colour='#7d1dd3', total=574)):
        if col == "Treebank":
            pass
        else:
            pbar.set_description(f"Normalizing: {col}")
            c = [database[col][i] for i in range(2, len(database[col]) - 1)]
            c = manhattan_normalizer(c)
            for i in range(len(c)):
                database.loc[i + 2, col] = c[i]
    wb = openpyxl.load_workbook("Proximity.xlsx")
    ws = wb.create_sheet(f"Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}")
    ws.cell(1, 1).value = "Treebank"
    treetab = {}
    mat = database[database.columns[1:]].to_numpy()[2:-1]
    for i in range(len(mat[0])):
        d = npl.norm(mat[:, i], ord=2)
        if d != 0:
            mat[:, i] /= d
    dot_mat = np.matmul(np.transpose(mat), mat)
    i = 2
    for (vec1, vec2) in [(vec1, vec2) for vec1 in database for vec2 in database]:
        if vec1 == "Treebank":
            pass
        else:
            if vec1 not in treetab:
                ws.cell(1, i).value = vec1
                ws.cell(i, 1).value = vec1
                treetab[vec1] = i
                i += 1
            if vec2 == "Treebank":
                pass
            else:
                if vec2 not in treetab:
                    ws.cell(1, i).value = vec2
                    ws.cell(i, 1).value = vec2
                    treetab[vec2] = i
                    i += 1
                row = treetab[vec1]
                col = treetab[vec2]
                d = dot_mat[row - 2, col - 2]
                ws.cell(row, col).value = d
                ws.cell(row, col).fill = openpyxl.styles.fills.PatternFill(
                    patternType='solid',
                    fgColor=openpyxl.styles.colors.Color(
                        indexed=math.floor(d * 5 + 2)
                    )
                )
    wb.save("Proximity.xlsx")


def tabulize_pair(gf1, gf2, res_wb):
    if f"Proximity_{gf1[0]}={gf1[1]}_{gf2[0]}={gf2[1]}" in res_wb.sheetnames:
        return
    wb = openpyxl.load_workbook("RelDep_Matches.xlsx")
    reldep_table = {}
    i = 0
    ws = wb[f"RelDep_matching_{gf1[0]}={gf1[1]}"]
    max_col1 = ws.max_column + 1
    max_row1 = ws.max_row
    all_vec = [{} for _ in range(2, max_col1)]
    for col in range(2, max_col1):
        all_vec[col - 2]["Name"] = f"{ws.cell(1, col).value}_{gf1[0]}={gf1[1]}"
        all_vec[col - 2]["Sum"] = ws.cell(max_row1, col).value
    pbar = tqdm(total=(max_row1 - 4) * (max_col1 - 2), colour="#7d1dd3", desc=f"Reading {gf1}")
    for row in range(4, max_row1):
        reldep = ws.cell(row, 1).value
        if reldep not in reldep_table:
            reldep_table[reldep] = i
            i += 1
        for col in range(2, max_col1):
            if all_vec[col - 2]["Sum"]:
                d = int(ws.cell(row, col).value) if ws.cell(row, col).value else 0
                all_vec[col - 2][reldep] = d / all_vec[col - 2]["Sum"]
            pbar.update(1)
    pbar.close()
    cur_col = len(all_vec)

    ws = wb[f"RelDep_matching_{gf2[0]}={gf2[1]}"]
    max_col2 = ws.max_column + 1
    max_row2 = ws.max_row
    all_vec = all_vec + [{} for _ in range(2, max_col2)]
    for col in range(2, max_col2):
        all_vec[cur_col + col - 2]["Name"] = f"{ws.cell(1, col).value}_{gf2[0]}={gf2[1]}"
    pbar = tqdm(total=(max_row2 - 4) * (max_col2 - 2), colour="#7d1dd3", desc=f"Reading {gf2}")
    for row in range(4, max_row2):
        reldep = ws.cell(row, 1).value
        if reldep not in reldep_table:
            reldep_table[reldep] = i
            i += 1
        for col in range(2, max_col2):
            if all_vec[col - 2]["Sum"]:
                d = int(ws.cell(row, col).value) if ws.cell(row, col).value else 0
                all_vec[cur_col + col - 2][reldep] = d / all_vec[col - 2]["Sum"]
            pbar.update(1)
    pbar.close()
    cur_col = len(all_vec)
    wb.close()

    mat = np.matrix(np.array([[0. for _ in range(cur_col)] for _ in range(i)]))
    for reldep in tqdm(reldep_table, desc="Reporting Matrix", position=42 * indent, colour="#ffe500"):
        row = reldep_table[reldep]
        for k in range(cur_col):
            mat[row, k] = all_vec[k].get(reldep, 0.)

    for col in tqdm(range(cur_col), desc="Normalizing Matrix", colour="#ffe500"):
        d = npl.norm(mat[:, col])
        if d != 0:
            mat[:, col] /= d
    dot_mat = np.matmul(np.transpose(mat), mat)

    ws = res_wb.create_sheet(f"Proximity_{gf1[0]}={gf1[1]}_{gf2[0]}={gf2[1]}")
    ws.cell(1, 1).value = "Treebank+GF"
    pbar = tqdm(desc="Reporting Values in Table", total=cur_col * cur_col, colour="#7d1dd3")
    for i in range(cur_col):
        ws.cell(1, i + 2).value = all_vec[i]["Name"]
        ws.cell(i + 2, 1).value = all_vec[i]["Name"]
        for j in range(cur_col):
            d = dot_mat[i, j]
            ws.cell(i + 2, j + 2).value = d
            ws.cell(i + 2, j + 2).fill = openpyxl.styles.fills.PatternFill(
                patternType='solid',
                fgColor=openpyxl.styles.colors.Color(
                    indexed=math.floor(d * 5 + 2)
                )
            )
            pbar.update(1)
    pbar.close()


def is_low(n):
    if n == np.nan:
        return False
    if 0 < n < 0.2:
        return True
    return False


def is_high(n):
    if n == np.nan:
        return False
    if 0.8 < n < 1:
        return True
    return False


def figurifier(grammar_feature):
    results = ""
    proximities = pandas.ExcelFile(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.xlsx")
    results += r"\begin{table}[H]" + "\n\t" + r"\centering" + "\n\t" + r"\begin{NiceTabular}{" + r"c" * (len(proximities.sheet_names)) + "}\n\t\t"
    results += r"Proximity with: "
    for s in proximities.sheet_names:
        if s != "Sheet":
            gf = s.split("_")[2]
            results += f"& {gf} "
    results += r"\\" + "\n"

    value_dict = {
        "Median": {},
        "Mean": {},
        "N_Low": {},
        "N_High": {},
        "First Quartile": {},
        "Third Quartile": {},
        }
    for s in proximities.sheet_names:
        if s != "Sheet":
            ws = proximities.parse(s)
            ws = ws[ws.columns[1:]].to_numpy()
            value_dict["Median"][s] = round(np.nanmedian(ws), 5)
            value_dict["First Quartile"][s] = round(np.nanquantile(ws, 0.25), 5)
            value_dict["Third Quartile"][s] = round(np.nanquantile(ws, 0.75), 5)
            value_dict["Mean"][s] = round(np.nanmean(ws), 5)
            value_dict["N_Low"][s] = round(np.count_nonzero((ws > 0) & (ws < .2)), 5)
            value_dict["N_High"][s] = round(np.count_nonzero((ws < 1) & (ws > .8)), 5)

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
    with open(f"DuoProximity/{grammar_feature[0]}={grammar_feature[1]}_Proximity.tex", 'a') as f:
        f.write(results)


gf = [("Case", "Voc"), ("Case", "Nom"), ("Case", "Dat"), ("Case", "Acc"), ("Case", "Gen"), ("Case", "Loc")]

if __name__ == "__main__":
    """gf1 = (args.feature1, args.value1)
    tabulize(gf1)
    gf2 = (args.feature2, args.value2)
    if gf2[0] is not None and gf2[1] is not None:
        tabulize_pair(gf1, gf2)"""
    # tabulize_pair(("Case", "Nom"), ("Case", "Acc"))
    try:
        os.mkdir('DuoProximity')
    except FileExistsError:
        pass
    for i in range(len(gf)):
        g1 = gf[i]
        # try:
        #     res_wb = openpyxl.load_workbook(f"DuoProximity/{g1[0]}={g1[1]}_Proximity.xlsx")
        # except FileNotFoundError:
        #     res_wb = openpyxl.Workbook()
        # for j in range(len(gf)):
        #     g2 = gf[j]
        #     print(f"Tabulating Pair {g1[0]}={g1[1]}, {g2[0]}={g2[1]}")
        #     tabulize_pair(g1, g2, res_wb)
        #     print("")
        # res_wb.save(f"DuoProximity/{g1[0]}={g1[1]}_Proximity.xlsx")
        print(f"Texifying {g1}")
        figurifier(g1)
