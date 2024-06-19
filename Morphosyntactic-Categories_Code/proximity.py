import math

import openpyxl
import openpyxl.styles
import pandas
from tqdm import tqdm
import numpy as np
import numpy.linalg as npl


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


def tabulize(grammar_feature):
    print(f"Tabulizing for {grammar_feature[0]}={grammar_feature[1]}")
    database = pandas.read_excel("RelDep_Matches.xlsx",
                                 sheet_name=f"RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}")
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
    wb = openpyxl.load_workbook("RelDep_Matches.xlsx")
    if f"Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}" in wb.sheetnames:
        return
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
                ws.cell(row, col).fill = openpyxl.styles.fills.PatternFill(patternType='solid',
                                                                           fgColor=openpyxl.styles.colors.Color(
                                                                               indexed=math.floor(d * 5 + 2)))
    wb.save("RelDep_Matches.xlsx")
