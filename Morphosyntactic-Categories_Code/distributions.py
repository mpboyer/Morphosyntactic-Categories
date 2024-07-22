import numpy as np
from scipy.special import rel_entr, kl_div
import pandas
from linalg import distance
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "../ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_Case_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = f"../{MODE}_Case_Proximities" if MODE else "../Case_Proximities"


def get_data_set(case):
    case_data_set = pandas.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv")
    case_data_set.insert(1, "Case", case, True)
    case_data_set["Treebank"].map(lambda n: n + f"_{case}")

    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )
    return case_data_set


def uniform_mean_distrib(case):
    case_data_set = get_data_set(case)
    features = case_data_set.columns[5:]
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    mean_vector = np.array([np.mean(data[:, i]) for i in range(data.shape[1])])
    return mean_vector, sum(mean_vector)


def euclidean_to_typical(case, typical):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distances = {case_data_set.loc[i, 'Treebank']: distance(typical[0], row) for (i, row) in enumerate(data)}
    return distances, np.sum(np.array([distances[k] for k in distances]) ** 2)


d, e = euclidean_to_typical('Acc', uniform_mean_distrib('Acc'))
res = "\n".join(f"Distance {k} - Typical = {v:.3f}" for (k, v) in d.items())
with open("uniform.txt", "w") as f:
    f.write(res)
    f.write("\n")
    f.write(f"Énergie = {e:.3f}")

