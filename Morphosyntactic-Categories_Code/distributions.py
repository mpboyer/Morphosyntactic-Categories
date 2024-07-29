import itertools
import os
from typing import Callable

import numpy as np
import ot
import pandas
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from linalg import distance, l2_distance, manhattan_distance
import argparse
from toolz.functoolz import curry
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "../ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = f"../{MODE}_Per_Appos_Stats" if MODE else "../Per_Appos_Stats"


def get_basis(case):
    return pandas.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv").columns[4:]


def overall_basis():
    basis = set()
    for case in filter(lambda t: t[16:20] == 'Case', os.listdir(VECTOR_DIR)):
        basis |= set(get_basis(case.split('=')[1][:-4]))
    return basis


def cast_case_to_basis(case, vector):
    basis = overall_basis()
    case_basis = get_basis(case)
    vector = {c: vector[i] for i, c in enumerate(case_basis)}
    m = np.array([0. for _ in basis])
    for i, b in enumerate(basis):
        m[i] = vector.get(b, 0.)
    return m


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
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    mean_vector = np.array([np.mean(data[:, i]) for i in range(data.shape[1])])
    return mean_vector


def euclidean_to_typical(case, typical):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distances = {case_data_set.loc[i, 'Treebank']: distance(typical[0], row) for (i, row) in enumerate(data)}
    return distances, np.sum(np.array([distances[k] for k in distances]) ** 2)


def euclidean_to_any(case):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]

    energies = {}
    for j, row in enumerate(data):
        distances = {case_data_set.loc[i, 'Treebank']: distance(row, other) ** 2 for (i, other) in enumerate(data)}
        energies[case_data_set.loc[j, 'Treebank']] = np.sum(np.array([distances[k] for k in distances]))

    results = "\n".join(f"Énergie pour {k} prototypique = {v:.3f}" for (k, v) in energies.items())
    with open("energies.txt", "w") as file:
        file.write(results)
        list_energy = [energies[k] for k in energies]
        file.write(f"\nMoyenne: {np.mean(list_energy):.3f}, Min: {np.min(list_energy):.3f}")


def kl(v1, v2):
    restil = []
    for a, b in zip(v1, v2):
        if b == 0.:
            continue
        restil.append(a * np.log(a / b) if a != 0. else 0.)
    return np.sum(restil)


def distance_to_any(case, dist: Callable):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]

    energies = {}
    for j, row in enumerate(data):
        distances = {case_data_set.loc[i, 'Treebank']: dist(row, other) for (i, other) in enumerate(data)}
        energies[case_data_set.loc[j, 'Treebank']] = np.mean(np.array([distances[k] for k in distances]))

    results = "\n".join(f"Énergie pour {k} prototypique selon {dist.__name__} = {v:.3f}" for (k, v) in energies.items())
    with open(f"{SAVE_DIR}/{case}/{case}_energies_{dist.__name__}.txt", "w") as file:
        file.write(results)
        list_energy = [energies[k] for k in energies]
        file.write(
            f"\nMoyenne: {np.mean(list_energy):.3f}, Min: {np.min(list_energy):.3f}, Max: {np.max(list_energy):.3f}"
        )


def distance_to_typical(case, dist: Callable, typical, name):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distances = {case_data_set.loc[i, 'Treebank']: dist(typical, row) for (i, row) in enumerate(data)}
    e = np.sum(np.array([distances[k] for k in distances]) ** 2)
    res = '\n'.join(f"Distance {k} - {name} = {v:.3f}" for (k, v) in distances.items())
    with open(f"{SAVE_DIR}/{case}/{case}_mean_to_barycenter_{dist.__name__}_{name}.txt", "w") as f:
        f.write(res)
        f.write("\n")
        f.write(f"Énergie = {e:.3f}")


def wasserstein_barycenter_plot(case):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    frequencies = np.array([0. for _ in range(data.shape[1])])
    for j in range(data.shape[1]):
        for row in data:
            frequencies[j] += (row[j] >= 3)
    frequencies /= float(len(data))
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distributions = data.T
    n_distributions = distributions.shape[1]
    n_bins = len(data[0])
    loss = np.array([[1. for _ in range(n_bins)] for _ in range(n_bins)])
    loss -= np.identity(n_bins)
    loss /= np.sum(loss)
    weights = [1 / n_distributions] * n_distributions
    bary_l2 = distributions.dot(weights)
    bary_wass = ot.lp.barycenter(distributions, loss, weights=weights, verbose=False)
    x = np.arange(n_bins, dtype=np.float64)

    f, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, num=1)
    # plt.title(f"Wasserstein Barycenter for Distributions on {case}")

    for row in distributions.T:
        ax1.scatter(x, row, marker='.', color="#ffe500")
    ax1.set_title('Distributions')

    ax2.bar(x - 1 / 4, bary_l2, width=1 / 5, color='r', label='l2')
    ax2.bar(x, bary_wass, width=1 / 5, color='#ffe500', label='Wasserstein')
    ax2.bar(x + 1 / 4, frequencies, width=1 / 5, color="#7d1dd3", label='Apparition Frequency')
    ax2.set_title('Barycenters')

    plt.legend()
    savepath = f"Figures/Visualisations/{MODE}_Wasserstein_Barycenter_{case}.pdf" if MODE else f"Figures/Visualisations/Wasserstein_Barycenter_{case}.pdf"
    plt.savefig(savepath)
    return bary_wass


def wasserstein_barycenter(case):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distributions = data.T
    n_distributions = distributions.shape[1]
    n_bins = len(data[0])
    loss = ot.utils.dist0(n_bins)
    loss /= loss.max()
    weights = [1 / n_distributions] * n_distributions
    bary_wass = ot.lp.barycenter(distributions, loss, weights=weights, verbose=False)
    return bary_wass


def compute_barycenters():
    distance_functions = [wasserstein_distance]
    cases = ["Acc", "Dat", "Nom", "Gen", "Abs", "Erg", "Loc", "Ins", "Abl"]

    try:
        os.mkdir(SAVE_DIR)
    except FileExistsError:
        pass

    for studied_case in (pbar1 := tqdm(cases, colour="#7d1dd3")):
        pbar1.set_description(f"Computing Barycenters for {studied_case}")
        wasserstein_mean = wasserstein_barycenter_plot(studied_case)
        uniform_mean = uniform_mean_distrib(studied_case)
        with open(f"{SAVE_DIR}/barycenters.txt", "a") as file:
            file.write(f"{studied_case=}\n")
            file.write(f"{", ".join(str(w) for w in wasserstein_mean)}\n")
            file.write(f"{", ".join(str(w) for w in uniform_mean)}\n\n\n")
        savepath = f"{MODE}_barycenters.txt" if MODE else "barycenters.txt"
        with open(savepath, "a") as file:
            file.write(f"{studied_case=}\n")
            file.write(f"{", ".join(str(w) for w in wasserstein_mean)}\n")
            file.write(f"{", ".join(str(w) for w in uniform_mean)}\n\n\n")
        for dfunc in (pbar2 := tqdm(
                distance_functions, total=3 * len(distance_functions), colour="#ffe500", position=1, leave=False
        )
        ):
            try:
                os.mkdir(f"{SAVE_DIR}/{studied_case}")
            except FileExistsError:
                pass
            pbar2.set_description("Distance to Any")
            distance_to_any(studied_case, dfunc)
            pbar2.update(1)
            pbar2.set_description("Distance to Uniform Barycenter")
            distance_to_typical(studied_case, dfunc, uniform_mean, name="Uniform_Barycenter")
            pbar2.update(1)
            pbar2.set_description("Distance to Wasserstein Barycenter")
            distance_to_typical(studied_case, dfunc, wasserstein_mean, name="Wasserstein_Barycenter")
        pbar2.update(1)


def get_barycenters():
    savepath = f"{MODE}_barycenters.txt" if MODE else "barycenters.txt"
    with open(savepath, 'r') as f:
        results = f.read()

    barycenters = list(
        map(
            lambda tup: (
                tup[0][-4:-1], (list(map(float, tup[1][:].split(','))), list(map(float, tup[2][1:-1].split(','))))),
            list(map(lambda t: tuple(t.split("\n")), results.split('\n\n\n')))[:-1]
        )
    )
    return barycenters


def print_barycenters():
    barycenters = get_barycenters()
    features = [list(get_basis(b[0])) for b in barycenters]

    format_float = lambda t: f"{t:.3f}"
    savepath = f"{MODE}_barycenters_joli.csv" if MODE else "barycenters_joli.csv"
    results = "\n\n\n".join(
        f"{barycenters[i][0]},{",".join(f)}\nWasserstein,{",".join(map(format_float, barycenters[i][1][0]))}\nUniform,{",".join(map(format_float, barycenters[i][1][1]))}"
        for (i, f) in enumerate(features)
    )

    with open(savepath, 'w') as csv:
        csv.write(results)


def barycenter_distances(dist: Callable):
    r = get_barycenters()
    wassies = [(k[0], np.array(k[1][1])) for k in r]
    result = "\n".join(
        f"Distance {b1[0]} - {b2[0]} = {dist(cast_case_to_basis(*b1), cast_case_to_basis(*b2)):.5f}"
        for b1, b2 in itertools.combinations(wassies, 2)
    )
    save_path = f"{MODE}_uniform_barycenters_{dist.__name__}.txt" if MODE else f"uniform_barycenters_{dist.__name__}.txt"
    with open(save_path, 'w') as f:
        f.write(result)


@curry
def nearest_barycenter(barycenters, dist: Callable, case, point):
    distances = {
        b[0]: dist(point, b[1][0]) for b in barycenters
    }
    b = min(distances, key=distances.get)
    return b, distances.get(b), distances.get(case)


def bary_space_part(dist: Callable, case: str):
    barycenters = get_barycenters()
    compute = nearest_barycenter(barycenters, dist, case)
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    results = {case_data_set.loc[i, 'Treebank']: compute(p) for (i, p) in enumerate(data)}
    text = f"Nombre d'accusatifs = {len(results)}\n" + "\n".join(f"{k}, Nearest={v[0]}, Distance={v[1]:.5f}, Self={v[2]:.5f}, Diff={v[2]-v[1]:.5f}" for k, v in results.items())
    confusion = {}
    for k, v in results.items():
        confusion[v[0]] = confusion.get(v[0], 0) + 1
    text += f"\nConfusion:\n{f"\n".join(f"\t{k}: {v}" for k, v in confusion.items())}"
    savepath = f"Figures/KNN/{MODE}_barycenters_{dist.__name__}_{case}.txt" if MODE else f"Figures/KNN/barycenters_{dist.__name__}_{case}.txt"
    with open(savepath, 'w') as f:
        f.write(text)


def data_from_case_list(cases):
    case1 = cases[0]
    case_data_set = pandas.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case1}.csv")
    case_data_set.insert(1, "Case", case1, True)
    case_data_set["Treebank"].map(lambda n: n + f"_{case1}")

    for case in cases[1:]:
        case_set = pandas.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv")
        case_set['Treebank'].map(lambda n: n + f"_{case}")
        case_set.insert(1, "Case", case, True)
        case_data_set = case_data_set._append(case_set, ignore_index=True)

    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )

    for row in case_data_set.index:
        for column in case_data_set.columns[5:]:
            d = case_data_set.loc[row, 'Total']
            case_data_set.loc[row, column] /= d
    return case_data_set


def gen_knn(k, cases, dist: Callable):
    case_data_set = data_from_case_list(cases)
    points = case_data_set.values[:, 5:]
    nbrs = NearestNeighbors(n_neighbors=k, metric=dist, n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    # graph = nbrs.kneighbors_graph(points).toarray() - np.identity(len(points))
    knn_wrapped = [
        [(case_data_set.loc[index, 'Treebank'], case_data_set.loc[index, 'Case'], distances[i][j]) for (j, index) in
         enumerate(row)] for i, row in enumerate(indices)]
    result = "\n\n".join(["\n".join(str(t) for t in row) for row in knn_wrapped])
    savepath = f"Figures/KNN/{MODE}_{k}nn_{dist.__name__}_{"_".join(cases)}.txt" if MODE else f"Figures/KNN/{k}nn_{dist.__name__}_{"_".join(cases)}.txt"
    with open(savepath, 'w') as f:
        f.write(result)


def count_knn(k, cases, dist: Callable):
    case_dict = dict([(c, i) for i, c in enumerate(cases)] + [(i, c) for i, c in enumerate(cases)])
    case_data_set = data_from_case_list(cases)
    points = case_data_set.values[:, 5:]
    nbrs = NearestNeighbors(n_neighbors=k, metric=dist, n_jobs=-1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    knn_wrapped = [
        [(case_data_set.loc[index, 'Treebank'], case_data_set.loc[index, 'Case'], distances[i][j]) for (j, index) in
         enumerate(row)] for i, row in enumerate(indices)]
    result = "\n\n".join(["\n".join(str(t) for t in row) for row in knn_wrapped])
    savepath = f"Figures/KNN/{MODE}_{k}nn_{dist.__name__}_{"_".join(cases)}.txt" if MODE else f"Figures/KNN/{k}nn_{dist.__name__}_{"_".join(cases)}.txt"
    with open(savepath, 'w') as f:
        f.write(result)
    confusion_matrix = np.array([[0 for _ in cases] for _ in cases])
    for nn in knn_wrapped:
        c1 = nn[0][1]
        values = {c: 0 for c in cases}
        for p in nn[1:]:
            values[p[1]] += 1
        confusion_matrix[case_dict[c1], case_dict[max(values, key=values.get)]] += 1
    result = f"Case,{",".join(cases)}\n"
    result += "\n".join(
        f"{case_dict[i]},{",".join(f"{conf:.3f}" for conf in row)}" for i, row in enumerate(confusion_matrix))
    savepath = f"Figures/KNN/{MODE}_{k}nn_{dist.__name__}_{"_".join(cases)}.csv" if MODE else f"Figures/KNN/{k}nn_{dist.__name__}_{"_".join(cases)}.csv"
    with open(savepath, 'w') as csv:
        csv.write(result)


def better_wasser(v1, c1, v2, c2):
    return wasserstein_distance(cast_case_to_basis(c1, v1), cast_case_to_basis(c2, v2))


def appos_barycenters(appos):
    try:
        os.mkdir(SAVE_DIR)
    except FileExistsError:
        pass

    appos_data_sets = pandas.read_csv(f"{VECTOR_DIR}/RelDep_matching_Lemma={appos}.csv")
    for column in appos_data_sets.columns:
        appos_data_sets.replace(
            {
                column: np.nan}, 0., inplace=True
        )
    overall_data = appos_data_sets.values[:, 4:]
    features = appos_data_sets.columns[4:]
    broken_indices = []
    for i in range(len(overall_data)):
        try:
            overall_data[i] /= appos_data_sets.loc[i, "Total"]
        except ZeroDivisionError:
            broken_indices.append(i)

    all_data = {}
    for row, treebank in enumerate(appos_data_sets["Treebank"]):
        lang = treebank.split('_')[0]
        if row in broken_indices:
            continue
        try:
            all_data[lang] = np.concatenate((all_data[lang], np.array([overall_data[row]])), axis=0)
        except KeyError:
            all_data[lang] = np.array([overall_data[row]])

    for lang, data in all_data.items():
        distributions = data.T
        n_distributions = distributions.shape[1]
        n_bins = len(data[0])
        loss = ot.utils.dist0(n_bins)
        loss /= loss.max()
        weights = [1 / n_distributions] * n_distributions
        wasserstein_mean = ot.lp.barycenter(distributions, loss, weights=weights, verbose=False)
        uniform_mean = np.array([np.mean(data[:, i]) for i in range(data.shape[1])])


        savepath = "Figures/Barycenter_Distances/Adpos_Barycenters"
        with open(f"{savepath}/barycenter_{lang}.csv", "a") as file:
            file.write(f"{appos=}\n")
            file.write(f"Features,{",".join(features)}\n")
            file.write(f"Wasserstein, {", ".join(f"{w:.5f}" for w in wasserstein_mean)}\n")
            file.write(f"Uniform, {", ".join(f"{w:.5f}"for w in uniform_mean)}\n\n\n")


if __name__ == '__main__':
    appos_barycenters('sauf')
    # compute_barycenters()
    # barycenter_distances(wasserstein_distance)
    # bary_space_part(wasserstein_distance, 'Acc')
    # bary_space_part(wasserstein_distance, 'Nom')
    # bary_space_part(wasserstein_distance, 'Gen')
    # print_barycenters()
    # args = (5, ["Acc", "Gen", "Loc", "Nom"], wasserstein_distance)
    # gen_knn(*args)
    # count_knn(*args)
