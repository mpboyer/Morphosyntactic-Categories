from typing import Callable

import numpy as np
import ot
import pandas
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from linalg import distance, l2_distance, manhattan_distance
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "../ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_Case_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = f"../{MODE}_Per_Case_Stats" if MODE else "../Per_Case_Stats"


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
        energies[case_data_set.loc[j, 'Treebank']] = np.sum(np.array([distances[k] for k in distances]))

    results = "\n".join(f"Énergie pour {k} prototypique selon {dist.__name__} = {v:.3f}" for (k, v) in energies.items())
    with open(f"{SAVE_DIR}/{case}/{case}_energies_{dist.__name__}.txt", "w") as file:
        file.write(results)
        list_energy = [energies[k] for k in energies]
        file.write(f"\nMoyenne: {np.mean(list_energy):.3f}, Min: {np.min(list_energy):.3f}")


def distance_to_typical(case, dist: Callable, typical, name):
    case_data_set = get_data_set(case)
    data = case_data_set.values[:, 5:]
    for i in range(len(data)):
        data[i] /= case_data_set.values[i, 4]
    distances = {case_data_set.loc[i, 'Treebank']: dist(typical, row) for (i, row) in enumerate(data)}
    e = np.sum(np.array([distances[k] for k in distances]) ** 2)
    res = "\n".join(f"Distance {k} - {name} = {v:.3f}" for (k, v) in distances.items())
    with open(f"{SAVE_DIR}/{case}/{case}_mean_to_barycenter_{dist.__name__}_{name}.txt", "w") as f:
        f.write(res)
        f.write("\n")
        f.write(f"Énergie = {e:.3f}")


def wasserstein_barycenter_plot(case):
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
    bary_l2 = distributions.dot(weights)
    bary_wass = ot.lp.barycenter(distributions, loss, weights=weights, verbose=True)
    x = np.arange(n_bins, dtype=np.float64)

    f, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, num=1)
    ax1.plot(x, distributions, color="black")
    ax1.set_title('Distributions')

    ax2.plot(x, bary_l2, 'r', label='l2')
    ax2.plot(x, bary_wass, 'g', label='Wasserstein')
    ax2.set_title('Barycenters')

    plt.legend()
    plt.savefig(f"Figures/Visualisations/Wasserstein Barycenter for Distributions on {case}.pdf")
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


distance_functions = [wasserstein_distance, l2_distance, manhattan_distance, kl]

studied_case = "Dat"
print("Computing Barycenters")
wasserstein_mean = wasserstein_barycenter_plot(studied_case)
uniform_mean = uniform_mean_distrib(studied_case)
with open("{SAVE_DIR}/barycenters.txt", "a") as file:
    file.write(f"{studied_case=}\n")
    file.write(f"[{", ".join(str(w) for w in wasserstein_mean)}]\n")
    file.write(f"[{", ".join(str(w) for w in uniform_mean)}]\n\n\n")
for dfunc in (pbar2 := tqdm(
              distance_functions, total=3 * len(distance_functions), colour="#ffe500", position=1, leave=False)
              ):
    pbar2.set_description("Distance to Any")
    distance_to_any(studied_case, dfunc)
    pbar2.update(1)
    pbar2.set_description("Distance to Uniform Barycenter")
    distance_to_typical(studied_case, dfunc, uniform_mean, name="Uniform_Barycenter")
    pbar2.update(1)
    pbar2.set_description("Distance to Wasserstein Barycenter")
    distance_to_typical(studied_case, dfunc, wasserstein_mean, name="Wasserstein_Barycenter")
    pbar2.update(1)
