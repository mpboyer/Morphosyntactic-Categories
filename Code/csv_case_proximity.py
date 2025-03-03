import joblib
import itertools

import numpy
import pandas
import scipy.linalg
from tqdm import tqdm
import numpy as np
import numpy.linalg as npl
from operator import itemgetter
import os
import contextlib
import graphviz

from linalg import angle, distance

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--f1")
parser.add_argument("--f2")
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

UDDIR = "../ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_Case_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = f"../{MODE}_Case_Proximities" if MODE else "../Case_Proximities"


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def void_to_zero(s):
    if s == '':
        return 0.
    return float(s)


def get_all_banks():
    abank = []
    treebanks = os.listdir(UDDIR)
    for treebank in treebanks:
        content = os.listdir(f"{UDDIR}/{treebank}")
        for c in list(filter(lambda t: t[-7:] == ".conllu", content)):
            abank.append(c[:-7])
    return abank


def get_all_cases():
    cases = list(filter(lambda t: t[0] == "R", os.listdir(VECTOR_DIR)))
    return [c[-7:-4] for c in cases]


def overall_basis_csv():
    basis = set()
    for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
        basis |= set(attributes)
    return basis


def get_vector_csv(treebank, case):
    with open(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv", "r") as csv_file:
        attributes = next(csv_file).split(",")[4:]
        for tree in csv_file:
            if tree[0] == treebank:
                coordinates = tree.rstrip().split(",")[4:]
    return dict(zip(attributes, coordinates))


def get_matrix_csv(treebank):
    basis = overall_basis_csv()
    case_space = []
    cases = []
    for csv in sorted(filter(lambda t: t[-4:] == ".csv" and t[16:20] == "Case", os.listdir(f"{VECTOR_DIR}"))):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] == treebank:
                    coordinates = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                    coordinates["Total"] = void_to_zero(parsed_tree[3])
                    case_space.append(coordinates)
                    cases.append(csv[-7:-4])

    matrix = np.array([[0. for _ in case_space] for _ in basis])
    for row, b in enumerate(basis):
        for column, vector in enumerate(case_space):
            total = vector["Total"]
            if total != 0.:
                matrix[row, column] = vector.get(b, 0.) / total

    return cases, matrix


def enhanced_get_matrix_csv(treebank, treebank_case):
    basis = overall_basis_csv()
    case_space_vectors = []
    case_dict = {}
    for csv in sorted(filter(lambda t: t[-4:] == ".csv" and t[16:20] == "Case", os.listdir(f"{VECTOR_DIR}"))):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]

            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if csv[-7:-4] == treebank_case[1]:
                    if parsed_tree[0] == treebank_case[0]:
                        case_dict = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                        case_dict["Total"] = void_to_zero(parsed_tree[3])

                if parsed_tree[0] == treebank:
                    case_coordinates = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                    case_coordinates["Total"] = void_to_zero(parsed_tree[3])
                    case_space_vectors.append(case_coordinates)

    case_space_matrix = np.array([[0. for _ in case_space_vectors] for _ in basis])
    case_vector = np.array([0. for _ in basis])

    for row, b in enumerate(basis):
        case_vector[row] = case_dict.get(b, 0.) / case_dict["Total"] if case_dict["Total"] != 0. else 0.
        for column, vector in enumerate(case_space_vectors):
            case_space_matrix[row, column] = vector.get(b, 0.) / vector["Total"] if vector["Total"] != 0. else 0.

    return case_space_matrix, case_vector


def has_case(treebank, c):
    with open(f"{VECTOR_DIR}/RelDep_matching_Case={c}.csv") as csv:
        for line in csv:
            if line.split(",")[0] == treebank:
                return True
    return False


def banks_with_case():
    all_banks = set()
    for case in filter(lambda t: t[-4:] == '.csv', os.listdir(VECTOR_DIR)):
        with open(f"{VECTOR_DIR}/{case}") as csv:
            _ = next(csv)
            for line in csv:
                all_banks.add(line.split(",")[0])
    return all_banks


def treebank_case_pairs():
    return list(filter(lambda tup: has_case(*tup), ((a, c) for a in get_all_banks() for c in get_all_cases())))


def case_space_case_angle_csv(treebank, treebank_case):
    case_space_matrix, case_vector = enhanced_get_matrix_csv(treebank, treebank_case)
    dimension = scipy.linalg.orth(case_space_matrix)
    if dimension.shape[1]:
        a = angle(case_vector, dimension)
    else:
        a = np.nan
    return treebank, treebank_case[0] + f"_Case={treebank_case[1]}", a


def compute_angles_csv():
    all_banks = get_all_banks()
    basis = overall_basis_csv()
    treebanks_and_cases = treebank_case_pairs()
    angles = []
    for corpus in tqdm(all_banks, colour="#7d1dd3", leave=True, desc="Computing Angles"):
        case_space = []
        for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
            with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
                attributes = next(csv_file).rstrip().split(",")[4:]
                for tree in csv_file:
                    parsed_tree = tree.rstrip().split(",")
                    if parsed_tree[0] == corpus:
                        case_coordinates = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                        case_coordinates["Total"] = void_to_zero(parsed_tree[3])
                        case_space.append(case_coordinates)

        matrix = np.array([[0. for _ in case_space] for _ in basis])
        for row, b in enumerate(basis):
            for column, case_vec in enumerate(case_space):
                matrix[row, column] = case_vec.get(b, 0.) / case_vec["Total"] if case_vec["Total"] != 0. else 0.

        def get_angle(case_space_matrix, vector_filename):
            with open(f"{VECTOR_DIR}/RelDep_matching_Case={vector_filename[1]}.csv", "r") as csvfile:
                vec_attr = next(csvfile).rstrip().split(",")[4:]
                for frequencies in csvfile:
                    parsed_vector = frequencies.rstrip().split(",")
                    if parsed_vector[0] == vector_filename[0]:
                        case_dict = dict(zip(vec_attr, map(void_to_zero, parsed_vector[4:])))
                        case_dict["Total"] = void_to_zero(parsed_vector[3])

            vector = [0. for _ in basis]
            for coordinate, base_vector in enumerate(basis):
                vector[coordinate] = case_dict.get(base_vector, 0.) / case_dict["Total"] if case_dict[
                    "Total"] != 0. else 0.

            if case_space_matrix.shape[1]:
                a = angle(vector, case_space_matrix)
            else:
                a = np.nan
            return vector_filename[0] + f"_Case={vector_filename[1]}", a

        dimension = scipy.linalg.orth(matrix)
        with tqdm_joblib(
                tqdm(
                    desc=f"Angles for {corpus}", leave=False, colour="#ffe500", position=1,
                    total=len(treebanks_and_cases)
                )
        ) as progress_bar:
            for_corpus = dict(
                joblib.Parallel(n_jobs=8, verbose=0)(
                    joblib.delayed(get_angle)(dimension, tup) for tup in treebanks_and_cases
                )
            )
            for_corpus["Treebank"] = corpus
            angles.append(for_corpus)

    pandas.DataFrame(angles, columns=sorted(for_corpus, key=lambda t: "" if t == "Treebank" else t)).to_csv(
        f"{SAVE_DIR}/Vector_Angle_Proximity.csv", index=False
    )


def euclidean_reldep_matrix_csv(grammar_feature):
    basis = overall_basis_csv()
    treebanks = []
    with open(f"{VECTOR_DIR}/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.csv") as f:
        header = next(f).split(",")[4:]
        lines = f.readlines()
        mat = np.array([[0. for _ in lines] for _ in basis])
        for row, line in enumerate(lines):
            reldep_frequencies = line.rstrip().split(",")
            treebanks.append(f"{reldep_frequencies[0]}_{grammar_feature[0]}={grammar_feature[1]}")
            coordinate_dict = dict(zip(header, map(void_to_zero, reldep_frequencies[4:])))
            for index, column in enumerate(basis):
                mat[index, row] = coordinate_dict.get(column, 0.)
    for i in range(len(mat[0])):
        euclid = npl.norm(mat[:, i])
        if euclid != 0.:
            mat[:, i] /= euclid

    return treebanks, mat


def manhattan_reldep_matrix_csv(grammar_feature):
    basis = overall_basis_csv()
    treebanks = []
    with open(f"{VECTOR_DIR}/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.csv") as f:
        header = next(f).split(",")[4:]
        lines = f.readlines()
        mat = np.array([[0. for _ in lines] for _ in basis])
        for row, line in enumerate(lines):
            reldep_frequencies = line.rstrip().split(",")
            treebanks.append(f"{reldep_frequencies[0]}_{grammar_feature[0]}={grammar_feature[1]}")
            coordinate_dict = dict(zip(header, map(void_to_zero, reldep_frequencies[4:])))
            total = void_to_zero(reldep_frequencies[3])
            if total != 0.:
                for index, column in enumerate(basis):
                    mat[index, row] = coordinate_dict.get(column, 0.) / total

    return treebanks, mat


def tabulize_angles_csv(grammar_feature):
    try:
        with open(f"{SAVE_DIR}/Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}.csv"):
            pass
    except FileNotFoundError:
        print(f"Tabulating for {grammar_feature[0]}={grammar_feature[1]}")
        treebanks = []
        with open(f"{VECTOR_DIR}/RelDep_matching_{grammar_feature[0]}={grammar_feature[1]}.csv") as f:
            header = next(f).split(",")[4:]
            lines = f.readlines()
            mat = np.array([[0. for _ in lines] for _ in header])
            for row, line in enumerate(lines):
                reldep_frequencies = line.rstrip().split(",")
                treebanks.append(reldep_frequencies[0])
                coordinates = np.array(list(map(void_to_zero, reldep_frequencies[4:])))
                total = void_to_zero(reldep_frequencies[3])
                if total != 0.:
                    mat[:, row] = coordinates / total
        for i in range(len(mat[0])):
            euclidean_norm = npl.norm(mat[:, i])
            if euclidean_norm != 0:
                mat[:, i] /= euclidean_norm

        dot_mat = np.matmul(np.transpose(mat), mat)
        pandas.DataFrame(data=dot_mat, columns=treebanks).to_csv(
            f"{SAVE_DIR}/Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}.csv", index=False
        )


def tabulize_angle_pairs_csv():
    gf = get_all_cases()
    for c1, c2 in (pbar := tqdm(
            itertools.product(gf, gf), colour="#7d1dd3", desc="Computing Vector-Vector Angles", total=len(gf) ** 2
    )):
        pbar.set_description(f"Computing Angles for {c1} {c2}")
        gf1 = ("Case", c1)
        gf2 = ("Case", c2)
        try:
            with open(f"{SAVE_DIR}/Angles/DuoProximity_for_{gf1[1]}_and_{gf2[1]}.csv"):
                pass
        except FileNotFoundError:
            treebanks1, mat1 = euclidean_reldep_matrix_csv(gf1)
            treebanks2, mat2 = euclidean_reldep_matrix_csv(gf2)
            dot_mat = mat1.T.dot(mat2)
            result_dicts = [{
                "Treebank": f"{bank}_{c1}"} for bank in treebanks2]
            for column, bank1 in enumerate(treebanks1):
                for row, bank2 in enumerate(treebanks2):
                    result_dicts[row][bank1] = dot_mat[column, row]
            pandas.DataFrame(data=result_dicts).to_csv(
                f"{SAVE_DIR}/Angles/DuoProximity_for_{gf1[1]}_and_{gf2[1]}.csv", index=False
            )


def compute_distances_csv():
    gf = get_all_cases()
    for c1, c2 in (pbar := tqdm(
            itertools.product(gf, gf), colour="#7d1dd3", desc="Computing Vector-Vector Distances", total=len(gf) ** 2
    )):
        pbar.set_description(f"Computing Distances for {c1} {c2}")
        try:
            with open(f"{SAVE_DIR}/Distances/Distances_{c1}_{c2}.csv"):
                pass
        except FileNotFoundError:
            treebanks1, mat1 = manhattan_reldep_matrix_csv(("Case", c1))
            treebanks2, mat2 = manhattan_reldep_matrix_csv(("Case", c2))
            distance_dicts = [{
                "Treebank": c} for c in treebanks2]
            for column, treebank in enumerate(treebanks1):
                for row in range(len(treebanks2)):
                    v1 = mat1[:, column]
                    v2 = mat2[:, row]
                    distance_dicts[row][treebank] = distance(v1, v2)
            pandas.DataFrame(distance_dicts).to_csv(
                f"{SAVE_DIR}/Distances/Distances_{c1}_{c2}.csv", index=False
            )


def closest(treebank1, treebank2):
    case1, matrix1 = get_matrix_csv(treebank1)
    # print(case1)
    # print(matrix1.sum(axis=0))
    case2, matrix2 = get_matrix_csv(treebank2)
    m1_dicts = {}
    m2_dicts = {}

    distance_matrix = [["" for _ in case2] for _ in case1]
    for row in range(len(case1)):
        for col in range(len(case2)):
            distance_matrix[row][col] = str(distance(matrix1[:, row], matrix2[:, col]))[:5]
    # print(str.join("\t", tuple(case1)))
    # print(str.join("\t", tuple(case2)))
    # for d in distance_matrix:
    #     print(str.join("\t", tuple(d)))

    for col, case in enumerate(case1):
        m1_dicts[case] = dict(
            [(c, distance(matrix1[:, col], matrix2[:, i])) for i, c in enumerate(case2)]
        )

    for col, case in enumerate(case2):
        m2_dicts[case] = dict(
            [(c, distance(matrix2[:, col], matrix1[:, i])) for i, c in enumerate(case1)]
        )

    for m in m1_dicts:
        mk = min(m1_dicts[m], key=m1_dicts[m].get)
        m1_dicts[m] = mk, float(m1_dicts[m][mk])
    for m in m2_dicts:
        mk = min(m2_dicts[m], key=m2_dicts[m].get)
        m2_dicts[m] = mk, float(m2_dicts[m][mk])
    return treebank1, m1_dicts, treebank2, m2_dicts


def closest_graph(treebank1, treebank2):
    node1, edge1, node2, edge2 = closest(treebank1, treebank2)
    corpus1 = treebank1.split("-")[0]
    corpus2 = treebank2.split("-")[0]
    graphical = graphviz.Digraph(
        f"Graph of Nearest Neighbours for {MODE} Cases in {corpus1}, {corpus2}" if MODE else f"Graph of Nearest Neighbours for Cases in {corpus1}, {corpus2}"
    )
    for (k, v) in edge1.items():
        graphical.edge(f"{corpus1}_{k}", f"{corpus2}_{v[0]}", label=f"{v[1]:.3f}")
    for (k, v) in edge2.items():
        graphical.edge(f"{corpus2}_{k}", f"{corpus1}_{v[0]}", label=f"{v[1]:.3f}")

    if MODE:
        graphical.render(f"Figures/GNN/gnn_{corpus1}_{corpus2}_{MODE}_Only", format="pdf")
        try:
            os.remove(f"Figures/GNN/gnn_{corpus1}_{corpus2}_{MODE}_Only")
        except FileNotFoundError:
            pass
    else:
        graphical.render(f"Figures/GNN/gnn_{corpus1}_{corpus2}", format="pdf")
        try:
            os.remove(f"Figures/GNN/gnn_{corpus1}_{corpus2}")
        except FileNotFoundError:
            pass


def format_tuple_dict(d):
    for key, value in d.items():
        print(f'{key} : {value[0]}, Distance = {value[1]:.5f}')


def format_dict(d):
    for key, value in d.items():
        print(f'{key} : {value}')


def sample_size(treebank):
    sample_size = {}
    for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] == treebank:
                    sample_size[csv[-7:-4]] = int(parsed_tree[3])
    return sample_size


def closest_list(treebanks):
    matrices = list(get_matrix_csv(t) for t in treebanks)
    result = []
    for (i1, m1), (i2, m2) in tqdm(
            itertools.combinations(enumerate(matrices), 2), desc="Computing Distances", colour="#7d1dd3",
            total=len(matrices) * (len(matrices) - 1) / 2
    ):
        distances = np.array([[0. for _ in m2[0]] for _ in m1[0]])
        t1 = treebanks[i1]
        t2 = treebanks[i2]
        for i in range(len(m1[0])):
            for j in range(len(m2[0])):
                distances[i, j] = distance(m1[1][:, i], m2[1][:, j])
        print(m1[0], m2[0])
        if MODE:
            numpy.savetxt(f"Figures/GNN/distmat_{MODE}_" + '_'.join(treebanks) + ".csv", distances, delimiter=',', fmt='%.3f', newline='\n')
        else:
            numpy.savetxt(f"Figures/GNN/distmat_" + '_'.join(treebanks) + ".csv", distances, delimiter=',', fmt='%.3f', newline='\n')

        pair_result_dict = {}
        for row, c in enumerate(m1[0]):
            d = min(enumerate(distances[row, :]), key=itemgetter(1))
            pair_result_dict[c] = m2[0][d[0]], d[1]
        result.append((t1, t2, pair_result_dict))

        pair_result_dict = {}
        for column, c in enumerate(m2[0]):
            d = min(enumerate(distances[:, column]), key=itemgetter(1))
            pair_result_dict[c] = m1[0][d[0]], d[1]
        result.append((t2, t1, pair_result_dict))

    return result


def closest_graph_list(treebanks):
    edges = closest_list(list(treebanks.keys()))
    graphical = graphviz.Digraph(
        f"Graph of Nearest Neighbours on Cases for {MODE} in " + ",".join(
            treebanks
        ) if MODE else "Graph of Nearest Neighbours on Cases in " + ",".join(treebanks)
    )
    # graphical.graph_attr['ratio'] = '0.1'
    graphical.graph_attr['engine'] = 'circo'
    for treebank1, treebank2, distances in tqdm(edges, desc="Reporting Edges to the Graph"):
        for n1, (n2, length) in distances.items():
            cbar = ['violet', '#cea2fd', 'pink', 'magenta', 'red', '#ffb07c', 'orange', 'gold',
                    'yellow', 'chartreuse', 'green', 'turquoise', 'lightblue', 'royalblue', 'darkblue', 'deeppurple']
            c = str(cbar[int(np.floor((len(cbar) / 2) * length))])
            graphical.edge(
                f"{treebanks[treebank1]} {n1}", f"{treebanks[treebank2]} {n2}", label=f"{length:.3f}", color=c,
                penwidth='2.0'
            )

    # graphical = graphical.unflatten(stagger=3)
    if MODE:
        graphical.render(f"Figures/GNN/gnn_{MODE}_Case_Only_" + "_".join(treebanks), format="pdf")
    else:
        graphical.render("Figures/GNN/gnn_Case_Only_" + "_".join(treebanks), format="pdf")


studied_languages = ['tr', 'sk', 'ab', 'eu', 'fi', 'hit', 'ta', 'wbp']

russian_czech = {
    'cs_cltt-ud-dev': 'Czech',
    'ru_gsd-ud-dev': 'Russian'}


def tmp():
    pcbar = ['violet', 'lavender', 'pink', 'magenta', 'red', 'peach', 'orange', 'gold',
             'yellow', 'chartreuse', 'green', 'turquoise', 'light blue', 'royal blue', 'dark blue', 'deep purple']
    cbar = list(f"xkcd:{t}" for t in pcbar)
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(1, 6), layout='constrained')
    cbar2 = LinearSegmentedColormap.from_list('cbar', cbar)
    norm = mpl.colors.Normalize(vmin=0, vmax=2)
    ax.invert_yaxis()
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cbar2),
        cax=ax, ticks=np.arange(0, 2.1, 0.2)
    )
    plt.savefig("bar.pdf", format="pdf")


if __name__ == "__main__":
    # # compute_distances_csv()
    # # tabulize_angle_pairs_csv()
    # # compute_angles_csv()
    # results = []
    # banks = banks_with_case()
    # for bank1, bank2 in tqdm(itertools.combinations(banks, 2), total=len(banks) * (len(banks) - 1) /  2):
    #     t1, m1, t2, m2 = closest(bank1, bank2)
    #     r = "\n".join(f"{c} in {t1} closest to {m1[c][0]} in {t2}" for c in m1 if c in m2 and m1[c][0] != c)
    #     r += "\n"
    #     r += "\n".join(f"{c} in {t2} closest to {m2[c][0]} in {t1}" for c in m2 if c in m1 and m2[c][0] != c)
    #     results.append(r)
    # with open(f"{SAVE_DIR}/diff.txt", 'w') as f:
    #     f.write("\n\n".join(results))
    closest_list(['cs_cltt-ud-dev', 'cs_cltt-ud-dev'])
    closest_list(['ru_gsd-ud-dev', 'ru_gsd-ud-dev'])
