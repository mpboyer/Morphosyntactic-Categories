import math
import joblib
import itertools
import openpyxl
import openpyxl.styles
import pandas
import scipy.linalg
from tqdm import tqdm, trange
import numpy as np
import numpy.linalg as npl
from sympy import Matrix
import os
import contextlib
import argparse

from linalg import project, angle, distance, dict_distance, is_in_cone, manhattan_normalizer

UDDIR = "ud-treebanks-v2.14"
VECTOR_DIR = "RelDep_Matches"


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


def overall_basis_csv():
    basis = set()
    for csv in filter(lambda t: t[-4:] == ".csv", os.listdir(f"{VECTOR_DIR}")):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
        basis |= set(attributes)
    return basis


def get_vector_csv(treebank, case):
    with open(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv", "r") as csv_file:
        attributes = csv_file[0].split(",")[4:]
        for tree in filter(lambda t: t[0] == treebank, csv_file):
            coordinates = next(csv_file).rstrip().split(",")[4:]
    return dict(zip(attributes, coordinates))


def get_matrix_csv(treebank):
    basis = set()
    case_space = []
    for csv in sorted(filter(lambda t: t[-4:] == ".csv" and t[16:20] == "Case", os.listdir(f"{VECTOR_DIR}"))):
        with open(f"{VECTOR_DIR}/{csv}", "r") as csv_file:
            attributes = next(csv_file).rstrip().split(",")[4:]
            for tree in csv_file:
                parsed_tree = tree.rstrip().split(",")
                if parsed_tree[0] == treebank:
                    coordinates = dict(zip(attributes, map(void_to_zero, parsed_tree[4:])))
                    coordinates["Total"] = void_to_zero(parsed_tree[3])
        case_space.append(coordinates)
        basis |= set(attributes)

    matrix = np.array([[0. for _ in case_space] for _ in basis])
    for row, b in enumerate(basis):
        for column, vector in enumerate(case_space):
            total = vector["Total"]
            if total != 0.:
                matrix[row, column] = vector.get(b, 0.) / total

    return matrix


def enhanced_get_matrix_csv(treebank, treebank_case):
    basis = set()
    case_space_vectors = []
    case_dict = {}
    for csv in sorted(filter(lambda t: t[-4:] == ".csv" and t[16:20] == "Case", os.listdir(f"{VECTOR_DIR}"))):
        case_coordinates = {
            "Total": 0.}
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
        basis |= set(attributes)

    case_space_matrix = np.array([[0. for _ in case_space_vectors] for _ in basis])
    case_vector = np.array([0. for _ in basis])

    for row, b in enumerate(basis):
        case_vector[row] = case_dict.get(b, 0.) / case_dict["Total"] if case_dict["Total"] != 0. else 0.
        for column, vector in enumerate(case_space_vectors):
            case_space_matrix[row, column] = vector.get(b, 0.) / vector["Total"] if vector["Total"] != 0. else 0.

    return case_space_matrix, case_vector


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
                    if parsed_vector[0] == corpus:
                        case_dict = dict(zip(vec_attr, map(void_to_zero, parsed_vector[4:])))
                        case_dict["Total"] = void_to_zero(parsed_vector[3])

            vector = [0. for _ in basis]
            for coordinate, base_vector in enumerate(basis):
                vector[coordinate] = case_dict.get(base_vector, 0.) / case_vec["Total"] if case_vec[
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
                    total=len(all_banks) * len(gf)
                )
        ) as progress_bar:
            angles.append(
                dict(
                    joblib.Parallel(n_jobs=8, verbose=0)(
                        joblib.delayed(get_angle)(dimension, [all_banks[j], c]) for j in range(len(all_banks))
                        for c in gf
                    )
                )
            )
        pandas.DataFrame(angles).to_csv(f"Proximities/Vector_Angle_Proximity.csv", index=False)


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
            for index, koln in enumerate(basis):
                mat[index, row] = coordinate_dict.get(koln, 0.)
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
                for index, koln in enumerate(basis):
                    mat[index, row] = coordinate_dict.get(koln, 0.) / total

    return treebanks, mat


def tabulize_angles_csv(grammar_feature):
    try:
        with open(f"Proximities/Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}.csv"):
            pass
    except FileNotFoundError:
        print(f"Tabulizing for {grammar_feature[0]}={grammar_feature[1]}")
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
            f"Proximities/Proximity_Stats_for_{grammar_feature[0]}={grammar_feature[1]}.csv"
        )


def tabulize_angle_pair_csv(gf1, gf2):
    try:
        with open(f"Proximities/DuoProximity_for_{gf1[1]}_and_{gf2[1]}.csv"):
            pass
    except FileNotFoundError:
        treebanks1, mat1 = euclidean_reldep_matrix_csv(gf1)
        treebanks2, mat2 = euclidean_reldep_matrix_csv(gf2)
        dot_mat = mat1.T.dot(mat2)
        pandas.DataFrame(data=dot_mat, columns=treebanks1).to_csv(
            f"Proximities/DuoProximity_for_{gf1[1]}_and_{gf2[1]}.csv"
        )


def compute_distances_csv():
    for c1, c2 in tqdm(
            itertools.product(gf, gf), colour="#7d1dd3", desc="Computing Vector-Vector Distances", total=len(gf) ** 2
    ):
        try:
            with open(f"Proximities/Distances_{c1}_{c2}.csv"):
                pass
        except FileNotFoundError:
            treebanks1, mat1 = manhattan_reldep_matrix_csv(("Case", c1))
            treebanks2, mat2 = manhattan_reldep_matrix_csv(("Case", c2))
            shape = np.shape(mat1)
            distance_mat = np.array([[0. for _ in range(shape[1])] for _ in range(shape[1])])
            for column, v1 in enumerate(mat1):
                for row, v2 in enumerate(mat2):
                    distance_mat[row, column] = distance(v1, v2)
            pandas.DataFrame(data=distance_mat, columns=treebanks1).to_csv(
                f"Proximities/Distances_{c1}_{c2}.csv"
            )


def closest(treebank1, treebank2):
    matrix1, matrix2 = get_matrix_csv(treebank1), get_matrix_csv(treebank2)
    m1_dicts = {}
    m2_dicts = {}
    all_cases = sorted(filter(lambda t: t[-4:] == ".csv" and t[16:20] == "Case", os.listdir(f"{VECTOR_DIR}")))
    for col, csv in filter(lambda n: np.any(matrix1[:, n[0]]), enumerate(all_cases)):
        m1_dicts[csv[-7:-4]] = dict(
            [(c[-7:-4], distance(matrix1[:, col], matrix2[:, i])) for i, c in
             filter(lambda n: np.any(matrix2[:, n[0]]), enumerate(all_cases))]
            )
    for col, csv in filter(lambda n: np.any(matrix2[:, n[0]]), enumerate(all_cases)):
        m2_dicts[csv[-7:-4]] = dict(
            [(c[-7:-4], distance(matrix2[:, col], matrix1[:, i])) for i, c in
             filter(lambda n: np.any(matrix1[:, n[0]]), enumerate(all_cases))]
            )
    for m in m1_dicts:
        mk = min(m1_dicts[m], key=m1_dicts[m].get)
        m1_dicts[m] = mk, m1_dicts[m][mk]
    for m in m2_dicts:
        mk = min(m2_dicts[m], key=m2_dicts[m].get)
        m2_dicts[m] = mk, m2_dicts[m][mk]
    return treebank1, m1_dicts, treebank2, m2_dicts










