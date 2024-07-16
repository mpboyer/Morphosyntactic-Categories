import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import os
import contextlib
import joblib
import argparse
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import itertools
# import plotly.express as px

parser = argparse.ArgumentParser()
parser.add_argument("--f1")
parser.add_argument("--f2")
parser.add_argument("-i", "--interactive")
parser.add_argument("-mode", "--mode", default="")
files = parser.parse_args()

if not files.interactive:
    matplotlib.use("pgf")
    preamble = r"\usepackage{xcolor}\definecolor{vulm}{HTML}{7d1dd3}\definecolor{yulm}{HTML}{ffe500}"
    matplotlib.rc("pgf", texsystem="pdflatex", preamble=preamble)


UDDIR = "ud-treebanks-v2.14"
MODE = files.mode
VECTOR_DIR = f"../{MODE}_Case_RelDep_Matches" if MODE else "../Case_RelDep_Matches"
SAVE_DIR = "Figures/Visualisations"


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


def is_prefix(s1: str, s2: str) -> bool:
    """
    Looks at len(s1) first chars in b to check if s1 is a prefix of s2.
    :rtype: bool
    :param s1: pattern
    :param s2: text
    :return: True iff s1 is a prefix of s2.
    """
    if len(s1) > len(s2):
        return False
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            return False
    return True


def fronce(s):
    match s:
        case("Nouns"): return "Noms"
        case("Pronouns"): return "Pronoms"
        case(_): raise ValueError("Calice!")


def get_data_set(case1, case2):
    case1_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case1}.csv")
    case1_data_set.insert(1, "Case", case1, True)
    # drop_indices = [i for i in range(len(case1_data_set['Treebank'])) if is_prefix('sa_vedic', case1_data_set.loc[i, "Treebank"])]
    # case1_data_set.drop(drop_indices, inplace=True)
    case1_data_set["Treebank"].map(lambda n: n + f"_{case1}")

    case2_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case2}.csv")
    case2_data_set.insert(1, "Case", case2, True)
    case2_data_set["Treebank"].map(lambda n: n + f"_{case2}")

    case_data_set = case1_data_set._append(case2_data_set, ignore_index=True)
    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )
    return case_data_set


def pca(case1, case2):
    case_data_set = get_data_set(case1, case2)
    features = np.array(sorted(case_data_set.columns[5:]))

    feature_data = case_data_set.loc[:, features].values
    feature_data = StandardScaler().fit_transform(feature_data)

    pca_case = PCA(n_components=2)
    principal_components_case = pca_case.fit_transform(feature_data)
    principal_case_df = pd.DataFrame(data=principal_components_case, columns=['Component 1', 'Component 2'])

    print(pca_case.explained_variance_, pca_case.explained_variance_ratio_)

    fig, ax = plt.subplots()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Composante 1', fontsize=20)
    plt.ylabel('Composante 2', fontsize=20)
    if files.interactive:
        title = f"Analyse en deux Composantes Principales sur les {fronce(MODE)} pour {case1}, {case2}" if MODE else f"Analyse en deux Composantes Principales sur {case1}, {case2}"
    else:
        title = f"Analyse en deux Composantes Principales\n sur les {fronce(MODE)} " + r"\textcolor{vulm}{" + f"{case1}" + r"}, \textcolor{yulm!80!black}{" + f"{case2}" + r"}" if MODE else "Analyse en deux Composantes Principales sur\n " + r" \textcolor{vulm}{" + f"{case1}" + r"}, \textcolor{yulm!80!black}{" + f"{case2}" + r"}"
    plt.title(title, fontsize=20)
    targets = [case1, case2]
    keep_indices = case_data_set['Case'] == case1
    sc1 = plt.scatter(
        principal_case_df.loc[keep_indices, 'Component 1'],
        principal_case_df.loc[keep_indices, 'Component 2'], c='#7d1dd3', s=50)

    keep_indices = case_data_set['Case'] == case2
    sc2 = plt.scatter(
        principal_case_df.loc[keep_indices, 'Component 1'],
        principal_case_df.loc[keep_indices, 'Component 2'], c='#ffe500', s=50)

    names = case_data_set["Treebank"]
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot1(ind):
        pos = sc1.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def update_annot2(ind):
        pos = sc2.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc1.contains(event)
            if cont:
                update_annot1(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            cont, ind = sc2.contains(event)
            if cont:
                update_annot2(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    plt.legend(targets, prop={'size': 15})
    if files.interactive:
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
    else:
        save_path = f"{SAVE_DIR}/pca_{case1}_{case2}_{MODE}.pdf" if MODE else f"{SAVE_DIR}/pca_{case1}_{case2}.pdf"
        plt.savefig(save_path)


# pca(files.f1, files.f2)


def tsne(case1, case2):
    case1_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case1}.csv")
    case1_data_set.insert(1, "Case", case1, True)
    # drop_indices = [i for i in range(len(case1_data_set['Treebank'])) if is_prefix('sa_vedic', case1_data_set.loc[i, "Treebank"])]
    # case1_data_set.drop(drop_indices, inplace=True)
    case1_data_set["Treebank"].map(lambda n: n + f"_{case1}")

    case2_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case2}.csv")
    case2_data_set.insert(1, "Case", case2, True)
    case2_data_set["Treebank"].map(lambda n: n + f"_{case2}")

    case_data_set = case1_data_set._append(case2_data_set, ignore_index=True)
    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )

    features = np.array(sorted(case_data_set.columns[5:]))

    feature_data = case_data_set.loc[:, features].values
    feature_data = StandardScaler().fit_transform(feature_data)

    tsne_case = TSNE(n_components=2, random_state=42)
    principal_components_case = tsne_case.fit_transform(feature_data)
    principal_case_df = pd.DataFrame(data=principal_components_case, columns=['Component 1', 'Component 2'])

    # perplexity = np.arange(5, 700, 5)
    # divergence = []

    # for i in perplexity:
    #     model = TSNE(n_components=2, init="pca", perplexity=i)
    #     reduced = model.fit_transform(feature_data)
    #     divergence.append(model.kl_divergence_)
    # fig = px.line(x=perplexity, y=divergence, markers=True)
    # fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    # fig.update_traces(line_color="red", line_width=1)
    # fig.show()

    fig, ax = plt.subplots()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Component 1', fontsize=20)
    plt.ylabel('Component 2', fontsize=20)

    if files.interactive:
        title = f"Analyse t-SNE àn deux Composantes sur les {fronce(MODE)} pour {case1}, {case2}" if MODE else f"Analyse t-SNE à deux Composantes sur {case1}, {case2}"
    else:
        title = "Analyse t-SNE à deux Composantes sur\n" + f"les {fronce(MODE)} " + r"\textcolor{vulm}{" + f"{case1}" + r"}, \textcolor{yulm!80!black}{" + f"{case2}" + r"}" if MODE else "Analyse t-SNE à deux Composantes\n sur " + r"\textcolor{vulm}{" + f"{case1}" + r"}, \textcolor{yulm!80!black}{" + f"{case2}" + r"}"
    plt.title(title, fontsize=20)
    targets = [case1, case2]
    keep_indices = case_data_set['Case'] == case1
    sc1 = plt.scatter(
        principal_case_df.loc[keep_indices, 'Component 1'],
        principal_case_df.loc[keep_indices, 'Component 2'], c='#7d1dd3', s=50)

    keep_indices = case_data_set['Case'] == case2
    sc2 = plt.scatter(
        principal_case_df.loc[keep_indices, 'Component 1'],
        principal_case_df.loc[keep_indices, 'Component 2'], c='#ffe500', s=50)

    names = case_data_set["Treebank"]
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot1(ind):
        pos = sc1.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def update_annot2(ind):
        pos = sc2.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = "{}".format(" ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.4)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc1.contains(event)
            if cont:
                update_annot1(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
            cont, ind = sc2.contains(event)
            if cont:
                update_annot2(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    plt.legend(targets, prop={'size': 15})

    if files.interactive:
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
    else:
        save_path = f"{SAVE_DIR}/tsne_{case1}_{case2}_{MODE}.pdf" if MODE else f"{SAVE_DIR}/tsne_{case1}_{case2}.pdf"
        plt.savefig(save_path)


def tsne_lang_list(allowed_languages):
    cases = get_cases(allowed_languages)
    case1 = cases[0]
    case_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case1}.csv")
    case_data_set.insert(1, "Case", case1, True)
    case_data_set["Treebank"].map(lambda n: n + f"_{case1}")

    for case in cases[1:]:
        case_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case}.csv")
        case_set['Treebank'].map(lambda n: n + f"_{case}")
        case_set.insert(1, "Case", case, True)
        case_data_set = case_data_set._append(case_set, ignore_index=True)

    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )
        
    # print(case_data_set.head)
    drop_indices = [i for i, c in enumerate(case_data_set["Treebank"]) if c.split("_")[0] not in allowed_languages]
    # print(drop_indices)
    case_data_set.drop(index=drop_indices, axis=0, inplace=True)
    case_data_set.reset_index(inplace=True)
    # print(case_data_set.head)

    languages = set([c.split("_")[0] for c in case_data_set["Treebank"]])
    
    
    features = np.array(sorted(case_data_set.columns[5:]))
    feature_data = case_data_set.loc[:, features].values
    feature_data = StandardScaler().fit_transform(feature_data)

    shapes = ['2', '+', 'x', '|', '_', '*', '.', '^']
    # colours = ['#7d1dd3', '#ffe500', '#a45a2a', '#da291c', "#007a33", "#e89cae", "#10069f", "#00a3e0", "#6eceb2"][:len(cases)]
    cases_indices = dict([(c, i) for i, c in enumerate(cases)])

    tsne_case = TSNE(n_components=2, random_state=42)
    principal_components_case = tsne_case.fit_transform(feature_data)
    principal_case_df = pd.DataFrame(data=principal_components_case, columns=['Component 1', 'Component 2'])

    # perplexity = np.arange(5, 700, 5)
    # divergence = []

    # for i in perplexity:
    #     model = TSNE(n_components=2, init="pca", perplexity=i)
    #     reduced = model.fit_transform(feature_data)
    #     divergence.append(model.kl_divergence_)
    # fig = px.line(x=perplexity, y=divergence, markers=True)
    # fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    # fig.update_traces(line_color="red", line_width=1)
    # fig.show()

    fig, ax = plt.subplots()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Composante 1', fontsize=10)
    plt.ylabel('Composante 2', fontsize=10)

    title = f"Analyse t-SNE à deux Composantes sur les {fronce(MODE)} pour\nles cas de " + " ".join(allowed_languages) if MODE else "Analyse t-SNE à deux Composantes pour\nles cas de " + " ".join(allowed_languages)
    plt.title(title, fontsize=20)
    targets = cases
    
    for target, shape in zip(allowed_languages, shapes):
        keep_indices = [i for i, c in enumerate(case_data_set["Treebank"]) if c.split("_")[0] == target]
        # print(target, keep_indices)
        colour_indices = [cases_indices[c] for c in case_data_set.loc[keep_indices, 'Case']]
        # color_indices = [0]

        sc = plt.scatter(
            principal_case_df.loc[keep_indices, 'Component 1'],
            principal_case_df.loc[keep_indices, 'Component 2'], c=colour_indices, s=50, marker=shape, cmap='jet')

    
    save_path = f"{SAVE_DIR}/tsne_cases_{MODE}_{allowed_languages}.pdf" if MODE else f"{SAVE_DIR}/tsne_cases_{allowed_languages}.pdf"
    plt.savefig(save_path)


def get_cases(languages):
    cases = list(filter(lambda t: t[0] == "R", os.listdir(VECTOR_DIR)))
    res = []
    for c in cases:
        with open(f"{VECTOR_DIR}/{c}", 'r') as csv_file: 
            for tree in csv_file:
                if tree.split(',')[0].split('_')[0] in languages:
                    res.append(c[-7:-4])
                    break
    return res


studied_languages = ['tr', 'sk', 'ab', 'eu', 'fi', 'hit', 'ta', 'wbp']
# print(case_list)
tsne_list(studied_languages)    


# TODO: Comparer langues très différents (gnn + t-SNE)

