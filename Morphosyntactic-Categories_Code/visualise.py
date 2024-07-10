import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import contextlib
import tqdm
import joblib

UDDIR = "ud-treebanks-v2.14"
MODE = ""
VECTOR_DIR = f"{MODE}_Case_RelDep_Matches" if MODE else "Case_RelDep_Matches"
SAVE_DIR = f"{MODE}_Case_Proximities" if MODE else "Case_Proximities"


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


def pca(case1, case2):
    case1_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case1}.csv")
    case1_data_set.insert(1, "Case", case1, True)
    case2_data_set = pd.read_csv(f"{VECTOR_DIR}/RelDep_matching_Case={case2}.csv")
    case2_data_set.insert(1, "Case", case2, True)

    case_data_set = case1_data_set._append(case2_data_set)
    for column in case_data_set.columns:
        case_data_set.replace(
            {
                column: np.nan}, 0., inplace=True
        )

    features = np.array(sorted(case_data_set.columns[5:]))
    features_with_labels = np.append(features, "Case")

    feature_data = case_data_set.loc[:, features].values
    feature_data = StandardScaler().fit_transform(feature_data)
    feat_cols = ['Feature ' + str(i) for i in range(feature_data.shape[1])]
    normalised_data_set = pd.DataFrame(feature_data, columns=feat_cols)

    pca_case = PCA(n_components=2)
    principal_components_case = pca_case.fit_transform(feature_data)
    principal_case_df = pd.DataFrame(data=principal_components_case, columns=['Component 1', 'Component 2'])
    print(principal_case_df.tail())

    print(case_data_set['Case'] == case1)

    print('Explained variation per principal component: {}'.format(pca_case.explained_variance_ratio_))
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1', fontsize=20)
    plt.ylabel('Principal Component - 2', fontsize=20)
    plt.title("Principal Component Analysis of Case Dataset", fontsize=20)
    targets = [case1, case2]
    colors = ['r', 'g']
    for target, color in zip(targets, colors):
        keep_indices = case_data_set['Case'] == target
        print(target)
        print(keep_indices)
        plt.scatter(
            principal_case_df.loc[keep_indices, 'Component 1'],
            principal_case_df.loc[keep_indices, 'Component 2'], c=color, s=50
            )

    plt.legend(targets, prop={
        'size': 15}
               )


pca("Acc", "Nom")
