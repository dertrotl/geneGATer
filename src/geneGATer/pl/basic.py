import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def roc_auc_classification(adata: AnnData, all_metas, raw_cluster, seed=1337):
    """Calculate and plot the ROC-AUC of spot clustering using metagenes.

    Parameters
    ----------
     adata
         The AnnData object to preprocess.
    all_metas
         Result of getComGenes function output out[2].
     raw_cluster
         Result of pre_clustering function output.
     seed
         Set seed for reproducibility.


    Returns
    -------
     Heatmap of ROC-AUC of spot clustering using metagenes.

    """
    result_auc_2 = pd.DataFrame()
    random_state = seed
    adata.obs["raw_cluster"] = raw_cluster

    for cluster in adata.obs["raw_cluster"].cat.categories:
        check = []

        for cluster_inner in adata.obs["raw_cluster"].cat.categories:
            try:
                test = pd.DataFrame(
                    {
                        "metas": all_metas[cluster][cluster_inner],
                        "cluster": adata.obs["raw_cluster"],
                        "classic": (
                            (adata.obs["raw_cluster"] == cluster) | (adata.obs["raw_cluster"] == cluster_inner)
                        ).astype(int),
                    }
                )

                target_cluster = test[test["classic"] == 1]
                nontarget_cluster = test[test["classic"] != 1]
                nontarget_expression = nontarget_cluster["metas"]
                target_expression = target_cluster["metas"]

                X = np.concatenate([nontarget_expression, target_expression], axis=0).reshape(-1, 1)
                y = np.concatenate([np.zeros(len(nontarget_expression)), np.ones(len(target_expression))])
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

                # Step 3: Train a classifier using the training set
                # clf = LogisticRegression(random_state=random_state)
                # clf.fit(X_train, y_train)

                # Step 4: Evaluate the classifier using the test set
                # y_pred = clf.predict(X_test)
                # y_pred_proba = clf.predict_proba(X_test)[:, 1]
                # fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                # roc_auc = roc_auc_score(y_test, y_pred_proba)
                fpr, tpr, _ = roc_curve(y_test, X_test)
                roc_auc = roc_auc_score(y_test, X_test)
                check.append(roc_auc)
            # Plot

            # plt.plot(fpr, tpr)
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('ROC Curve (AUC = {:.3f})'.format(roc_auc))
            # plt.show()
            except KeyError:
                check.append(np.nan)

        result_auc_2[cluster] = check

    result_auc_2.index = result_auc_2.columns

    plt.figure(figsize=(10, 10), dpi=100)
    plt.rcParams["font.size"] = 12
    plt.title("ROC-AUC of target (merged) clusters / spot classification.")
    return sns.heatmap(result_auc_2, cmap="Blues", annot=result_auc_2, fmt=".2f", mask=result_auc_2.isnull())
