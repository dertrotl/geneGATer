import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch_geometric.utils as pyg_utils
from anndata import AnnData
from matplotlib.cm import get_cmap
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from umap import UMAP

from geneGATer.tl.models import GAT_linear_negbin, GAT_negbin

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


def attention_matrix(model, data, font_size=14, cmap="Reds"):
    """Create Attention Matrix plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    font_size
        Font size for the plot.
    cmap
        Color map for the plot.

    Returns
    -------
    Attention Matrix plot.

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    adj : scipy.sparse.csr_matrix
    The attention matrix.
    """
    # print("Create Attention Matrix plot...")
    # Model Eval
    model.eval()

    with torch.no_grad():
        (_, att) = model(data.x, data.edge_index)[1]

    # if heads != 1:
    #    att_mean = att.mean(dim=-1)
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att_mean)
    # else:
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)

    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)
    # adj = pyg_utils.to_scipy_sparse_matrix(index.data, att.data)
    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots()
    # plt.figure(figsize=(14, 14))
    ax = sns.heatmap(adj.todense(), cmap=cmap)
    ax.set_aspect("equal")
    # plt.savefig('/Users/benjaminweinert/Desktop/test_heatmap.png', dpi=600)
    # plt.show()
    # fig.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Attention Matrix": wandb.Image("plot.png")})
    # wandb.log({"Attention Matrix": fig})
    # print("done.\n")
    return fig, adj


def attention_pca(model, data, font_size=14, n_components=10):
    """Create Attention PCA plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    font_size
        Font size for the plot.
    n_components
        Number of components for the PCA.

    Returns
    -------
    Attention PCA plot.

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    pca : sklearn.decomposition.pca.PCA
    The PCA object.
    """
    model.eval()

    with torch.no_grad():
        (index, att) = model(data.x, data.edge_index)[1]

    # if heads != 1:
    #    att_mean = att.mean(dim=-1)
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att_mean)
    # else:
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)
    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)
    # print("Create Attention PCA...")
    scaled_adj = preprocessing.scale(np.asarray(adj.todense()))
    adj_eval = pd.DataFrame(scaled_adj)
    pca = PCA(n_components=n_components)  # create a PCA object
    pca.fit(adj_eval)  # do the math
    # pca_data = pca.transform(adj_eval)  # get PCA coordinates for scaled_data

    # print("done.\n")

    # print("Create PCA Elbow plot...")
    # The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    # labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots()
    ax.plot(range(1, len(per_var) + 1), per_var)
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_xlabel("Principal Component")
    # plt.show()
    # wandb.log({"Attention PCA": wandb.Image(fig)})
    # print("done.\n")
    return fig, pca


def attention_umap(
    model,
    data,
    adata: AnnData,
    n_components=10,
    random_state=1337,
    font_size=14,
    cmap="tab20",
    cluster_key="cluster",
    umap_model="umap",
    set_size=(10, 10),
):
    """Create Attention UMAP plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    adata
        The AnnData object.
    n_components
        Number of components for the PCA.
    random_state
        Random state for reproducibility.
    font_size
        Font size for the plot.
    cmap
        Color map for the plot.
    cluster_key
        Key in which clustering is getting saved in adata.obs.
    umap_model
        "umap" or "tsne".
    set_size
        Size of the plot.

    Returns
    -------
    Attention UMAP plot.

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    umap : numpy.ndarray
        The UMAP coordinates.
    """
    # print("Create Attention UMAP plot...")
    model.eval()

    with torch.no_grad():
        (index, att) = model(data.x, data.edge_index)[1]

    # if heads != 1:
    #    att_mean = att.mean(dim=-1)
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att_mean)
    # else:
    #    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)
    adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)
    scaled_adj = preprocessing.scale(np.asarray(adj.todense()))
    adj_eval = pd.DataFrame(scaled_adj)
    pca = PCA(n_components=n_components)  # create a PCA object
    pca.fit(adj_eval)  # do the math
    pca_data = pca.transform(adj_eval)  # get PCA coordinates for scaled_data

    # Adj UMAP
    if umap_model == "umap":
        umap = UMAP(n_components=2, init="random", random_state=random_state).fit_transform(pca_data)
    elif umap_model == "tsne":
        umap = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(pca_data)

    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots()
    name = cmap
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(set_size[0], set_size[1])
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(umap, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # fig.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Attention UMAP": wandb.Image("plot.png")})

    # wandb.log({"Attention UMAP": wandb.Image(fig)})
    # wandb.log({"Attention UMAP": fig})
    # print("done.\n")
    return fig, umap


def model_pca(model, data, n_components=10, font_size=14):
    """Create Model PCA plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    n_components
        Number of components for the PCA.
    font_size
        Font size for the plot.

    Returns
    -------
    Model PCA plot.

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    pca : sklearn.decomposition.pca.PCA
    The PCA object.
    """
    model.eval()

    pred = model(data.x, data.edge_index)[0]

    if isinstance(model, GAT_negbin) or isinstance(model, GAT_linear_negbin):
        pred, var = pred
    pred = pred.data

    # print("Model PCA...")
    scaled_data_2 = preprocessing.scale(pred.cpu())
    pca_2 = PCA()
    pca_2 = PCA(n_components=n_components)  # create a PCA object
    pca_2.fit(scaled_data_2)  # do the math
    # pca_data_2 = pca_2.transform(scaled_data_2)  # get PCA coordinates for scaled_data
    # print("done.\n")

    # print("Create Model PCA Elbow plot...")
    # The following code constructs the Scree plot
    per_var_2 = np.round(pca_2.explained_variance_ratio_ * 100, decimals=1)
    # labels = ["PC" + str(x) for x in range(1, len(per_var_2) + 1)]

    # plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots()
    ax.plot(range(1, len(per_var_2) + 1), per_var_2)
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_xlabel("Principal Component")
    # plt.show()
    # wandb.log({"Model PCA": wandb.Image(fig)})
    # wandb.log({"Model PCA": fig})
    # print("done.\n")
    return fig, pca_2


def model_umap(
    model,
    data,
    adata: AnnData,
    n_components=10,
    random_state=1337,
    font_size=14,
    cmap="tab20",
    cluster_key="cluster",
    umap_model="umap",
    set_size=(10, 10),
):
    """Create Model UMAP plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    adata
        The AnnData object.
    n_components
        Number of components for the PCA.
    random_state
        Random state for reproducibility.
    font_size
        Font size for the plot.
    cmap
        Color map for the plot.
    cluster_key
        Key in which clustering is getting saved in adata.obs.
    umap_model
        "umap" or "tsne".
    set_size
        Size of the plot.

    Returns
    -------
    Model UMAP plot.

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    umap : numpy.ndarray
        The UMAP coordinates.
    """
    # print("Create Model UMAP plot...")
    model.eval()

    pred = model(data.x, data.edge_index)[0]

    if isinstance(model, GAT_negbin) or isinstance(model, GAT_linear_negbin):
        pred, var = pred
    pred = pred.data

    scaled_data_2 = preprocessing.scale(pred.cpu())
    pca_2 = PCA()
    pca_2 = PCA(n_components=n_components)  # create a PCA object
    pca_2.fit(scaled_data_2)  # do the math
    pca_data_2 = pca_2.transform(scaled_data_2)

    # Trained UMAP
    if umap_model == "umap":
        umap = UMAP(n_components=2, init="random", random_state=random_state).fit_transform(
            pca_data_2[:, :n_components]
        )
    elif umap_model == "tsne":
        umap = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(pca_data_2[:, :n_components])

    plt.rcParams["font.size"] = font_size
    fig, ax = plt.subplots()
    name = "tab20"
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.axis('off')
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(set_size[0], set_size[1])
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(umap, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    # fig.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Model UMAP": wandb.Image("plot.png")})

    # wandb.log({"Model UMAP": wandb.Image(fig)})
    # wandb.log({"Model UMAP": fig})
    # print("done.\n")
    return fig, umap


def get_top_k_genes(model, gene_list, k, marker_genes=None, font_size=14, set_size=(10, 6), cmap="tab:blue"):
    """Create Top k sender and receiver genes plot.

    Parameters
    ----------
    model
        The model from learn_model function.
    gene_list
        List of genes from getComGenes function.
    k
        Number of top k genes to plot.
    marker_genes
        List of marker genes.
    font_size
        Font size for the plot.
    set_size
        Size of the plot.
    cmap
        Color map for the plot.

    Returns
    -------
    Top k sender and receiver genes plot.

    fig1 : matplotlib.pyplot.figure
        The matplotlib receiving signal figure object.
    fig2 : matplotlib.pyplot.figure
        The matplotlib sending signal figure object.
    df : pandas.DataFrame
        The top k genes.
    """
    if marker_genes is None:
        marker_genes = []
    model.eval()
    # print("Top Ten most important genes receiver and sender nodes:")

    w = model.conv2.state_dict()["lin_l.weight"].abs().detach().cpu().numpy()

    # Get sum of each column (each gene across all rows)
    sums = w.sum(axis=0)

    # Get indices that would sort the sums in descending order
    sorted_indices = sums.argsort()[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Get the corresponding top-k gene names
    df = pd.DataFrame({"test": gene_list})
    gene_plot_name_w = df.iloc[top_k_indices]["test"].tolist()

    # Get the corresponding top-k columns from w
    matrix_subset = w[:, top_k_indices]

    # Create a new dataframe with the subset of the matrix and gene names as column names
    df_plot = pd.DataFrame(matrix_subset, columns=gene_plot_name_w)

    # fig, ax = plt.subplots()
    # Create violin plot
    plt.rcParams["font.size"] = font_size
    plt.figure(figsize=set_size)
    # ax = sns.violinplot(data=df_plot.iloc[:,:10], inner="box", stripplot=False)
    ax = sns.boxplot(data=df_plot.iloc[:, :10], color=cmap, showfliers=False)
    plt.title("Weight Distribution Top 10 Receiving Genes")
    plt.xlabel("Genes")
    plt.ylabel("W_r Values")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace labels in list with * if they are in marker_genes list
    labels = [label + "*" if label in marker_genes else label for label in labels]

    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # plt.show()
    fig = plt.gcf()
    # plt.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Violin W_r": wandb.Image("plot.png")})

    # Get absolute weights
    w2 = model.conv2.state_dict()["lin_r.weight"].abs().detach().cpu().numpy()

    # Get sum of each column (each gene across all rows)
    sums = w2.sum(axis=0)

    # Get indices that would sort the sums in descending order
    sorted_indices = sums.argsort()[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Get the corresponding top-k gene names
    df = pd.DataFrame({"test": gene_list})
    gene_plot_name_w2 = df.iloc[top_k_indices]["test"].tolist()

    # Get the corresponding top-k columns from w
    matrix_subset = w2[:, top_k_indices]

    # Create a new dataframe with the subset of the matrix and gene names as column names
    df_plot = pd.DataFrame(matrix_subset, columns=gene_plot_name_w2)

    # fig2, ax2 = plt.subplots()
    # Create violin plot
    plt.rcParams["font.size"] = font_size
    plt.figure(figsize=set_size)
    # ax = sns.violinplot(data=df_plot.iloc[:,:10], inner="box", stripplot=False)
    ax2 = sns.boxplot(data=df_plot.iloc[:, :10], color=cmap, showfliers=False)
    plt.title("Weight Distribution Top 10 Sender Genes")
    plt.xlabel("Genes")
    plt.ylabel("W_s Values")

    labels = [item.get_text() for item in ax2.get_xticklabels()]

    # Replace labels in list with * if they are in marker_genes list
    labels = [label + "*" if label in marker_genes else label for label in labels]

    # Set the new labels
    ax2.set_xticklabels(labels)
    ax2.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    # plt.show()
    fig2 = plt.gcf()
    # plt.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Violin W_s": wandb.Image("plot.png")})

    df = pd.DataFrame({"Receiver Genes": gene_plot_name_w, "Sender Genes": gene_plot_name_w2})
    # wandb.log({"Top ten weighted genes": wandb.Table(dataframe=df)})
    # df = pd.DataFrame({"all_genes": list(set(gene_plot_name_w).union(gene_plot_name_w2))})
    # wandb.log({"Top ten weighted genes list": wandb.Table(dataframe=df)})
    # print("done.\n")

    return fig, fig2, df


def get_top_k_genes_saliency(
    model, data, gene_list, k, marker_genes=None, font_size=14, set_size=(10, 6), cmap="tab:blue"
):
    """Create Top k genes plot (sorted by saliency scores).

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.
    gene_list
        List of genes from getComGenes function.
    k
        Number of top k genes to plot.
    marker_genes
        List of marker genes to compare to.
    font_size
        Font size for the plot.
    set_size
        Size of the plot.
    cmap
        Color map for the plot.

    Returns
    -------
    Top k genes plot (sorted by saliency scores).

    fig : matplotlib.pyplot.figure
        The matplotlib figure object.
    df : pandas.DataFrame
        The top k genes.
    """
    # print("Top k most important input genes (Saliency):")

    if marker_genes is None:
        marker_genes = []
    model.eval()

    # Get saliency
    sal = compute_saliency(model, data)

    # Get sum of each column (each gene across all rows)
    sums = sal.sum(axis=0)

    # Get indices that would sort the sums in descending order
    sorted_indices = sums.argsort()[::-1]

    # Get the top-k indices
    top_k_indices = sorted_indices[:k]

    # Get the corresponding top-k gene names
    df = pd.DataFrame({"test": gene_list})
    gene_plot_name_w = df.iloc[top_k_indices]["test"].tolist()

    # Get the corresponding top-k columns from w
    matrix_subset = sal[:, top_k_indices]

    # Create a new dataframe with' the subset of the matrix and gene names as column names
    df_plot = pd.DataFrame(matrix_subset, columns=gene_plot_name_w)

    # fig, ax = plt.subplots()
    plt.rcParams["font.size"] = font_size
    plt.figure(figsize=set_size)
    ax = sns.boxplot(data=df_plot.iloc[:, :10], color=cmap, showfliers=False)
    # ax = sns.boxplot(data=df_plot.iloc[:,:])
    plt.title("Saliency Distributions Top 10 Input Genes")
    plt.xlabel("Genes")
    plt.ylabel("Saliency Scores")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace labels in list with * if they are in marker_genes
    labels = [label + "*" if label in marker_genes else label for label in labels]

    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    fig = plt.gcf()
    # plt.savefig("plot.png", format="png", bbox_inches="tight")
    # wandb.log({"Saliency Boxplots": wandb.Image("plot.png")})

    df = pd.DataFrame({"all_genes": gene_plot_name_w})
    # wandb.log({"Top ten input genes list (Saliency Scores)": wandb.Table(dataframe=df)})
    # print("done.\n")
    return fig, df


def compute_saliency(model, data):
    """Compute the saliency map of the model.

    Parameters
    ----------
    model
        The model from learn_model function.
    data
        The data from learn_model function.

    Returns
    -------
    saliency : numpy.ndarray
    The saliency map.
    """
    model.eval()
    model.zero_grad()

    data.x = data.x.clone().detach().requires_grad_(True)

    # Forward pass through the model
    output = model(data.x, data.edge_index)[0]
    # if model_type in ["GAT_negbin", "GAT_linear_negbin"]:
    if isinstance(model, GAT_negbin) or isinstance(model, GAT_linear_negbin):
        output, _ = output

    output = torch.sum(output)

    output.backward()

    saliency = data.x.grad.detach().abs()

    return saliency.cpu().numpy()
