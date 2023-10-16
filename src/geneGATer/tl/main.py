import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import squidpy as sq
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import wandb
from matplotlib.cm import get_cmap
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from torch_geometric.data import Data
from tqdm.auto import tqdm
from umap import UMAP

from geneGATer.pp import NegLogNegBinLoss, _r_squared_linreg
from geneGATer.tl.models import GAT, GAT_linear, GAT_linear_negbin, GAT_negbin

warnings.filterwarnings("ignore")
wandb.login()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def learn_model(
    adata,
    gene_list,
    model_type,
    loss,
    epochs=1000,
    lr=0.05,
    weight_decay=5e-4,
    n_rings=1,
    tissue=None,
    seed=1337,
    heads=1,
    norm=False,
    top=10,
    library_key=None,
    gene_ids="gene_ids",
    cluster_key="cluster",
    data_name="Adata Dataset",
    project="my_model",
    compare_gene_list=None,
):
    """Learn a model with the given parameters to rank input genes list by importance.

    Parameters
    ----------
    adata
        The AnnData object.
    gene_list
        List of genes to rank, e.g. from getComGenes.
    model_type
        Type of model to use. (GAT, GAT_linear, GAT_linear_negbin, GAT_negbin)
    loss
        Loss function to use. (negbin, mse, poisson)
    epochs
        Number of epochs to train the model.
    lr
        Learning rate for the optimizer.
    weight_decay
        Weight decay for the optimizer.
    n_rings
        Number of rings to use for the spatial graph.
    tissue
        Tissue to use for the spatial graph. (None if from one donor, if multiple donors, 1 is first donor, 2 second donor, etc., 0 is all donors)))
    seed
        Seed for the random number generator.
    heads
        Number of heads to use for the GAT model. (currently not supported)
    norm
        Normalize the data, yes or no.
    top
        Number of top k genes to plot extracted from the model.
    library_key
        Key where donor names are saved in adata.obs.
    gene_ids
        Key where gene names are saved in adata.var.
    cluster_key
        Key where clustering should be saved in adata.obs.
    data_name
        Name of the dataset.
    project
        Name of the project, when uploaded to wandb.
    compare_gene_list
        List of genes to compare to, e.g. top k ranked genes are marked with an asterix if they are from this list.

    Returns
    -------
    model
        The trained model.
    data
        The data splits used for training. (soon)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = top

    if compare_gene_list is None:
        gbm_genes = [
            "FCGBP",
            "TIMP1",
            "LGALS1",
            "COL1A1",
            "CANX",
            "MIF",
            "SERPINE1",
            "SLC1A3",
            "RPS27A",
            "IGFBP2",
            "SLC17A7",
            "SNAP25",
            "PCSK1N",
            "MGP",
            "EGFR",
            "MT2A",
            "CHI3L1",
            "SYT1",
            "TMSB10",
            "CSF1R",
            "HSP90AA1",
            "S100A11",
            "RPS11",
            "IGFBP5",
            "S100B",
            "UBC",
            "PMP2",
            "FTH1",
            "CD81",
            "CXCL14",
            "UBB",
            "NPTX2",
            "EGR1",
            "PTX3",
            "THY1",
            "BIN1",
            "NES",
            "COL1A2",
            "VASP",
            "HSP90B1",
            "APOD",
            "PDPN",
            "OLIG1",
            "SLC1A2",
            "COL3A1",
            "PPIB",
            "HNRNPA2B1",
            "BSG",
            "CD44",
            "C1QB",
            "EEF2",
            "ATP1A2",
            "STMN1",
            "IGFBP7",
            "COL4A2",
            "HTRA1",
            "CTSB",
            "ENO2",
            "CD163",
            "GBP1",
            "PTGDS",
            "FN1",
            "LDHA",
            "C1QA",
            "ZYX",
            "ANXA1",
            "VIM",
            "BCAN",
            "COL4A1",
            "TNFRSF12A",
            "CPLX2",
            "HLA-A",
            "GADD45A",
            "HOPX",
            "TUBB",
            "IGFBP3",
            "IDH1",
            "PLP1",
            "NDUFA4",
            "RCAN1",
            "TUBB2A",
            "CD9",
            "PTN",
            "NNMT",
            "DIRAS3",
            "GAP43",
            "SPP1",
            "SERPINA3",
            "VSNL1",
            "RPLP1",
            "HLA-DRA",
            "SOD2",
            "HLA-B",
            "HLA-DPA1",
            "RPL18",
            "ADGRB1",
            "RPLP0",
            "CD63",
            "GPR37L1",
            "GSN",
            "FTL",
            "PTPRZ1",
            "CCN1",
            "SPARCL1",
            "CRYAB",
            "NDRG2",
        ]
    else:
        gbm_genes = compare_gene_list

    if loss == "negbin":
        loss_fct = NegLogNegBinLoss()
    elif loss == "mse":
        loss_fct = nn.MSELoss()
    elif loss == "poisson":
        loss_fct = nn.PoissonNLLLoss(log_input=True)

    if norm is True:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

    sc.pp.neighbors(adata)

    if library_key is not None:
        if tissue != 0:
            sample_name = adata.obs[library_key].cat.categories[tissue - 1]
            adata = adata[adata.obs[library_key].isin([sample_name]), :]

    # nn_graph_genes = adata.obsp["connectivities"]
    # sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=6, library_key="sample_name")
    # nn_graph_space = adata.obsp["spatial_connectivities"]
    # alpha = 0.2
    # joint_graph = (1 - alpha) * nn_graph_genes + alpha * nn_graph_space
    # sc.tl.leiden(adata, adjacency=joint_graph, key_added=cluster_key, resolution = 1)

    genes = []

    args = np.argwhere(gene_list)
    for it in range(0, len(args)):
        genes.append(args[it][0])

    x = torch.tensor(adata.X.todense()[:, genes])
    y = torch.tensor(adata.X.todense())

    if (library_key is not None) and (tissue == 0):
        sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=6, library_key=library_key)
    else:
        sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=6)

    edge_index = adata.obsp["spatial_connectivities"].nonzero()
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    perm = torch.randperm(data.num_nodes)
    data.train_mask = perm[: int(0.8 * data.num_nodes)]
    data.val_mask = perm[int(0.8 * data.num_nodes) : int(0.9 * data.num_nodes)]
    data.test_mask = perm[int(0.9 * data.num_nodes) :]

    config = {
        "learning_rate": lr,
        "architecture": model_type,
        "dataset": data_name,
        "epochs": epochs,
        "Loss": loss,
        "weight_decay": weight_decay,
        "nrings": n_rings,
        "sample": tissue,
    }

    # Define Model Pipeline
    def model_pipeline(hyperparameters, data, project):
        # tell wandb to get started
        run = wandb.init(project=project, config=hyperparameters)
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, criterion, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, data, criterion, optimizer, config, run)

        # and test its final performance
        test(model, data, run)

        return model

    def load_model(model_type, data, device):
        if model_type == "GAT_negbin":
            # Load GAT_negbin function
            return GAT_negbin(data.num_features, data.num_features, data.y.shape[1]).to(device)
        elif model_type == "GAT_linear":
            # Load linear_GAT function
            return GAT_linear(data.num_features, data.num_features, data.y.shape[1]).to(device)
        elif model_type == "GAT_linear_negbin":
            # Load linear_GAT_negbin function
            return GAT_linear_negbin(data.num_features, data.num_features, data.y.shape[1]).to(device)
        elif model_type == "GAT":
            # Load GAT function
            return GAT(data.num_features, data.num_features, data.y.shape[1]).to(device)
        else:
            raise ValueError("Invalid function type specified.")

    def make(config):
        # Make the model

        # model = GAT(data.num_features, data.num_features, data.y.shape[1]).to(device)
        model = load_model(config.architecture, data, device)
        lr = config.learning_rate
        weight_decay = config.weight_decay
        # Make the loss and optimizer
        criterion = loss_fct
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        return model, criterion, optimizer

    def train(model, data, criterion, optimizer, config, run):
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        run.watch(model, criterion, log="all", log_freq=1)
        model.train()

        # for epoch in range(1, config.epochs + 1):
        for epoch in tqdm(range(config.epochs)):
            loss = train_batch(data, model, optimizer, criterion)
            run.log({"epoch": epoch + 1, "loss": loss})

            # if (((epoch+1) % 10) == 0) and (epoch != config.epochs + 1):
            if (((epoch + 1) % 10) == 0) and (epoch + 1 != config.epochs):
                test(model, data, run)
                print("Epoch: ", float(epoch + 1), " Loss: ", float(loss))

    if model_type in ["GAT_negbin", "GAT_linear_negbin"]:

        def train_batch(data, model, optimizer, criterion):
            cumu_loss = 0
            feat = data.x.to(device)
            edges = data.edge_index.to(device)
            y = data.y.to(device)

            # Forward pass ➡
            out, _ = model(feat, edges)
            mean, var = out
            loss = criterion(mean[data.train_mask], y[data.train_mask], var[data.train_mask])
            cumu_loss += loss.item()
            # Backward pass ⬅

            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()
            # wandb.log({"Loss": loss.item()})
            return cumu_loss

        def test(model, data, run):
            model.eval()

            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.to(device)

            # Run the model on some test examples
            with torch.no_grad():
                pred, _ = model(data.x, data.edge_index)
                pred, _ = pred
                accs = []
                r_2 = []
                r_2_lin = []
                for mask in [data.train_mask, data.val_mask, data.test_mask]:
                    accs.append(F.mse_loss(pred[mask], data.y[mask]))
                    r_2.append(float(r2_score(data.y[mask].cpu(), pred[mask].cpu())))
                    r_2_lin.append(float(_r_squared_linreg(data.y[mask].cpu(), pred[mask].cpu())))

                train_acc, val_acc, test_acc = accs
                train_r_2, val_r_2, test_r_2 = r_2
                train_r_2_lin, val_r_2_lin, test_r_2_lin = r_2_lin
                run.log(
                    {
                        "Train Eval Error": train_acc,
                        "Validation Eval Error": val_acc,
                        "Test Eval Error": test_acc,
                        "Train R2": train_r_2,
                        "Validation R2": val_r_2,
                        "Test R2": test_r_2,
                        "Train R2_lin": train_r_2_lin,
                        "Validation R2_lin": val_r_2_lin,
                        "Test R2_lin": test_r_2_lin,
                    }
                )
                # print("Train Eval Error: ", float(train_acc), " Validation Eval Error: ", float(val_acc), " Test Eval Error: ", float(test_acc), " Train R2: ", float(train_r_2), " Validation R2: ", float(val_r_2), " Test R2: ", float(test_r_2))

    else:

        def train_batch(data, model, optimizer, criterion):
            cumu_loss = 0
            feat = data.x.to(device)
            edges = data.edge_index.to(device)
            y = data.y.to(device)

            # Forward pass ➡
            out, _ = model(feat, edges)
            loss = criterion(out[data.train_mask], y[data.train_mask])
            cumu_loss += loss.item()
            # Backward pass ⬅

            optimizer.zero_grad()
            loss.backward()

            # Step with optimizer
            optimizer.step()
            # wandb.log({"Loss": loss.item()})
            return cumu_loss

        def test(model, data, run):
            model.eval()

            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.to(device)

            # Run the model on some test examples
            with torch.no_grad():
                pred, _ = model(data.x, data.edge_index)
                accs = []
                r_2 = []
                r_2_lin = []
                for mask in [data.train_mask, data.val_mask, data.test_mask]:
                    accs.append(F.mse_loss(pred[mask], data.y[mask]))
                    r_2.append(float(r2_score(data.y[mask].cpu(), pred[mask].cpu())))
                    r_2_lin.append(float(_r_squared_linreg(data.y[mask].cpu(), pred[mask].cpu())))

                train_acc, val_acc, test_acc = accs
                train_r_2, val_r_2, test_r_2 = r_2
                train_r_2_lin, val_r_2_lin, test_r_2_lin = r_2_lin
                run.log(
                    {
                        "Train Eval Error": train_acc,
                        "Validation Eval Error": val_acc,
                        "Test Eval Error": test_acc,
                        "Train R2": train_r_2,
                        "Validation R2": val_r_2,
                        "Test R2": test_r_2,
                        "Train R2_lin": train_r_2_lin,
                        "Validation R2_lin": val_r_2_lin,
                        "Test R2_lin": test_r_2_lin,
                    }
                )
                # print("Train Eval Error: ", float(train_acc), " Validation Eval Error: ", float(val_acc), " Test Eval Error: ", float(test_acc), " Train R2: ", float(train_r_2), " Validation R2: ", float(val_r_2), " Test R2: ", float(test_r_2))

    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config, data, project)

    print("Create Attention Matrix plot...")
    # Model Eval
    model.eval()

    with torch.no_grad():
        (index, att) = model(data.x, data.edge_index)[1]

    if heads != 1:
        att_mean = att.mean(dim=-1)
        adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att_mean)
    else:
        adj = pyg_utils.to_scipy_sparse_matrix(data.edge_index, att)

    # adj = pyg_utils.to_scipy_sparse_matrix(index.data, att.data)
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    # plt.figure(figsize=(14, 14))
    ax = sns.heatmap(adj.todense(), cmap="Reds")
    ax.set_aspect("equal")
    # plt.savefig('/Users/benjaminweinert/Desktop/test_heatmap.png', dpi=600)
    # plt.show()
    fig.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Attention Matrix": wandb.Image("plot.png")})
    # wandb.log({"Attention Matrix": fig})
    print("done.\n")

    print("Create Attention PCA...")
    scaled_adj = preprocessing.scale(np.asarray(adj.todense()))
    adj_eval = pd.DataFrame(scaled_adj)
    pca = PCA(n_components=700)  # create a PCA object
    pca.fit(adj_eval)  # do the math
    pca_data = pca.transform(adj_eval)  # get PCA coordinates for scaled_data

    print("done.\n")

    print("Create PCA Elbow plot...")
    # The following code constructs the Scree plot
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var) + 1)]

    # plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    ax.plot(range(1, len(per_var) + 1), per_var)
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_xlabel("Principal Component")
    # plt.show()
    wandb.log({"Attention PCA": wandb.Image(fig)})
    print("done.\n")

    print("Create Attention UMAP plot...")
    # Adj UMAP
    umap = UMAP(n_components=2, init="random", random_state=0).fit_transform(pca_data)

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    name = "tab20"
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(10, 10)
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(umap, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Attention UMAP": wandb.Image("plot.png")})

    # wandb.log({"Attention UMAP": wandb.Image(fig)})
    # wandb.log({"Attention UMAP": fig})
    print("done.\n")

    print("Create Attention TSNE plot...")
    # Adj TSNE
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(pca_data)

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    name = "tab20"
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.axis('off')
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(10, 10)
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(tsne, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Attention TSNE": wandb.Image("plot.png")})

    # wandb.log({"Attention TSNE": wandb.Image(fig)})
    # wandb.log({"Attention TSNE": fig})
    print("done.\n")

    pred = model(data.x, data.edge_index)[0]
    if model_type in ["GAT_negbin", "GAT_linear_negbin"]:
        pred, var = pred
    pred = pred.data

    print("Model PCA...")
    scaled_data_2 = preprocessing.scale(pred.cpu())
    pca_2 = PCA()
    pca_2 = PCA(n_components=10)  # create a PCA object
    pca_2.fit(scaled_data_2)  # do the math
    pca_data_2 = pca_2.transform(scaled_data_2)  # get PCA coordinates for scaled_data
    print("done.\n")

    print("Create Model PCA Elbow plot...")
    # The following code constructs the Scree plot
    per_var_2 = np.round(pca_2.explained_variance_ratio_ * 100, decimals=1)
    labels = ["PC" + str(x) for x in range(1, len(per_var_2) + 1)]

    # plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    ax.plot(range(1, len(per_var_2) + 1), per_var_2)
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_xlabel("Principal Component")
    plt.show()
    wandb.log({"Model PCA": wandb.Image(fig)})
    # wandb.log({"Model PCA": fig})
    print("done.\n")

    print("Create Model UMAP plot...")
    # Trained UMAP
    umap = UMAP(n_components=2, init="random", random_state=0).fit_transform(pca_data_2[:, :7])

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    name = "tab20"
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.axis('off')
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(10, 10)
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(umap, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Model UMAP": wandb.Image("plot.png")})

    # wandb.log({"Model UMAP": wandb.Image(fig)})
    # wandb.log({"Model UMAP": fig})
    print("done.\n")

    print("Create Model TSNE plot...")
    # Adj TSNE
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(pca_data_2[:, :7])

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots()
    name = "tab20"
    cmap = get_cmap(name)
    colors = cmap.colors
    ax.set_prop_cycle(color=colors)
    ax.set_axis_off()
    # plt.axis('off')
    # plt.figure(figsize=(10, 10))

    fig.set_size_inches(10, 10)
    plt.subplots_adjust(right=0.7)

    groups = pd.DataFrame(tsne, columns=["x", "y"]).assign(category=list(adata.obs[cluster_key])).groupby("category")
    for name, points in groups:
        ax.scatter(points.x, points.y, label=name, s=20)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Model TSNE": wandb.Image("plot.png")})

    # wandb.log({"Model TSNE": wandb.Image(fig)})
    # wandb.log({"Model TSNE": fig})
    print("done.\n")

    # print("Create Tissue plot...")
    # if tissue!=0:
    #    sample_name_plot = adata.obs["sample_name"].cat.categories[tissue-1]
    # else:
    #    sample_name_plot = ["AT3-BRA5-FO-1_0","AT3-BRA5-FO-1_1","AT3-BRA5-FO-3_1","AT3-BRA5-FO-4_1"]

    # plt.rcParams['font.size'] = 14

    # if dataset == "GBM":
    #    sq.pl.spatial_scatter(adata, color="cluster", library_id = sample_name_plot, library_key="sample_name")
    # else:
    #    sq.pl.spatial_scatter(adata, color="cluster")

    # plt.savefig('plot.png', format='png', bbox_inches='tight')
    # wandb.log({"Tissue plot": wandb.Image('plot.png')})
    # print("done.\n")

    print("Top Ten most important genes receiver and sender nodes:")

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

    # Create violin plot
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 6))
    # ax = sns.violinplot(data=df_plot.iloc[:,:10], inner="box", stripplot=False)
    ax = sns.boxplot(data=df_plot.iloc[:, :10], color="tab:blue", showfliers=False)
    plt.title("Weight Distribution Top 10 Receiving Genes")
    plt.xlabel("Genes")
    plt.ylabel("W_r Values")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace labels in list with * if they are in gbm_genes
    labels = [label + "*" if label in gbm_genes else label for label in labels]

    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.show()

    plt.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Violin W_r": wandb.Image("plot.png")})

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

    # Create violin plot
    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 6))
    # ax = sns.violinplot(data=df_plot.iloc[:,:10], inner="box", stripplot=False)
    ax = sns.boxplot(data=df_plot.iloc[:, :10], color="tab:blue", showfliers=False)
    plt.title("Weight Distribution Top 10 Sender Genes")
    plt.xlabel("Genes")
    plt.ylabel("W_s Values")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace labels in list with * if they are in gbm_genes
    labels = [label + "*" if label in gbm_genes else label for label in labels]

    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.show()

    plt.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Violin W_s": wandb.Image("plot.png")})

    df = pd.DataFrame({"Receiver Genes": gene_plot_name_w, "Sender Genes": gene_plot_name_w2})
    wandb.log({"Top ten weighted genes": wandb.Table(dataframe=df)})
    df = pd.DataFrame({"all_genes": list(set(gene_plot_name_w).union(gene_plot_name_w2))})
    wandb.log({"Top ten weighted genes list": wandb.Table(dataframe=df)})
    print("done.\n")

    print("Top k most important input genes (Saliency):")

    def compute_saliency(model, data):
        model.eval()
        model.zero_grad()

        data.x = data.x.clone().detach().requires_grad_(True)

        # Forward pass through the model
        output = model(data.x, data.edge_index)[0]
        if model_type in ["GAT_negbin", "GAT_linear_negbin"]:
            output, _ = output

        output = torch.sum(output)

        output.backward()

        saliency = data.x.grad.detach().abs()

        return saliency.cpu().numpy()

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

    plt.rcParams["font.size"] = 14
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df_plot.iloc[:, :10], color="tab:blue", showfliers=False)
    # ax = sns.boxplot(data=df_plot.iloc[:,:])
    plt.title("Saliency Distributions Top 10 Input Genes")
    plt.xlabel("Genes")
    plt.ylabel("Saliency Scores")

    labels = [item.get_text() for item in ax.get_xticklabels()]

    # Replace labels in list with * if they are in gbm_genes
    labels = [label + "*" if label in gbm_genes else label for label in labels]

    # Set the new labels
    ax.set_xticklabels(labels)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig("plot.png", format="png", bbox_inches="tight")
    wandb.log({"Saliency Boxplots": wandb.Image("plot.png")})

    df = pd.DataFrame({"all_genes": gene_plot_name_w})
    wandb.log({"Top ten input genes list (Saliency Scores)": wandb.Table(dataframe=df)})
    print("done.\n")

    return model, data
