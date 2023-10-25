import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _neighbour_detection(adata, groupby="cluster", nrings=2, tresh=0.5, merge=None, library_key="sample_name"):
    if library_key in adata.obs:
        sq.gr.spatial_neighbors(adata, n_rings=nrings, coord_type="grid", n_neighs=6, library_key=library_key)
    else:
        sq.gr.spatial_neighbors(adata, n_rings=nrings, coord_type="grid", n_neighs=6)

    clusters = adata.obs[groupby].cat.categories
    idx = {}
    for cluster in clusters:
        idx[cluster] = np.where(adata.obs[groupby] == cluster)[0]

    if merge is not None:
        new_cluster = np.hstack([idx[merge[0]], idx[merge[1]]])

        adata.obs.iloc[new_cluster, adata.obs.columns.get_loc(groupby)] = merge[0]
        clusters = adata.obs[groupby].cat.categories
        idx = {}
        for cluster in clusters:
            idx[cluster] = np.where(adata.obs[groupby] == cluster)[0]

    neighbors = {}
    for cluster in clusters:
        _, neighbors[cluster] = adata.obsp["spatial_connectivities"][idx[cluster], :].nonzero()
        neighbors[cluster] = np.append(neighbors[cluster], idx[cluster])
        neighbors[cluster] = np.unique(neighbors[cluster])

    tresh = tresh
    sig_neighbors = {}
    for cluster in clusters:
        # print(cluster)
        dummy = []
        for n_cluster in adata[neighbors[cluster], :].obs[groupby].cat.categories:
            a = sum(adata[neighbors[cluster], :].obs[groupby] == n_cluster) / sum(adata.obs[groupby] == n_cluster)
            if 1 > a > tresh:
                dummy.append(n_cluster)
                # print(a)
        sig_neighbors[cluster] = dummy

    sig_idx = {}
    # nonempty_donors = {}
    for cluster in clusters:
        if sig_neighbors[cluster] != []:
            sig_neighbors[cluster]
            sig_idx[cluster] = np.hstack([idx[d] for d in sig_neighbors[cluster]])
            sig_idx[cluster] = np.hstack([sig_idx[cluster], idx[cluster]])
            sig_idx[cluster] = np.unique(sig_idx[cluster])
            # nonempty_donors[cluster] = adata.obs["sample_name"][sig_idx[cluster]].unique()
            # sq.pl.spatial_scatter(adata[sig_idx[cluster], :], img=True, color="cluster", title = f'{cluster} and significant Neighbours')

    return idx, sig_idx, sig_neighbors, neighbors  # , nonempty_donors


def _svg_detection(adata, cluster, tresh, groupby="cluster", nrings=2, merge=None, library_key="sample_name"):
    # idx, sig_idx, sig_neighbors, neighbors, nonempty_donors = _neighbour_detection(
    #    adata, groupby=groupby, nrings=nrings, tresh=tresh, merge=merge
    # )

    idx, sig_idx, sig_neighbors, neighbors = _neighbour_detection(
        adata, groupby=groupby, nrings=nrings, tresh=tresh, merge=merge, library_key=library_key
    )

    cluster = cluster
    test1 = adata.obs[groupby].isin(sig_neighbors[cluster])
    test2 = adata.obs[groupby].isin([cluster])

    adata.obs[f"{cluster}_neighbors"] = "Wayne"
    adata.obs.loc[test1, [f"{cluster}_neighbors"]] = "Neighbour"
    adata.obs.loc[test2, [f"{cluster}_neighbors"]] = cluster

    sc.tl.rank_genes_groups(
        adata,
        groupby=f"{cluster}_neighbors",
        method="wilcoxon",
        pts=True,
        groups=[cluster, "Neighbour"],
        reference="Neighbour",
        use_raw=False,
    )

    adata.uns["rank_genes_groups"]["pts"]["names"] = adata.uns["rank_genes_groups"]["pts"].index
    res = sc.get.rank_genes_groups_df(adata, group=cluster)
    res = res.merge(adata.uns["rank_genes_groups"]["pts"], on="names", how="inner")

    crit1 = res["pvals_adj"] < 0.05
    crit2 = np.exp(res["logfoldchanges"]) > 1.5
    crit3 = res[cluster] > 0.8
    crit4 = pd.DataFrame()
    for sig_n in sig_neighbors[cluster]:
        pct_expressed = np.array(np.sum(adata[idx[sig_n], :].raw.X.todense() > 0, axis=0) / len(idx[sig_n])).tolist()[0]
        crit4[sig_n] = adata.uns["rank_genes_groups"]["pts"][cluster] / pct_expressed

    crit4["names"] = crit4.index
    res = res.merge(crit4, on="names", how="inner").sort_values(
        [cluster, "pvals_adj", "logfoldchanges", "Neighbour"], ascending=[False, True, False, False]
    )
    res["Neighbour"] = res[cluster] / res["Neighbour"]
    crit4 = res["Neighbour"] >= 1
    res = res.sort_values([cluster, "pvals_adj", "logfoldchanges"], ascending=[False, True, False])
    crit = crit1 * crit2 * crit3 * crit4

    de_genes = res.loc[crit, "names"]

    return idx, sig_idx, sig_neighbors, neighbors, de_genes  # , nonempty_donors


def _metagene_detection(
    adata,
    cluster,
    tresh,
    groupby="cluster",
    nrings=2,
    merge=None,
    plot=False,
    verbosse=True,
    base_gene_idx=0,
    library_key="sample_name",
):
    # idx, sig_idx, sig_neighbors, neighbors, de_genes, nonempty_donors = _svg_detection(
    #    adata, cluster=cluster, tresh=tresh, groupby=groupby, nrings=nrings, merge=merge, library_key=library_key
    # )

    idx, sig_idx, sig_neighbors, neighbors, de_genes = _svg_detection(
        adata, cluster=cluster, tresh=tresh, groupby=groupby, nrings=nrings, merge=merge, library_key=library_key
    )

    base_gene = de_genes[de_genes.index[base_gene_idx]]
    base_idx = np.where(adata.var_names == base_gene)[0][0]
    mean_target = np.mean(adata[idx[cluster], base_idx].X)
    gene_in_metagenes = []
    gene_in_metagenes.append(base_idx)

    rest_idx = list(set(range(adata.shape[0])) - set(idx[cluster]))
    control_idx_plus = list(np.where((adata[rest_idx, base_idx].X >= mean_target).todense())[0])

    adata.obs["DE_plus"] = "Wayne"
    adata.obs.iloc[control_idx_plus, adata.obs.columns.get_loc("DE_plus")] = "Control"
    adata.obs.iloc[idx[cluster], adata.obs.columns.get_loc("DE_plus")] = cluster

    sc.tl.rank_genes_groups(
        adata, groupby="DE_plus", method="wilcoxon", groups=[cluster, "Control"], reference="Control", use_raw=False
    )

    i = 0
    n = 0
    p_vals = np.argsort(sc.get.rank_genes_groups_df(adata, group=cluster)["pvals_adj"])

    while i != 1:
        gene_id = np.where(adata.var_names == sc.get.rank_genes_groups_df(adata, group=cluster)["names"][p_vals[n]])[0][
            0
        ]

        if (np.mean(adata[control_idx_plus, gene_id].X) < np.mean(adata[idx[cluster], gene_id].X)) and (
            gene_id not in gene_in_metagenes
        ):
            i = 1
            gene_plus_idx = gene_id
            gene_in_metagenes.append(gene_id)
        else:
            n = n + 1

    control_idx_minus = list(np.where((adata[rest_idx, base_idx].X >= mean_target).todense())[0])
    adata.obs["DE_minus"] = "Wayne"
    adata.obs.iloc[control_idx_minus, adata.obs.columns.get_loc("DE_minus")] = "Control"
    adata.obs.iloc[idx[cluster], adata.obs.columns.get_loc("DE_minus")] = cluster
    sc.tl.rank_genes_groups(
        adata, groupby="DE_minus", method="wilcoxon", groups=[cluster, "Control"], reference=cluster, use_raw=False
    )

    i = 0
    n = 0
    p_vals = np.argsort(sc.get.rank_genes_groups_df(adata, group="Control")["pvals_adj"])
    while i != 1:
        gene_id = np.where(adata.var_names == sc.get.rank_genes_groups_df(adata, group="Control")["names"][p_vals[n]])[
            0
        ][0]

        if (np.mean(adata[control_idx_minus, gene_id].X) > np.mean(adata[idx[cluster], gene_id].X)) and (
            gene_id not in gene_in_metagenes
        ):
            i = 1
            gene_minus_idx = gene_id
            gene_in_metagenes.append(gene_id)
        else:
            n = n + 1

    log_meta_gene_1 = (adata.X[:, base_idx] + adata.X[:, gene_plus_idx] - adata.X[:, gene_minus_idx]).todense()
    if min(log_meta_gene_1) < 0:
        log_meta_gene_1 = log_meta_gene_1 - min(log_meta_gene_1)

    adata.obs["metagene_1"] = log_meta_gene_1

    s = 0
    control_ids_plus = {}
    control_ids_minus = {}
    control_ids_plus[f"{1}"] = control_idx_plus
    control_ids_minus[f"{1}"] = control_idx_minus
    k = 1

    while (s != 1) and (k != 10):
        try:
            mean_target = np.mean(adata.obs[f"metagene_{k}"][idx[cluster]])
            control_ids_plus[f"{k+1}"] = list(np.where(adata.obs[f"metagene_{k}"][rest_idx] >= mean_target)[0])
            if len(control_ids_plus[f"{k+1}"]) > len(control_ids_plus[f"{k}"]):
                s = 1
                if verbosse is True:
                    print(f"In iteration {k+1}, 1. Crit not met.")
            adata.obs["DE_plus"] = "Wayne"
            adata.obs.iloc[control_ids_plus[f"{k+1}"], adata.obs.columns.get_loc("DE_plus")] = "Control"
            adata.obs.iloc[idx[cluster], adata.obs.columns.get_loc("DE_plus")] = cluster

            sc.tl.rank_genes_groups(
                adata,
                groupby="DE_plus",
                method="wilcoxon",
                groups=[cluster, "Control"],
                reference="Control",
                use_raw=False,
            )

            i = 0
            n = 0

            p_vals = np.argsort(sc.get.rank_genes_groups_df(adata, group=cluster)["pvals_adj"])
            while i != 1:
                gene_id = np.where(
                    adata.var_names == sc.get.rank_genes_groups_df(adata, group=cluster)["names"][p_vals[n]]
                )[0][0]

                if (
                    np.mean(adata[control_ids_plus[f"{k+1}"], gene_id].X) < np.mean(adata[idx[cluster], gene_id].X)
                ) and (gene_id not in gene_in_metagenes):
                    i = 1
                    gene_in_metagenes.append(gene_id)
                    gene_plus_idx = gene_id
                else:
                    n = n + 1

            control_ids_minus[f"{k+1}"] = list(np.where(adata.obs[f"metagene_{k}"][rest_idx] >= mean_target)[0])
            adata.obs["DE_minus"] = "Wayne"
            adata.obs.iloc[control_ids_minus[f"{k+1}"], adata.obs.columns.get_loc("DE_minus")] = "Control"
            adata.obs.iloc[idx[cluster], adata.obs.columns.get_loc("DE_minus")] = cluster
            sc.tl.rank_genes_groups(
                adata,
                groupby="DE_minus",
                method="wilcoxon",
                groups=[cluster, "Control"],
                reference=cluster,
                use_raw=False,
            )

            i = 0
            n = 0
            p_vals = np.argsort(sc.get.rank_genes_groups_df(adata, group="Control")["pvals_adj"])
            while i != 1:
                gene_id = np.where(
                    adata.var_names == sc.get.rank_genes_groups_df(adata, group="Control")["names"][p_vals[n]]
                )[0][0]

                if (
                    np.mean(adata[control_ids_minus[f"{k+1}"], gene_id].X) > np.mean(adata[idx[cluster], gene_id].X)
                ) and (gene_id not in gene_in_metagenes):
                    i = 1
                    gene_in_metagenes.append(gene_id)
                    gene_minus_idx = gene_id
                else:
                    n = n + 1

            plus = np.array(adata.X.todense().transpose()[gene_plus_idx]).tolist()[0]
            minus = np.array(adata.X.todense().transpose()[gene_minus_idx]).tolist()[0]

            adata.obs[f"metagene_{k+1}"] = adata.obs[f"metagene_{k}"] + plus - minus
            if min(adata.obs[f"metagene_{k+1}"]) < 0:
                adata.obs[f"metagene_{k+1}"] = adata.obs[f"metagene_{k+1}"] - min(adata.obs[f"metagene_{k+1}"])

            if abs(
                np.mean(adata.obs[f"metagene_{k+1}"][control_ids_plus[f"{k+1}"]])
                - np.mean(adata.obs[f"metagene_{k}"][idx[cluster]])
            ) < (
                np.mean(adata.obs[f"metagene_{k+1}"][control_ids_plus[f"{k+1}"]])
                - np.mean(adata.obs[f"metagene_{k+1}"][idx[cluster]])
            ):
                s = 1
                if verbosse is True:
                    print(f"In iteration {k},2. Crit not met.")
            # sq.pl.spatial_scatter(adata, img=True, color=f"metagene_{k+1}", title = f'Metagene {k+1} should identify cluster {cluster}.')
            k = k + 1
        except IndexError:
            s = 1
            if verbosse is True:
                print(f"Error: In iteration {k}, couldn't find control group.")
            gene_in_metagenes.append("Ok")
            gene_in_metagenes.append("Ok2")

    # if plot is True:

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    #    if merge is not None:
    #        sq.pl.spatial_scatter(
    #            adata,
    #            img=True,
    #            color=f"metagene_{k-1}",
    #            title=f"Metagene {k-1} should identify cluster {cluster}+{merge[1]}.",
    #            library_id=nonempty_donors[cluster],
    #            library_key="sample_name",
    #        )
    #    else:
    #        sq.pl.spatial_scatter(
    #            adata,
    #            img=True,
    #            color=f"metagene_{k-1}",
    #            title=f"Metagene {k-1} should identify cluster {cluster}.",
    #            library_id=nonempty_donors[cluster],
    #            library_key="sample_name",
    #        )
    #    sq.pl.spatial_scatter(
    #        adata[neighbors[cluster], :],
    #        img=True,
    #        color="cluster",
    #        library_id=nonempty_donors[cluster],
    #        library_key="sample_name",
    #    )
    # plt.show()

    gene_in_metagenes = gene_in_metagenes[:-2]

    gg = pd.DataFrame()
    gg["Genes"] = gene_in_metagenes

    sort = []
    for i in range(len(gene_in_metagenes)):
        if (i == 0) or ((i % 2) == 1):
            sort.append(1)
        else:
            sort.append(0)
    gg["sort"] = sort
    gg = gg.sort_values(["sort"], ascending=[False])

    return adata.var_names[gg["Genes"]], k - 1, adata.obs[f"metagene_{k-1}"]  # , nonempty_donors


def _get_quality_metric(
    adata, raw_cluster, main_cluster, tresh, groupby, iter_k, plot=False, verbosse=True, library_key="sample_name"
):
    metagenes_per_comb = pd.DataFrame()
    genes_of_metagenes = {}
    clusters = adata.obs[groupby].cat.categories
    codes = adata.obs[groupby].cat.codes
    codes = np.unique(codes[codes != main_cluster])
    spots_above_median = {}
    spots_above_min = {}
    for cluster in tqdm(clusters[iter_k:]):
        b = 0
        stop = 0
        while (stop != 1) and (b != 5):
            try:
                # print("Test 1")
                adata.obs[groupby] = raw_cluster
                # print("Test 2")
                # test, k, metagene, nonempty_donors = _metagene_detection(
                #    adata,
                #    cluster=main_cluster,
                #    tresh=tresh,
                #    groupby=groupby,
                #    nrings=2,
                #    merge=[main_cluster, cluster],
                #    plot=plot,
                #    verbosse=verbosse,
                #    base_gene_idx=b,
                #    library_key=library_key
                # )

                test, k, metagene = _metagene_detection(
                    adata,
                    cluster=main_cluster,
                    tresh=tresh,
                    groupby=groupby,
                    nrings=2,
                    merge=[main_cluster, cluster],
                    plot=plot,
                    verbosse=verbosse,
                    base_gene_idx=b,
                    library_key=library_key,
                )
                adata.obs["metagene"] = metagene
                # print("Test 3")
                median_value = adata.obs.loc[adata.obs[groupby] == main_cluster]["metagene"].median()
                min_value = adata.obs.loc[adata.obs[groupby] == main_cluster]["metagene"].min()

                # if plot is True:
                #    fig, ax = plt.subplots()
                #    ax.axhline(y=median_value, color="red", linestyle="--", zorder=100)
                #    ax.axhline(y=min_value, color="green", linestyle="--", zorder=101)
                #    sc.pl.violin(
                #        adata,
                #        "metagene",
                #        groupby=groupby,
                #        rotation=90,
                #        inner="box",
                #        stripplot=False,
                #        ax=ax,
                #        library_id=nonempty_donors,
                #        library_key="sample_name",
                #    )

                dummy = pd.DataFrame()
                dummy["cluster"] = adata.obs[groupby]
                dummy["median"] = adata.obs["metagene"] > median_value
                dummy["min"] = adata.obs["metagene"] > min_value

                spots_above_median[cluster] = dummy.groupby("cluster").sum()["median"][codes].sum()
                spots_above_min[cluster] = dummy.groupby("cluster").sum()["min"][codes].sum()

                genes_of_metagenes[cluster] = test
                metagenes_per_comb[cluster] = metagene
                stop = 1
            except IndexError:
                if verbosse is True:
                    print("Error: No control group found or base gene found. -> No Metagene found!")
                b += 1

    return metagenes_per_comb, genes_of_metagenes, spots_above_median, spots_above_min


def getComGenes(
    adata: AnnData, raw_cluster, tresh=0, groupby="cluster", plot=False, verbosse=True, library_key="sample_name"
):
    """Extract communication metagenes.

    Parameters
    ----------
     adata
         The AnnData object to preprocess.
    raw_cluster
         List of raw_cluster for each spot, e.g. result of gt.pp.pre_clustering.
     tresh
         Threshold for the metagene detection, e.g. how many neighbouring spots should be included (the higher the less are included).
     groupby
         Key for the clustering in adata.obs.
     plot
         True or false if you want progress plots of the metagene iterations.
     verbosse
         True or false if you want error messages printed.
    library_key
         Key for donor names.

    Returns
    -------
     Dict of mulitple output parameters:

     out[0]
         Number of spots above the median metagene expression for each cluster/cluster combination.
     out[1]
         Number of spots above the minimum metagene expression for each cluster/cluster combination.
     out[2]["cluster_i"]
         Metagene expression for cluster_i and all other cluster combinations.
     out[3]["cluster_i"]
         Genes of metagenes for cluster_i and all other cluster combinations.
     out[4]
         All found genes in metagenes.

    """
    median_metric_df = pd.DataFrame()
    min_metric_df = pd.DataFrame()
    clusters = adata.obs[groupby].cat.categories
    all_genes = []
    metagenes_total = {}
    genes_in_metagenes_total = {}
    iter_k = 0

    for cluster in clusters:
        (
            metagene_per_comb,
            genes_of_metagenes,
            spots_above_median,
            spots_above_min,
        ) = _get_quality_metric(
            adata,
            raw_cluster,
            main_cluster=cluster,
            tresh=tresh,
            groupby=groupby,
            iter_k=iter_k,
            plot=plot,
            verbosse=verbosse,
            library_key=library_key,
        )
        median_metric_df[cluster] = spots_above_median
        min_metric_df[cluster] = spots_above_min

        metagenes_total[cluster] = metagene_per_comb
        genes_in_metagenes_total[cluster] = genes_of_metagenes

        for keys in genes_of_metagenes.keys():
            all_genes.extend(genes_of_metagenes[keys].tolist())

        iter_k += 1

    all_genes = list(set(all_genes))

    return median_metric_df, min_metric_df, metagenes_total, genes_in_metagenes_total, all_genes
