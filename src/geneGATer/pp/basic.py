import scanpy as sc
import squidpy as sq
from anndata import AnnData


def pre_clustering(
    adata: AnnData,
    n_rings=2,
    n_neighs=6,
    alpha=0.2,
    resolution=1,
    connectivities="connectivities",
    spatial_connectivities="spatial_connectivities",
    library_key="sample_name",
    cluster_key="cluster",
):
    """Create a basic leiden-pre-clustering and spatial graph.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.
    n_rings
        Number of rings to use for the spatial graph.
    n_neighs
        Number of neighbours to use for the spatial graph. (6 Standard for Visium data)
    alpha
        Alpha parameter for the joint graph.
    resolution
        Resolution parameter for the leiden clustering.
    connectivities
        Key for the gene connectivities.
    spatial_connectivities
        Key for the spatial connectivities.
    library_key
        Key for donor names.
    cluster_key
        Key in which clustering is getting saved in adata.obs.

    Returns
    -------
    Raw pre-clustering.
    """
    nn_graph_genes = adata.obsp[connectivities]
    sq.gr.spatial_neighbors(adata, n_rings=n_rings, coord_type="grid", n_neighs=n_neighs, library_key=library_key)
    nn_graph_space = adata.obsp[spatial_connectivities]
    alpha = alpha
    joint_graph = (1 - alpha) * nn_graph_genes + alpha * nn_graph_space
    sc.tl.leiden(adata, adjacency=joint_graph, key_added=cluster_key, resolution=resolution)
    raw_cluster = adata.obs[cluster_key]
    return raw_cluster
