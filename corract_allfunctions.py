import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster import hierarchy
from dynamicTreeCut import cutreeHybrid 
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, ranksums

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# UTILITIES
# ============================================================
def compute_tom(X_cells_by_genes: np.ndarray, power: int = 6) -> np.ndarray:
    C = np.abs(np.corrcoef(X_cells_by_genes, rowvar=False))
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    A = C ** power
    np.fill_diagonal(A, 0.0)
    k = A.sum(axis=1)
    L = A @ A
    denom = np.minimum.outer(k, k) + 1.0 - A
    W = (L + A) / (denom + 1e-12)
    np.fill_diagonal(W, 1.0)
    return W

def bh_fdr(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    qvals = ranked * n / (np.arange(n) + 1)
    qvals = np.minimum.accumulate(qvals[::-1])[::-1]
    out = np.empty_like(qvals)
    out[order] = np.clip(qvals, 0, 1)
    return out

# ============================================================
# PHASE 1: DISCRETE RECURSIVE CLUSTERING (Probabilistic)
# ============================================================
def dynamic_hvg_selection(data_sub, n_hvg=500, min_expr_frac=0.05):
    expr_frac = (data_sub > 0).mean(axis=0)
    valid_genes = expr_frac[expr_frac >= min_expr_frac].index
    if len(valid_genes) == 0: return []
    variances = data_sub[valid_genes].var(axis=0)
    return variances.sort_values(ascending=False).head(n_hvg).index.tolist()

def dynamic_hvg_selection_traj(data_sub, n_hvg_traj=500, min_expr_frac=0.05):
    expr_frac = (data_sub > 0).mean(axis=0)
    valid_genes = expr_frac[expr_frac >= min_expr_frac].index
    if len(valid_genes) == 0: return []
    variances = data_sub[valid_genes].var(axis=0)
    return variances.sort_values(ascending=False).head(n_hvg_traj).index.tolist()

def probabilistic_gmm_split(data_sub, module_genes, min_cells, confidence_threshold=0.95, random_state=0):
    """Fits a GMM and isolates confident cells from ambiguous bridge cells (-1)."""
    genes = [g for g in module_genes if g in data_sub.columns]
    if len(genes) < 2 or data_sub.shape[0] < 2 * min_cells: return None, None
    
    X = StandardScaler().fit_transform(data_sub[genes].values)
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=random_state)
    gmm.fit(X)
    
    probs = gmm.predict_proba(X)
    max_probs = np.max(probs, axis=1)
    labels01 = np.argmax(probs, axis=1)
    
    # Label cells below the threshold as -1 (Ambiguous/Bridge)
    labels01[max_probs < confidence_threshold] = -1
    
    # Calculate Silhouette ONLY on confident cells
    confident_mask = labels01 != -1
    if len(np.unique(labels01[confident_mask])) < 2: 
        return None, None
    if np.sum(labels01 == 0) < min_cells or np.sum(labels01 == 1) < min_cells: 
        return None, None
        
    score = silhouette_score(X[confident_mask], labels01[confident_mask])
    return labels01, score

def network_modules_from_data(data_sub, power=6, deepSplit=4, min_genes=10, ratio_thres=5.0):
    genes = list(data_sub.columns)
    n = len(genes)
    if n < min_genes: return {}
    T = compute_tom(data_sub.values, power=power)
    distances = pdist(1.0 - T, metric="euclidean")
    Z = linkage(distances, method="complete")
    clusters = cutreeHybrid(Z, distances, deepSplit=deepSplit)["labels"]
    module_genes = {}
    uniq, inv = np.unique(clusters, return_inverse=True)
    row_sum, diag = T.sum(axis=1), np.diag(T)

    for k, mod_id in enumerate(uniq):
        if mod_id == 0: continue
        idx = np.where(inv == k)[0]
        m = idx.size
        block = T[np.ix_(idx, idx)]
        S_in = block.sum() - diag[idx].sum()
        cohesion = S_in / (m * (m - 1)) if m > 1 else 0.0
        S_out = row_sum[idx].sum() - block.sum()
        external = S_out / (m * (n - m)) if (n - m) > 0 else 0.0
        ratio = (cohesion / external) if external != 0 else np.inf
        if ratio >= ratio_thres:
            module_genes[int(mod_id)] = [genes[i] for i in idx]
    return module_genes

def run_recursive_clustering(data, min_cells=50, min_child=20, min_silhouette=0.20, max_depth=15, n_hvg=500):
    cell_labels = pd.Series(index=data.index, dtype=object, name="cluster")
    split_history = {} 
    
    def recurse(cell_index, node_id, depth):
        if depth >= max_depth or len(cell_index) < min_cells:
            cell_labels.loc[cell_index] = node_id; return

        data_sub = data.loc[cell_index]
        hvgs = dynamic_hvg_selection(data_sub, n_hvg=n_hvg)
        if not hvgs:
            cell_labels.loc[cell_index] = node_id; return

        network_genes = network_modules_from_data(data_sub[hvgs])
        best = None
        for mid, genes in network_genes.items():
            labels01, score = probabilistic_gmm_split(data_sub, genes, min_child, confidence_threshold=0.95)
            if score is not None and score >= min_silhouette:
                if best is None or score > best[0]:
                    best = (score, mid, labels01, genes)

        if best is None:
            cell_labels.loc[cell_index] = node_id; return

        score, mid, labels01, active_genes = best
        
        # Isolate the groups
        child0 = data_sub.index[labels01 == 0]
        child1 = data_sub.index[labels01 == 1]
        bridge = data_sub.index[labels01 == -1]
        
        # Lock in bridge cells
        if len(bridge) > 0:
            cell_labels.loc[bridge] = f"Bridge_{node_id}"
        
        split_history[node_id] = {
            "module_id": mid,
            "silhouette": score,
            "genes": active_genes,
            "size": len(cell_index),
            "bridge_size": len(bridge),
            "child_0_node": node_id + "1",
            "child_1_node": node_id + "2"
        }
        
        # Recurse only on confident cells
        recurse(child0, node_id + "1", depth + 1)
        recurse(child1, node_id + "2", depth + 1)

    print("--- Phase 1: Running Probabilistic Recursive Clustering ---")
    recurse(data.index, "C", 0)
    return cell_labels, split_history

# ============================================================
# PHASE 2: DYNAMIC TRAJECTORY INFERENCE (Bridge Routing)
# ============================================================
def identify_intra_cluster_modules(data, labels, min_mod_size=10, deepSplit=4, n_hvg_traj=1000):
    records = []
    labels_str = labels.astype(str)
    
    # Ignore bridge cells for module finding
    confident_clusters = [cl for cl in labels_str.unique() if not cl.startswith("Bridge_")]
    
    for cl in sorted(confident_clusters):
        cells = labels.index[labels_str == cl]
        if len(cells) < 20: continue
        sub = data.loc[cells]
        hvgs = dynamic_hvg_selection_traj(sub, n_hvg_traj=n_hvg_traj)
        if len(hvgs) < min_mod_size: continue
        
        sub_clean = sub[hvgs]
        tom = compute_tom(sub_clean.values)
        diss = 1.0 - tom
        Z = linkage(squareform(diss, checks=False), method="average")
        out = cutreeHybrid(Z, distM=diss, deepSplit=deepSplit, minClusterSize=min_mod_size)
        
        gene_names = sub_clean.columns; labs = out['labels']
        for l in pd.Series(labs).unique():
            if l == 0: continue
            genes = [g for g, lab in zip(gene_names, labs) if lab == l]
            records.append({"module_id": f"{cl}_M{l}", "source_cluster": cl, "genes": genes})
    return pd.DataFrame(records)

def score_and_partition(data, labels, module_df):
    all_genes = list(set([g for genes in module_df['genes'] for g in genes if g in data.columns]))
    sub_data = np.log1p(data[all_genes])
    Z_data = pd.DataFrame(StandardScaler().fit_transform(sub_data), index=data.index, columns=all_genes)
    
    score_df = pd.DataFrame({row['module_id']: Z_data[row['genes']].mean(axis=1) for _, row in module_df.iterrows()})
    labels_str = labels.astype(str)
    
    # Calculate variance metrics ignoring bridge cells
    confident_mask = ~labels_str.str.startswith("Bridge_")
    across_vars = score_df[confident_mask].groupby(labels_str[confident_mask]).mean().var()
    cluster_vars = score_df[confident_mask].groupby(labels_str[confident_mask]).var() 
    
    va_cut, vw_cut = across_vars.quantile(0.75), cluster_vars.stack().quantile(0.75)
    va_low = across_vars.quantile(0.50)
    
    mvp = []
    for _, row in module_df.iterrows():
        m_id, src_cl = row['module_id'], str(row['source_cluster'])
        va, vw = across_vars[m_id], cluster_vars.loc[src_cl, m_id]
        if va >= va_cut and vw < va_low: m_type = "Type 1" 
        elif va < va_low and vw >= vw_cut: m_type = "Type 2" 
        elif va >= va_cut and vw >= vw_cut: m_type = "Type 3" 
        else: m_type = "Type 4" 
        mvp.append({"module_id": m_id, "cluster": src_cl, "module_type": m_type})
    return score_df, pd.DataFrame(mvp)

def build_lineage_backbone(score_df, labels, typed_mvp, n_neighbors=15):
    """Builds the macro-trajectory by forcing paths to route through ambiguous bridge cells."""
    backbone_mods = typed_mvp[typed_mvp['module_type'].isin(["Type 1", "Type 3"])]['module_id']
    if len(backbone_mods) == 0: return nx.Graph()
    
    # 1. Compress data into Module Space
    coords = PCA(n_components=min(15, len(backbone_mods))).fit_transform(score_df[backbone_mods])
    coords_df = pd.DataFrame(coords, index=score_df.index)
    
    # 2. Separate Anchors and Bridge Cells
    labels_str = labels.astype(str)
    is_bridge = labels_str.str.startswith("Bridge_")
    
    bridge_cells_df = coords_df[is_bridge]
    centroids = coords_df[~is_bridge].groupby(labels_str[~is_bridge]).mean()
    
    # 3. Combine for Routing
    routing_data = pd.concat([centroids, bridge_cells_df])
    anchor_indices = {cluster: i for i, cluster in enumerate(centroids.index)}
    
    # --- NEW SAFETY CHECK ---
    # Dynamically scale n_neighbors so it never exceeds the total number of available points
    num_routing_points = len(routing_data)
    safe_neighbors = min(n_neighbors, num_routing_points - 1)
    
    # If there are 1 or 0 points total, no trajectory can be built
    if safe_neighbors <= 0:
        return nx.Graph()
    # ------------------------
    
    # 4. Build the Micro-Graph using the safe neighbor count
    knn_matrix = kneighbors_graph(routing_data.values, n_neighbors=safe_neighbors, mode='distance')
    G_micro = nx.from_scipy_sparse_array(knn_matrix)
    
    # 5. Build the Macro-Backbone
    G_macro = nx.Graph()
    anchors = list(centroids.index)
    
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            start_cl, end_cl = anchors[i], anchors[j]
            start_idx, end_idx = anchor_indices[start_cl], anchor_indices[end_cl]
            
            try:
                # Must route through the G_micro edges (the bridge!)
                path_dist = nx.shortest_path_length(G_micro, source=start_idx, target=end_idx, weight='weight')
                G_macro.add_edge(start_cl, end_cl, weight=path_dist)
            except nx.NetworkXNoPath:
                pass # Disconnected island!
                
    if len(G_macro.nodes()) == 0: return G_macro
    return nx.minimum_spanning_tree(G_macro, weight="weight")

def get_nonlinear_local_ptime(target_cl, score_df, typed_mvp, labels):
    cells = labels.index[labels.astype(str) == str(target_cl)]
    relevant_mods = typed_mvp[(typed_mvp['cluster'] == str(target_cl)) & 
                              (typed_mvp['module_type'].isin(["Type 2", "Type 3"]))]['module_id'].tolist()
    if not relevant_mods: return pd.Series(0.0, index=cells)

    X = StandardScaler().fit_transform(score_df.loc[cells, relevant_mods].values)
    dists = cdist(X, X, 'sqeuclidean')
    W = np.exp(-dists / (2 * np.median(dists) + 1e-8))
    D_inv = np.diag(1.0 / (W.sum(axis=1) + 1e-8))
    P = D_inv @ W 
    evals, evecs = np.linalg.eig(P)
    diffusion_time = evecs[:, np.argsort(evals.real)[-2]].real
    norm_time = (diffusion_time - diffusion_time.min()) / (diffusion_time.max() - diffusion_time.min() + 1e-8)
    if np.corrcoef(norm_time, X.sum(axis=1))[0,1] < 0: norm_time = 1 - norm_time
    return pd.Series(norm_time, index=cells)

def stitch_global_pseudotime(path, local_pt_dict, score_df, labels):
    global_results = []
    current_offset = 0.0
    for i, cluster in enumerate(path):
        cluster_cells = labels[labels == cluster].index
        local_vals = local_pt_dict[cluster].loc[local_pt_dict[cluster].index.intersection(cluster_cells)]
        weight = 1.0
        if i < len(path) - 1:
            c1 = score_df.loc[labels == path[i]].mean()
            c2 = score_df.loc[labels == path[i+1]].mean()
            weight = np.linalg.norm(c1 - c2)
        global_results.append(current_offset + (local_vals * weight))
        current_offset += weight

    full = pd.concat(global_results)
    return (full - full.min()) / (full.max() - full.min())

def compute_trajectories(backbone, score_df, typed_mvp, labels, root_cluster=None):
    rooted_paths = {}
    components = list(nx.connected_components(backbone))
    
    for i, comp in enumerate(components):
        comp_graph = backbone.subgraph(comp)
        if len(comp) < 2: continue
            
        if root_cluster and root_cluster in comp:
            local_root = root_cluster
        else:
            local_root = max(dict(comp_graph.degree()).items(), key=lambda x: x[1])[0]
            
        tips = [n for n in comp_graph.nodes() if comp_graph.degree(n) == 1 and n != local_root]
        if not tips and len(comp) == 2:
            tips = [n for n in comp_graph.nodes() if n != local_root]
            
        for tip in tips:
            path_name = f"Island{i+1}_{local_root}_to_{tip}"
            rooted_paths[path_name] = nx.shortest_path(comp_graph, local_root, tip)
            
    # Calculate local time only for discrete clusters (ignore bridges)
    confident_clusters = [cl for cl in labels.unique() if not str(cl).startswith("Bridge_")]
    local_pt_results = {cl: get_nonlinear_local_ptime(cl, score_df, typed_mvp, labels) 
                        for cl in confident_clusters}
    
    global_trajectories = pd.DataFrame(index=score_df.index)
    for name, path in rooted_paths.items():
        global_trajectories[name] = stitch_global_pseudotime(path, local_pt_results, score_df, labels)
        
    return global_trajectories, rooted_paths

def reroot_lineage(data, backbone, score_df, typed_mvp, labels, root_by="cluster", target=None):
    if root_by == "gene":
        if target not in data.columns: raise ValueError(f"Gene {target} not found in dataset.")
        root_cluster = data.groupby(labels)[target].mean().idxmax()
        print(f"Auto-rooted to '{root_cluster}' based on max {target} expression.")
    elif root_by == "cell":
        if target not in labels.index: raise ValueError(f"Cell {target} not found.")
        root_cluster = labels.loc[target]
        print(f"Auto-rooted to '{root_cluster}' based on cell {target}.")
    elif root_by == "cluster":
        if target not in backbone.nodes(): raise ValueError(f"Cluster {target} not found in backbone.")
        root_cluster = target
        print(f"Rooted explicitly to '{root_cluster}'.")
    else:
        raise ValueError("root_by must be 'gene', 'cell', or 'cluster'")
        
    return compute_trajectories(backbone, score_df, typed_mvp, labels, root_cluster)

def run_corract_pipeline(data: pd.DataFrame, min_cells=50, min_child=20, min_silhouette=0.20, max_depth=15, n_hvg=500, n_hvg_traj=1000):
    clusters, split_history = run_recursive_clustering(data, min_cells, min_child, min_silhouette, max_depth, n_hvg)
    
    print("\n--- Phase 2: Running Dynamic Trajectory Inference ---")
    mod_df = identify_intra_cluster_modules(data, clusters, n_hvg_traj=n_hvg_traj)
    score_df, typed_mvp = score_and_partition(data, clusters, mod_df)
    backbone = build_lineage_backbone(score_df, clusters, typed_mvp)
    
    if len(backbone.nodes()) > 0:
        trajectories, paths = compute_trajectories(backbone, score_df, typed_mvp, clusters, root_cluster=None)
        print(f"Mapped {len(paths)} unique trajectories across {nx.number_connected_components(backbone)} disconnected islands.")
    else:
        trajectories, paths = pd.DataFrame(), {}
        print("No backbone edges formed. Clusters are completely isolated.")
        
    return clusters, split_history, backbone, trajectories, paths, score_df, typed_mvp

# ============================================================
# PHASE 3: BIOLOGICAL DISCOVERY & PLOTTING
# ============================================================
def plot_pipeline_results(clusters, backbone, trajectories, paths, title_prefix=""):
    plt.figure(figsize=(18, 8))
    
    plt.subplot(1, 3, 1)
    if backbone.nodes():
        pos = nx.spring_layout(backbone, seed=42)
        nx.draw(backbone, pos, with_labels=True, node_color='lightblue', node_size=2000, font_weight='bold')
        plt.title(f"{title_prefix}Learned Backbone")

    plt.subplot(1, 3, 2)
    if not trajectories.empty:
        melted_traj = trajectories.melt(var_name='Trajectory', value_name='Pseudotime').dropna()
        sns.violinplot(data=melted_traj, x='Trajectory', y='Pseudotime', palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Global Pseudotime Density")

    plt.subplot(1, 3, 3)
    if paths and not trajectories.empty:
        example_traj = list(paths.keys())[0]
        example_cells = trajectories[example_traj].dropna().index
        sns.boxplot(x=clusters.loc[example_cells], y=trajectories.loc[example_cells, example_traj], palette="muted")
        plt.xticks(rotation=90)
        plt.title(f"Progression: {example_traj}")

    plt.tight_layout()
    plt.show()

def get_cluster_split_logic(split_history, cluster_A, cluster_B):
    lca_node = os.path.commonprefix([str(cluster_A), str(cluster_B)])
    if lca_node not in split_history:
        return f"Clusters '{cluster_A}' and '{cluster_B}' do not share a valid parent split."
        
    split_info = split_history[lca_node]
    genes = split_info['genes']
    
    report = (
        f"--- Split Report ---\n"
        f"Lowest Common Ancestor: Node '{lca_node}'\n"
        f"Separated by: Module {split_info['module_id']} (Silhouette: {split_info['silhouette']:.3f})\n"
        f"Key Genes ({len(genes)} total): {', '.join(genes[:15])}{'...' if len(genes)>15 else ''}\n"
    )
    return report

def plot_single_gene_global(data, global_trajectories, path_name, gene_name):
    if gene_name not in data.columns: return
    if path_name not in global_trajectories.columns: return
        
    cells_in_path = global_trajectories[path_name].dropna()
    expr = np.log1p(data.loc[cells_in_path.index, gene_name])
    
    plt.figure(figsize=(8, 5))
    sns.regplot(x=cells_in_path.values, y=expr.values, 
                scatter_kws={'alpha':0.4, 's':15, 'color':'#2b8cbe'}, 
                line_kws={'color':'black'}, lowess=True)
    plt.title(f"Expression of {gene_name} along {path_name}")
    plt.xlabel("Global Pseudotime")
    plt.ylabel(f"Expression (log1p)")
    plt.show()

def plot_single_gene_local(data, target_cl, score_df, typed_mvp, labels, gene_name):
    if gene_name not in data.columns: return
        
    cells = labels.index[labels.astype(str) == str(target_cl)]
    local_time = get_nonlinear_local_ptime(target_cl, score_df, typed_mvp, labels)
    expr = np.log1p(data.loc[cells, gene_name])
    
    plt.figure(figsize=(6, 4))
    sns.regplot(x=local_time, y=expr.values, 
                scatter_kws={'alpha':0.5, 's':15, 'color':'#d95f02'}, 
                line_kws={'color':'black'}, lowess=True)
    plt.title(f"{gene_name} Maturation inside {target_cl}")
    plt.xlabel("Local Maturation Time (0 to 1)")
    plt.ylabel(f"Expression (log1p)")
    plt.tight_layout()
    plt.show()

def get_temporal_genes(data, target_cl, score_df, typed_mvp, labels, top_n=10):
    cells = labels.index[labels.astype(str) == str(target_cl)]
    sub_data = data.loc[cells]
    local_time = get_nonlinear_local_ptime(target_cl, score_df, typed_mvp, labels)
    results = []
    for gene in sub_data.columns:
        expr = sub_data[gene].values
        if np.mean(expr > 0) < 0.10: continue
        corr, pval = spearmanr(local_time, expr)
        if abs(corr) > 0.3: results.append({'gene': gene, 'spearman_rho': corr, 'abs_rho': abs(corr), 'pval': pval})
    res_df = pd.DataFrame(results)
    if res_df.empty: return res_df
    res_df['fdr'] = bh_fdr(res_df['pval'])
    return res_df[res_df['fdr'] < 0.05].sort_values('abs_rho', ascending=False).head(top_n)

def get_bifurcation_genes(data, global_trajectories, pathA_name, pathB_name, top_n=10):
    cellsA = global_trajectories[pathA_name].dropna().index
    cellsB = global_trajectories[pathB_name].dropna().index
    uniqueA = cellsA.difference(cellsB)
    uniqueB = cellsB.difference(cellsA)
    if len(uniqueA) < 15 or len(uniqueB) < 15: return pd.DataFrame()
    results = []
    for gene in data.columns:
        exprA = data.loc[uniqueA, gene]
        exprB = data.loc[uniqueB, gene]
        if exprA.mean() < 0.05 and exprB.mean() < 0.05: continue
        stat, pval = ranksums(exprA, exprB)
        log2fc = np.log2(exprA.mean() + 1) - np.log2(exprB.mean() + 1)
        if abs(log2fc) > 0.5: results.append({'gene': gene, 'log2FC': log2fc, 'abs_fc': abs(log2fc), 'pval': pval})
    res_df = pd.DataFrame(results)
    if res_df.empty: return res_df
    res_df['fdr'] = bh_fdr(res_df['pval'])
    return res_df[res_df['fdr'] < 0.05].sort_values('abs_fc', ascending=False).head(top_n)

def plot_temporal_genes(data, target_cl, top_genes_df, score_df, typed_mvp, labels):
    cells = labels.index[labels.astype(str) == str(target_cl)]
    local_time = get_nonlinear_local_ptime(target_cl, score_df, typed_mvp, labels)
    genes_to_plot = top_genes_df['gene'].tolist()[:4] 
    
    plt.figure(figsize=(15, 4))
    for i, gene in enumerate(genes_to_plot, 1):
        plt.subplot(1, len(genes_to_plot), i)
        expr = np.log1p(data.loc[cells, gene])
        sns.regplot(x=local_time, y=expr, scatter_kws={'alpha':0.5, 's':15, 'color': 'gray'}, 
                    line_kws={'color':'red'}, lowess=True)
        plt.title(f"{gene} in {target_cl}")
        plt.xlabel("Local Maturation Time")
        plt.ylabel("Expression (log1p)")
    plt.tight_layout()
    plt.show()

def plot_branch_divergence(data, global_trajectories, pathA_name, pathB_name, gene):
    cellsA = global_trajectories[pathA_name].dropna()
    cellsB = global_trajectories[pathB_name].dropna()
    
    plt.figure(figsize=(8, 5))
    sns.regplot(x=cellsA.values, y=np.log1p(data.loc[cellsA.index, gene]), 
                scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'red'}, 
                label=pathA_name, lowess=True)
    sns.regplot(x=cellsB.values, y=np.log1p(data.loc[cellsB.index, gene]), 
                scatter_kws={'alpha':0.3, 's':15}, line_kws={'color':'blue'}, 
                label=pathB_name, lowess=True)
    
    plt.title(f"Divergence of {gene}")
    plt.xlabel("Global Pseudotime")
    plt.ylabel("Expression (log1p)")
    plt.legend()
    plt.show()

def plot_module_dynamics(score_df, module_id, global_trajectories, path_name):
    if module_id not in score_df.columns: return
        
    cells_in_path = global_trajectories[path_name].dropna()
    module_expr = score_df.loc[cells_in_path.index, module_id]
    
    plt.figure(figsize=(8, 5))
    sns.regplot(x=cells_in_path.values, y=module_expr.values, 
                scatter_kws={'alpha':0.5, 's':15, 'color':'teal'}, line_kws={'color':'black'}, lowess=True)
    plt.title(f"Dynamics of Module {module_id} along {path_name}")
    plt.xlabel("Global Pseudotime")
    plt.ylabel("Module Activation Score")
    plt.show()

# ============================================================
# PHASE 4: CUSTOM DENDROGRAM UTILITIES
# ============================================================
def build_children_from_split_rules(split_rules: dict, cell_clusters: pd.Series):
    internal = set(split_rules.keys())
    # Ensure bridge nodes are removed from the pure dendrogram logic if needed, 
    # but currently leaving them allows us to see bridge extraction points!
    leaves = sorted([cl for cl in pd.unique(cell_clusters.astype(str)) if not cl.startswith("Bridge_")])
    all_nodes = list(internal) + leaves
    root = min(all_nodes, key=len) if len(all_nodes) else "C"
    children = {}
    for node in internal:
        children[node] = (node + "1", node + "2")
    return children, leaves, root

def build_linkage_from_tree(children: dict, root: str, leaves: list):
    leaf_ids = sorted(leaves)
    n = len(leaf_ids)
    leaf_to_id = {leaf: i for i, leaf in enumerate(leaf_ids)}
    Z_rows = []
    merge_to_node = []

    def postorder(node: str):
        if node not in children: return leaf_to_id[node], 0.0, 1
        left, right = children[node]
        l_id, l_h, l_cnt = postorder(left)
        r_id, r_h, r_cnt = postorder(right)
        height = max(l_h, r_h) + 1.0
        cluster_id = n + len(Z_rows)
        Z_rows.append([l_id, r_id, height, l_cnt + r_cnt])
        merge_to_node.append(node)
        return cluster_id, height, l_cnt + r_cnt

    postorder(root)
    Z = np.asarray(Z_rows, dtype=float)
    return Z, leaf_ids, merge_to_node

def leaf_label_composition(cell_clusters: pd.Series, labels_df: pd.DataFrame, min_count_to_show: int = 5) -> dict:
    tmp = labels_df.copy()
    tmp["cells"] = tmp["cells"].astype(str)
    tmp = tmp.set_index("cells")
    s = cell_clusters.copy()
    s.index = s.index.astype(str)
    s = s.loc[tmp.index]
    ct = pd.crosstab(tmp["labels"], s)
    
    if hasattr(ct, "map"): ct = ct.map(lambda x: int(x) if x >= min_count_to_show else 0)
    else: ct = ct.applymap(lambda x: int(x) if x >= min_count_to_show else 0)

    mapping = {}
    for leaf in ct.columns:
        parts = []
        for tl in ct.index:
            v = ct.at[tl, leaf]
            if v > 0: parts.append(f"'{tl}':{v}")
        suffix = "; ".join(parts)
        mapping[str(leaf)] = f"{leaf} ({suffix})" if suffix else str(leaf)
    return mapping

def dendrogram_with_rules(cell_clusters: pd.Series, labels_df: pd.DataFrame, split_rules: dict, root: str = None, min_count_to_show: int = 5, figsize=(16, 7), internal_fontsize=8, leaf_rotation=90):
    children, leaves, inferred_root = build_children_from_split_rules(split_rules, cell_clusters)
    if root is None: root = inferred_root
    Z, leaf_ids, merge_to_node = build_linkage_from_tree(children, root=root, leaves=leaves)
    leaf_map = leaf_label_composition(cell_clusters, labels_df, min_count_to_show=min_count_to_show)
    leaf_labels = [leaf_map.get(leaf, leaf) for leaf in leaf_ids]

    # Capture the figure object
    fig = plt.figure(figsize=figsize)
    dd = hierarchy.dendrogram(Z, labels=leaf_labels, orientation="top", count_sort=False, distance_sort=False)

    for i in range(len(dd["icoord"])):
        node = merge_to_node[i]
        rule = split_rules.get(node)
        if rule is None: continue
        x = 0.5 * (dd["icoord"][i][1] + dd["icoord"][i][2])
        y = dd["dcoord"][i][1]
        
        txt = (f"{node} M{rule.get('module_id', '?')}\n")
        plt.text(x, y + 0.05, txt, ha="center", va="bottom", fontsize=internal_fontsize)

    plt.ylabel("Recursive Depth (Split Height)")
    plt.title("Corract Split History & Decision Tree")
    plt.xticks(rotation=leaf_rotation)
    plt.tight_layout()
    
    # Return the figure object instead of showing or saving it here
    return Z, dd, leaf_ids, fig