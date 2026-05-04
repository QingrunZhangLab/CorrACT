
# CorrACT
CorrACT (Correlation-Aware Clustering and Trajectory Inference)
CorrACT is a novel algorithm tailored for single-cell data analysis. Moving beyond the limitations of isolated cell states and single marker genes, CorrACT leverages gene-gene correlation patterns to capture the fluid, continuous spectrum of cellular biology.



           
Clustering            | Trajectory
:-------------------------:|:-------------------------:
<img width="4185" height="2347" alt="SuppFig1" src="https://github.com/user-attachments/assets/de8ed5db-601e-46ed-8052-f53550e0ee02" /> | <img width="2663" height="1739" alt="g7" src="https://github.com/user-attachments/assets/0fedff42-9a68-4fd9-95d4-093cbfed1dd3" />

The input for CorrACT is a gene expression matrix with cells in rows and genes in columns. You can read the file using read_csv function from pandas and run run_corract_pipeline from corract_allfunctions.py file to identify clusters and trajectory in your data. 

```
import pandas as pd
import corract_allfunctions as cr

print("Loading data...")
data = pd.read_csv("sample_data/1099.csv", index_col=0)

clusters, split_history, backbone, trajectories, paths, score_df, typed_mvp = cr.run_corract_pipeline(
    data, 
    min_cells=50, 
    min_child=20, 
    min_silhouette=0.25, 
    n_hvg=1000, 
    n_hvg_traj=2000
)
```

You can use the following code to get dendogram of the resulting clustering results. 
```
Z, dd, leaf_ids, fig = cr.dendrogram_with_rules(
    cell_clusters=clusters,
    labels_df=true_labels,
    split_rules=split_history,
    figsize=(8, 8),
    leaf_rotation=90
)
```

To get lineages and their pseudotime, you can run:

```
cr.plot_pipeline_results(clusters, backbone, trajectories, paths, "Default ")
```
