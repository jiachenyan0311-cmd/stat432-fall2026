---
name: robust-unsupervised-clustering
description: Compare at least three clustering methods with dimensionality reduction, separation, and stability checks; select and report one interpretable solution. Use only when explicitly invoked.
---

# Robust Unsupervised Clustering

## Invocation
Use **only when explicitly invoked**, never automatically. Accept a data file, observation ID (optional), numeric feature columns, optional external labels, and seed. Use Python/scikit-learn or equivalent R; set a reproducible seed. Keep external labels hidden until final model selection.

## 1. Data audit and preprocessing
1. Verify rows = observations and columns = features. Report n, p, missingness, and constant features. Exclude IDs, outcomes, and external labels from preprocessing, tuning, and clustering.
2. Remove constant features; handle missing values using a documented featurewise method. Inspect prior transformations and scales before any additional standardization; do not blindly rescale processed data.
3. Examine PCA for high-dimensional/correlated features. As a primary representation, retain enough PCs for about **90% explained variance** (within available rank); record the PC count and actual variance. If PCA is inappropriate, explain why and use processed features. PCA does not itself assign clusters.

## 2. Compare three clustering algorithms
- Fit **K-means**, **agglomerative hierarchical** (try Ward and average linkage where meaningful), and **spectral clustering** (nearest-neighbor affinity): at least three distinct algorithms.
- Try k = 2,...,min(6,n-1), as feasible. Use multiple K-means starts; tune linkage and valid spectral neighbor counts (< n). Track failures/warnings.
- Use the **same primary feature representation** for all methods and all internal separation scores. An embedding (PCA/UMAP) does not count as a clustering algorithm.

## 3. Select by separation and robustness
1. Compute **silhouette score** (Euclidean distance in the common primary space) for valid partitions only (2 <= k < n). Flag singletons, extreme cluster-size imbalance, or algorithm failures; imbalance is not automatically wrong.
2. Assess **stability** for leading candidates: with fixed seed, repeat 20 random 80%-sample-without-replacement refits. Compare each subset's assignments against the full-data partition restricted to that subset using **adjusted Rand index (ARI)**. This compares anonymous cluster assignments, not external labels. Report mean ARI and any failed repetitions; do not invent estimates.
3. Prefer high silhouette; when scores are close, consider stability, degenerate groups, failures, and interpretability. Explain trade-offs; do not create an arbitrary weighted score.
4. Check sensitivity using PCA retaining about 80% and 95% variance where feasible. Do **not** directly compare silhouette scores across different representations. Instead compare leading methods/partitions and use ARI for full-data partition agreement. Acknowledge unstable choices.
5. Lock in **one** algorithm, k, settings, and partition using only unlabeled diagnostics. Reveal external labels, if any, **only afterward** for descriptive interpretation; never reselect based on them.

## 4. Visualize and interpret
- Plot only the **selected partition**, preferably a 2D PCA scatter colored by cluster. Label PC axes with explained variance; explain that 2D visual separation is not proof of real populations. If appropriate, use a clearly described alternative view.
- Report selected algorithm, k, cluster sizes, silhouette, stability (if computed), group patterns, and limitations (n vs p, preprocessing, projection, sensitivity). Optional known labels may be compared with selected groups descriptively, with no causal/biological claims from agreement alone.

## 5. Deliverable
- Produce a clear **one-printed-page maximum** analysis report, including all text, figures, and tables. Briefly name the algorithms compared, criteria, and why the selected method won.
- **Present clustering results from only the selected method.** No scores, cluster-size tables, or plots of competing methods in the report. Include one meaningful figure, interpretation, and limitations.
- Check all numerical statements against output. Keep code reproducible. Append the **entire SKILL.md** after the one-page report as an appendix outside the page limit. Follow the assignment's exact submission filenames and folder.
