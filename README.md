
# scRNA-seq Imputation and Cell Type Clustering Pipeline

A two-stage pipeline for single-cell RNA sequencing (scRNA-seq) analysis that combines dropout imputation and cell type identification into a single run.

---



```

 
```

### Stage 1 — ccImpute

scRNA-seq data often contains zero expression values that are not truly zero, but result from technical limitations during sequencing — known as **dropout events**. ccImpute identifies these positions and replaces them with estimated values by:

1. Computing a weighted cell–cell correlation matrix
2. Reducing dimensions via truncated SVD
3. Running an ensemble of K-means clustering to build a consensus matrix
4. Detecting likely dropout positions based on the consensus
5. Imputing dropout values using a weighted average of similar cells

### Stage 2 — scCLUE

Using the imputed expression matrix from Stage 1, scCLUE identifies biologically meaningful cell type groups by:

1. Filtering genes by biological pathway membership and expression/variance criteria
2. Building an ensemble of KNN graphs using UMAP and PCA dimensionality reductions
3. Aggregating graphs into a consensus network
4. Applying Louvain community detection to partition cells into clusters
5. Merging over-partitioned clusters using t-test-based separation scores

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn joblib anndata umap-learn igraph networkx python-louvain seaborn matplotlib
```

---

## Input Files

| File | Description |
|---|---|
| `dataset.csv` | Normalized gene expression matrix (genes × cells). Can be any normalized scRNA-seq dataset — rows are genes, columns are cells. |
| `dataset_label.csv` | Cell type labels for evaluation (optional, used for ARI/NMI/Purity metrics) |
| `exported_pathways.csv` | Biological pathway gene sets — rows are pathway names, columns are member genes |

> The pipeline accepts any normalized scRNA-seq dataset. Simply update the file paths in **Section 1** of `main_pipeline.py` to point to your own data.

---

## Usage

**1. Update the input file paths in Section 1 of `main_pipeline.py`:**

```python
logX = pd.read_csv("your_expression_data.csv", index_col=0, engine="python")
data_label  = pd.read_csv("your_labels.csv", index_col=0)
pathways_df = pd.read_csv("exported_pathways.csv", index_col=0, low_memory=False, dtype=object)
```

**2. Run the pipeline:**

```bash
python main_pipeline.py
```

**3. Output:**

- `dataset_imputed.csv` — imputed expression matrix saved after Stage 1
- Cluster assignments and evaluation metrics (ARI, NMI, Purity) printed to console

---

## Key Parameters

### ccImpute

| Parameter | Default | Description |
|---|---|---|
| `nCeil` | 2000 | Maximum cells used for SVD component selection |
| `svdMaxRatio` | 0.08 | Fraction of SVD components retained |
| `maxSets` | 8 | Number of K-means ensemble subsets |
| `k` | auto | Number of clusters (auto-estimated if not set) |
| `nCores` | 4 | Number of parallel workers |

### scCLUE

| Parameter | Default | Description |
|---|---|---|
| `nEns` | 15 | Number of ensemble iterations |
| `K` | [5, 10] | KNN neighbour counts per iteration |
| `nCls` | 3 | Target number of cell type clusters |
| `resolution` | 1.0 | Louvain resolution parameter |

---

## Datasets

This pipeline was evaluated on the following publicly available scRNA-seq datasets:

| Dataset | Cells | Cell Types | Genes | Reference |
|---|---|---|---|---|
| Buettner | 182 | 3 | 38,293 | Buettner, F., Natarajan, K.N., Casale, F.P., et al. (2015). Computational analysis of cell-to-cell heterogeneity in single-cell RNA-sequencing data reveals hidden subpopulations of cells. *Nature Biotechnology*, 33(2), 155–160. https://doi.org/10.1038/nbt.3102 |
| Usoskin | 622 | 4 | 19,534 | Usoskin, D., Furlan, A., Islam, S., et al. (2015). Unbiased classification of sensory neuron types by large-scale single-cell RNA sequencing. *Nature Neuroscience*, 18(1), 145–153. https://doi.org/10.1038/nn.3881 |
| Darmanis | 466 | 9 | 22,085 | Darmanis, S., Sloan, S.A., Zhang, Y., et al. (2015). A survey of human brain transcriptome diversity at the single cell level. *Proceedings of the National Academy of Sciences*, 112(23), 7285–7290. https://doi.org/10.1073/pnas.1507125112 |
| Fletcher | 616 | 13 | 46,988 | Fletcher, R.B., Das, D., Gadye, L., et al. (2017). Deconstructing Olfactory Stem Cell Trajectories at Single-Cell Resolution. *Cell Stem Cell*, 20(6), 817–830. https://doi.org/10.1016/j.stem.2017.04.003 |
| Romanov | 2,881 | 7 | 24,341 | Romanov, R.A., Zeisel, A., Bakker, J., et al. (2017). Molecular interrogation of hypothalamic organization reveals distinct dopamine neuronal subtypes. *Nature Neuroscience*, 20(2), 176–188. https://doi.org/10.1038/nn.4462 |
| Zeisel | 3,005 | 9 | 19,972 | Zeisel, A., Hochgerner, H., Lönnerberg, P., et al. (2018). Molecular Architecture of the Mouse Nervous System. *Cell*, 174(4), 999–1014. https://doi.org/10.1016/j.cell.2018.06.021 |
| Kolod | 704 | 5 | 38,561 | Kolodziejczyk, A.A., Kim, J.K., Tsang, J.C.H., et al. (2015). Single Cell RNA-Sequencing of Pluripotent States Unlocks Modular Transcriptional Variation. *Cell Stem Cell*, 17(4), 471–485. https://doi.org/10.1016/j.stem.2015.09.011 |
| Deng | 248 | 6 | 22,431 | Deng, Q., Ramsköld, D., Reinius, B., & Sandberg, R. (2014). Single-Cell RNA-Seq Reveals Dynamic, Random Monoallelic Gene Expression in Mammalian Cells. *Science*, 343(6167), 193–196. https://doi.org/10.1126/science.1245316 |
