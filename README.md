# Graph Label Alignment

Given graph classification dataset $(\mathcal{G},y_\mathcal{G})\in\mathcal{D}$ our process is straightforward:

1. Compute Structural summary vector $S(\mathcal{G})$ which contains topological, spectral, and statistical information about $\mathcal{G}$
2. (Optionally) Compute Feature summary vector $F(\mathcal{G})$ which contains class prototype distance and centroid of the point cloud
3. Feed both $S(\mathcal{G})$ and $F(\mathcal{G})$ into the following basic models

Logistic Regression and Gaussian Kernel SVM

over 5-seeded 10-fold CV set the respective Label Alignment score to be the average of the two models.


From `summarize.py`
```py
from summarize import StructSummary, FeatSummary

ds = # load PyG dataset here

# Sanity check dataset
y = np.array([int(ds[i].y.item()) for i in range(len(ds))])

Xs, _ = StructSummary(ds)
Xf, _ = FeatSummary(ds)

# Train LogReg and RBF-SVM on (Xs, y) and (Xf, y)
```
