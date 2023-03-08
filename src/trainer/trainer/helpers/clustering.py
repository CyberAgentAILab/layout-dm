import numpy as np
from einops import rearrange

EPS = 1e-12


class Percentile:
    """
    It resembles KMeans interface in scikit-learn
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    def __init__(self, n_clusters: int = 32, v_min: float = 0.0, v_max: float = 1.0):
        self.n_clusters = n_clusters
        self.v_min = v_min
        self.v_max = v_max

    def fit(self, X: np.ndarray):
        assert X.ndim == 2  # (B, 1)
        X = X[:, 0].clip(self.v_min, self.v_max)
        # n_points_per_bin = int(np.ceil(len(data_sorted) / self.n_clusters))
        # thresholds = data_sorted[::n_points_per_bin]
        # self._thresholds = np.concatenate([thresholds[1:], [self.v_max,]]).reshape(
        #     1, -1
        # )  # (1, T)
        # cond = self._thresholds <= X.reshape(-1, 1)  # (B, T)
        # ids = cond.argmax(axis=1)  # get smallest index of non-zero (false) item

        X = np.sort(np.unique(X))
        thresholds = np.linspace(0.0, 1.0, self.n_clusters + 1)[:-1]
        thresholds = [X[int(t * len(X))] for t in thresholds]
        ids = (thresholds <= X.reshape(-1, 1)).sum(axis=1).astype(np.uint64) - 1

        self.cluster_centers_ = np.full(
            (self.n_clusters, 1), -1.0, dtype=np.float32
        )  # -1 will not be queried
        for i in range(self.n_clusters):
            values = X[ids == i]
            if len(values) > 0:
                self.cluster_centers_[i, 0] = values.mean().astype(np.float32)
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise NotImplementedError

        assert X.ndim == 2
        X = X.clip(self.v_min, self.v_max)
        # cond = self._thresholds >= X.reshape(-1, 1)  # (B, T)
        # ids = cond.argmax(axis=1)
        # https://github.com/jannerm/trajectory-transformer/blob/c77076d1c39e8c8edc3d1e5032b55499de556d73/trajectory/utils/discretization.py#L196-L213
        dist = np.fabs(self.cluster_centers_ - rearrange(X, "s 1 -> 1 s"))
        ids = np.argmin(dist, axis=0)

        return ids
