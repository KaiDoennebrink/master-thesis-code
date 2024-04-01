from enum import Enum


class SelectionStrategy(Enum):
    PLVI_CE_KNN = "plvi_ce_knn"
    PLVI_CE_TOPK = "plvi_ce_topk"
    RANDOM = "random"
    ENTROPY = "entropy"
    BADGE = "badge"
