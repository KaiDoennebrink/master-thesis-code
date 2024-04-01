from enum import Enum


class HybridAnnotatorModelSetting(Enum):
    """Enumeration of the hybrid annotator model settings."""

    LABEL_BASED_MODEL = "label"
    SIGNAL_BASED_MODEL = "signal"

