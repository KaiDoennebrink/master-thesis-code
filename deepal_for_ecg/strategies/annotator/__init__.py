from enum import Enum


class AnnotatorSetting(Enum):
    """Enumeration of annotator settings."""

    FULL_HUMAN = "full_human"
    HYBRID_LABEL = "hybrid_label"
    FULL_WSA = "full_wsa"
