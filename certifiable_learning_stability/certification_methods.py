from enum import IntEnum


class CertificationMethod(IntEnum):
    """
    Enum representing different certification methods for certifiable learning stability.
    """

    SGD = 0
    HYBRID_RDP = 1
    POINTWISE_RDP = 2
    AGT = 3

    def __str__(self):
        return self.name.lower()


class RobustnessSetup(IntEnum):
    """
    Enum representing different robustness setups for certifiable learning stability.
    """

    LOW = 0
    MEDIUM = 1
    HIGH = 2


class AggregationType(IntEnum):
    """
    Enum representing different aggregation types for certifiable learning stability. Options include:
    - DPA: Deep Partition Aggregation
    - ROE: Run-Off Elections
    """

    DPA = 0
    ROE = 1

    def __str__(self):
        match self.value:
            case 0:
                return "dpa"
            case 1:
                return "roe"
            case _:
                raise ValueError(f"Unknown AggregationType value: {self.value}")
