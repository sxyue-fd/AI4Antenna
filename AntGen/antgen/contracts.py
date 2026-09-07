"""Shared output and electromagnetic-response contracts for AntGen."""

SCHEMA_VERSION = 3
SPATIAL_COORDINATE_CONVENTION = "physical_row_col_v1"
PATTERN_CHANNEL_ORDER = ("XOZ_Gtheta", "XOZ_Gphi", "YOZ_Gtheta", "YOZ_Gphi")
PATTERN_CHANNEL_ORDER_CSV = ",".join(PATTERN_CHANNEL_ORDER)
PATTERN_VALUE_TYPE = "linear_gain"
PATTERN_PLANES_PHI_DEG = (0.0, 90.0)
