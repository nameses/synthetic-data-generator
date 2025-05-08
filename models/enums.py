"""Python module to define enumerations for various data types."""

from enum import Enum


class DataType(Enum):
    """Enumeration for different data types."""

    INTEGER = "int"
    DECIMAL = "decimal"
    STRING = "string"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
