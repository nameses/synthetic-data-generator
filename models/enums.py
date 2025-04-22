from enum import Enum


class DataType(Enum):
    INTEGER = "int"
    DECIMAL = "decimal"
    STRING = "string"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"