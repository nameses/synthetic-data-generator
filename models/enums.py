from enum import Enum


class DataType(Enum):
    INTEGER = "int"
    DECIMAL = "decimal"
    STRING = "string"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"


class MethodType(Enum):
    GAN = "gan"
    VAE = "vae"
    WGAN = "wgan"