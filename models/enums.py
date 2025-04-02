from enum import Enum


class DataType(Enum):
    INTEGER = "int"
    DECIMAL = "decimal"
    STRING = "string"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATE_TIME = "time"


class MethodType(Enum):
    GAN = "gan"
    VAE = "vae"
    WGAN = "wgan"  # Added WGAN as the preferred method