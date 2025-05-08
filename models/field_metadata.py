"""Field metadata for data generation."""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

from models.enums import DataType


@dataclass(slots=True)
class FieldMetadata:
    """Metadata for a field in the dataset."""

    data_type: DataType
    """The type of data (e.g., Integer, Float, String, DateTime)."""

    decimal_places: Optional[int] = None
    """For numeric fields: number of decimal places (if Float)."""

    faker_method: Optional[Callable] = None
    """For string fields: Faker method to generate data."""

    faker_args: Dict = field(default_factory=dict)
    """Arguments to pass to the Faker method."""

    datetime_format: Optional[str] = None
    """For DateTime fields: Python datetime format string."""
