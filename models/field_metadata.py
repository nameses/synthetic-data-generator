from typing import Dict, Any, Optional, Callable, List
from faker import Faker

from models.enums import DataType


class FieldMetadata:
    def __init__(
            self,
            data_type: DataType,
            *,
            # For numerical fields
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,

            # For string fields
            fake_strategy: Optional[str] = None,
            custom_faker: Optional[Callable] = None,
            string_format: Optional[str] = None
    ):
        self.data_type = data_type
        self.min_value = min_value
        self.max_value = max_value
        self.fake_strategy = fake_strategy
        self.custom_faker = custom_faker
        self.string_format = string_format

        self._validate_config()

    def _validate_config(self):
        """Validate field configuration"""
        if self.data_type in [DataType.INTEGER, DataType.DECIMAL]:
            if self.min_value is not None and self.max_value is not None:
                if self.min_value > self.max_value:
                    raise ValueError(
                        f"min_value ({self.min_value}) cannot be greater than max_value ({self.max_value})")

        if self.data_type == DataType.STRING:
            if self.custom_faker and (self.fake_strategy or self.string_format):
                raise ValueError("Cannot specify both custom_faker and fake_strategy/string_format")

            if self.fake_strategy:
                fake = Faker()
                if self.fake_strategy not in dir(fake):
                    raise ValueError(f"Invalid fake strategy: {self.fake_strategy}")