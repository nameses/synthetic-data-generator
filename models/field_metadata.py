from typing import Dict, Any, Optional, Callable
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

        # Numerical fields
        if data_type in [DataType.INTEGER, DataType.DECIMAL]:
            self.min_value = min_value
            self.max_value = max_value

        # String fields
        elif data_type == DataType.STRING:
            self.fake_strategy = fake_strategy
            self.custom_faker = custom_faker
            self.string_format = string_format
            self._validate_string_config()

    def _validate_string_config(self):
        if self.custom_faker and (self.fake_strategy or self.string_format):
            raise ValueError("Cannot specify both custom_faker and fake_strategy/string_format")

        if self.fake_strategy and self.fake_strategy not in dir(Faker()):
            raise ValueError(f"Invalid fake strategy: {self.fake_strategy}")

    def get_string_generator(self) -> Callable:
        """Returns a function that generates synthetic strings"""
        fake = Faker()

        if self.custom_faker:
            return self.custom_faker

        if self.fake_strategy:
            if self.string_format:
                return lambda: getattr(fake, self.fake_strategy)(self.string_format)
            return getattr(fake, self.fake_strategy)

        # Default string generation
        return fake.text