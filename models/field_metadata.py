from typing import Optional, Callable, Dict

from models.enums import DataType


class FieldMetadata:
    def __init__(
            self,
            data_type: DataType,
            # numerical fields
            decimal_places: Optional[int] = None,

            # string fields
            faker_method: Optional[Callable] = None,
            faker_args: Optional[Dict] = None,

            # datetime fields
            datetime_format: Optional[str] = None,
    ):
        self.data_type = data_type
        self.decimal_places = decimal_places

        self.faker_method = faker_method
        self.faker_args = faker_args or {}

        self.datetime_format = datetime_format
