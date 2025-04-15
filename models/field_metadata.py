import logging
from typing import Optional, Callable, Dict

from models.enums import DataType
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FieldMetadata:
    def __init__(
            self,
            data_type: DataType,
            # numerical fields
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            decimal_places: Optional[int] = None,

            # string fields
            faker_method: Optional[Callable] = None,
            faker_args: Optional[Dict] = None,

            # NaN handling
            allow_nans: bool = False,
            nan_probability: float = 0.1,

            # datetime fields
            datetime_format: Optional[str] = None,
            datetime_min: Optional[str] = None,
            datetime_max: Optional[str] = None,
    ):
        self.data_type = data_type
        self.min_value = min_value
        self.max_value = max_value
        self.decimal_places = decimal_places
        self.faker_method = faker_method
        self.faker_args = faker_args or {}
        self.allow_nans = allow_nans
        self.nan_probability = min(max(nan_probability, 0), 1)
        self.datetime_format = datetime_format
        self.datetime_type = self._detect_datetime_type(datetime_format)
        self.datetime_min = datetime_min
        self.datetime_max = datetime_max

        self._validate_config()

    def _detect_datetime_type(self, fmt):
        if not fmt:
            return None

        has_date = any(x in fmt for x in ['%d', '%m', '%Y', '%y'])
        has_time = any(x in fmt for x in ['%H', '%M', '%S', '%I', '%p'])

        if has_date and has_time:
            return 'datetime'
        elif has_date:
            return 'date'
        elif has_time:
            return 'time'
        return None

    def _validate_config(self):
        """Validate field configuration"""
        if self.data_type == DataType.DATETIME:
            if not self.datetime_format:
                raise ValueError("datetime_format must be specified for datetime fields")

            # Validate the format string
            try:
                test_date = datetime.now()
                test_date.strftime(self.datetime_format)
            except ValueError as e:
                raise ValueError(f"Invalid datetime format: {str(e)}")

            # Validate min/max if provided
            if self.datetime_min or self.datetime_max:
                try:
                    if self.datetime_min:
                        datetime.strptime(self.datetime_min, self.datetime_format)
                    if self.datetime_max:
                        datetime.strptime(self.datetime_max, self.datetime_format)
                except ValueError as e:
                    logger.warning(f"Invalid datetime min/max value for format {self.datetime_format}: {str(e)}")
                    # Reset min/max if they don't match format
                    self.datetime_min = None
                    self.datetime_max = None