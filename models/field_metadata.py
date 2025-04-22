from typing import Optional, Callable, Dict, Literal, List, Union

from models.enums import DataType


class FieldMetadata:
    def __init__(
            self,
            data_type: DataType,
            # numerical fields
            decimal_places: Optional[int] = None,
            # range constraints (autofilled)
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,

            # categorical/boolean fields
            #  - "uniform" → np.random.randint (old behavior)
            #  - "empirical" → sample according to real data frequencies
            sampling: Literal["uniform", "empirical"] = "uniform",

            # string fields
            faker_method: Optional[Callable] = None,
            faker_args: Optional[Dict] = None,

            # datetime fields
            datetime_format: Optional[str] = None,

            # transformers for numeric data
            # - 'standard' → standardize using mean and std
            # - 'minmax' → scale to [-1, 1] using min and max
            # - 'log' → log(1+x) transform to handle skewed data
            # - 'yeo-johnson' → power transform for skewed data
            # - ['log', 'minmax'] → chain of transformations (applied left to right)
            transformer: Optional[Union[str, List[str]]] = None,
            
            # conditional generation - column name that this field is conditionally dependent on
            conditional_on: Optional[str] = None,
            
            # clamping outliers with percentiles or absolute values
            clamp_min: Optional[float] = None,  # either percentile (if < 1) or absolute value
            clamp_max: Optional[float] = None,  # either percentile (if < 1) or absolute value
            
            # preserve exact zeros (important for delays, etc.)
            preserve_zeros: bool = False,
    ):
        self.data_type = data_type
        self.decimal_places = decimal_places

        self.sampling = sampling

        self.faker_method = faker_method
        self.faker_args = faker_args or {}

        self.datetime_format = datetime_format
        
        # Ensure transformer is either None, a string, or a list of strings
        if isinstance(transformer, str):
            self.transformer = [transformer]
        else:
            self.transformer = transformer
            
        self.conditional_on = conditional_on
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.preserve_zeros = preserve_zeros
