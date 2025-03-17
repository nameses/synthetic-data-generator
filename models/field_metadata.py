from models.enums import DataType


class FieldMetadata:
    def __init__(self, data_type: DataType, faker_func=None):
        self.data_type = data_type
        if data_type == DataType.STRING:
            self.faker_func = faker_func
