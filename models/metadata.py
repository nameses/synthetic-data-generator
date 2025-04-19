from faker import Faker

from models.enums import DataType
from models.field_metadata import FieldMetadata

metadata_airline = {
    'Gender': FieldMetadata(DataType.CATEGORICAL, sampling='empirical'),
    'Customer Type': FieldMetadata(DataType.CATEGORICAL, sampling='empirical'),
    'Age': FieldMetadata(DataType.INTEGER),
    'Type of Travel': FieldMetadata(DataType.CATEGORICAL, sampling='empirical'),
    'Class': FieldMetadata(DataType.CATEGORICAL, sampling='empirical'),
    'Flight Distance': FieldMetadata(DataType.INTEGER),
    'Inflight wifi service': FieldMetadata(DataType.INTEGER),
    'Departure or Arrival time convenient': FieldMetadata(DataType.INTEGER),
    'Ease of Online booking': FieldMetadata(DataType.INTEGER),
    'Gate location': FieldMetadata(DataType.INTEGER),
    'Food and drink': FieldMetadata(DataType.INTEGER),
    'Online boarding': FieldMetadata(DataType.INTEGER),
    'Seat comfort': FieldMetadata(DataType.INTEGER),
    'Inflight entertainment': FieldMetadata(DataType.INTEGER),
    'On-board service': FieldMetadata(DataType.INTEGER),
    'Leg room service': FieldMetadata(DataType.INTEGER),
    'Baggage handling': FieldMetadata(DataType.INTEGER),
    'Checkin service': FieldMetadata(DataType.INTEGER),
    'Inflight service': FieldMetadata(DataType.INTEGER),
    'Cleanliness': FieldMetadata(DataType.INTEGER),
    'Departure Delay in Minutes': FieldMetadata(DataType.INTEGER),
    'Arrival Delay in Minutes': FieldMetadata(DataType.DECIMAL, decimal_places=1),
    'satisfaction': FieldMetadata(DataType.CATEGORICAL, sampling='empirical'),
}

metadata_helthcare = {
    'Name': FieldMetadata(DataType.STRING, faker_method=Faker().name),
    'Age': FieldMetadata(DataType.INTEGER),
    'Gender': FieldMetadata(DataType.CATEGORICAL),
    'Blood Type': FieldMetadata(DataType.CATEGORICAL),
    'Medical Condition': FieldMetadata(DataType.CATEGORICAL),
    'Date of Admission': FieldMetadata(DataType.DATETIME, datetime_format='%Y-%m-%d'),
    'Doctor': FieldMetadata(DataType.STRING, faker_method=Faker().name),
    'Hospital': FieldMetadata(DataType.STRING, faker_method=Faker().company),
    'Insurance Provider': FieldMetadata(DataType.CATEGORICAL),
    'Billing Amount': FieldMetadata(DataType.DECIMAL, decimal_places=12),
    'Room Number': FieldMetadata(DataType.INTEGER),
    'Admission Type': FieldMetadata(DataType.CATEGORICAL),
    'Discharge Date': FieldMetadata(DataType.DATETIME, datetime_format='%Y-%m-%d'),
    'Medication': FieldMetadata(DataType.CATEGORICAL),
    'Test Results': FieldMetadata(DataType.CATEGORICAL),
}

metadata_adult = {
    'age': FieldMetadata(DataType.INTEGER),
    'workclass': FieldMetadata(DataType.CATEGORICAL),
    'fnlwgt': FieldMetadata(DataType.INTEGER),
    'education': FieldMetadata(DataType.CATEGORICAL),
    'education.num': FieldMetadata(DataType.INTEGER),
    'marital.status': FieldMetadata(DataType.CATEGORICAL),
    'occupation': FieldMetadata(DataType.CATEGORICAL),
    'relationship': FieldMetadata(DataType.CATEGORICAL),
    'race': FieldMetadata(DataType.CATEGORICAL),
    'sex': FieldMetadata(DataType.CATEGORICAL),
    'capital.gain': FieldMetadata(DataType.INTEGER),
    'capital.loss': FieldMetadata(DataType.INTEGER),
    'hours.per.week': FieldMetadata(DataType.INTEGER),
    'native.country': FieldMetadata(DataType.CATEGORICAL),
    'income': FieldMetadata(DataType.CATEGORICAL)
}