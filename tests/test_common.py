"""Unit tests for common utilities, transformers, and data schema."""

import re
import unittest
import random
import numpy as np
import torch
import pandas as pd


from common.utilities import set_seed
from common.transformers import ContTf, DtTf
from common.dataclasses import DataSchema
from models.field_metadata import FieldMetadata
from models.enums import DataType


class CommonTests(unittest.TestCase):
    """Unit tests for common utilities, transformers, and data schema."""

    def test_set_seed_affects_all_rngs(self):
        """Test that seeding reproduces outputs in random, numpy, and torch."""
        set_seed(2025)
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        set_seed(2025)
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        self.assertEqual(r1, r2)
        self.assertAlmostEqual(n1, n2)
        self.assertAlmostEqual(t1, t2)

    def test_cont_tf_transform_inverse_properties(self):
        """Test that ContTf transform/inverse preserve shape and range."""
        data = pd.Series(np.linspace(-1, 1, 100))
        tf = ContTf().fit(data)
        transformed = tf.transform(data)
        # transformed should be finite numeric
        self.assertTrue(np.isfinite(transformed).all())
        inverted = tf.inverse(transformed)
        # shape preserved
        self.assertEqual(len(inverted), len(data))
        # values within clipped original range
        q_low, q_hi = data.quantile([0.0025, 0.9975])
        self.assertTrue(((inverted >= q_low) & (inverted <= q_hi)).all())

    def test_dt_tf_inverse_format(self):
        """Test that DtTf inverse outputs strings matching the expected datetime format."""
        fmt = "%Y/%m/%d %H:%M"
        dates = pd.Series(["2025/05/01 00:00", "2025/05/09 23:59", "2025/12/31 12:34"])
        tf = DtTf(fmt).fit(dates)
        transformed = tf.transform(dates)
        self.assertIsInstance(transformed, np.ndarray)
        inverted = tf.inverse(transformed)
        # Check format via regex
        pattern = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$")
        for s in inverted:
            self.assertRegex(s, pattern)

    def test_data_schema_from_dataframe_filters_and_classifies(self):
        """Test that DataSchema.from_dataframe cleans invalid rows and classifies columns."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, None, 4],
                "dec_col": [1.1, None, 3.3, 4.4],
                "dt_col": ["2025-01-01", "2025-01-02", "", "2025-01-04"],
                "cat_col": ["a", "b", "a", None],
                "str_col": ["x", None, "z", "w"],
                "bool_col": [True, False, True, False],
            }
        )
        meta = {
            "int_col": FieldMetadata(data_type=DataType.INTEGER),
            "dec_col": FieldMetadata(data_type=DataType.DECIMAL, decimal_places=1),
            "dt_col": FieldMetadata(
                data_type=DataType.DATETIME, datetime_format="%Y-%m-%d"
            ),
            "cat_col": FieldMetadata(data_type=DataType.CATEGORICAL),
            "str_col": FieldMetadata(data_type=DataType.STRING),
            "bool_col": FieldMetadata(data_type=DataType.BOOLEAN),
        }
        schema = DataSchema.from_dataframe(df, meta)
        # After dropping invalid rows, only one valid remains
        self.assertEqual(len(schema.real_df), 1)
        # Numeric columns
        self.assertListEqual(schema.num_cols, ["int_col", "dec_col"])
        # Datetime columns
        self.assertListEqual(schema.dt_cols, ["dt_col"])
        # Categorical includes both categorical and boolean
        self.assertSetEqual(set(schema.cat_cols), {"cat_col", "bool_col"})
        # String columns
        self.assertListEqual(schema.str_cols, ["str_col"])


if __name__ == "__main__":
    unittest.main()
