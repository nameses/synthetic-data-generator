"""Unit tests for utilities, transformers, and data schema"""

from __future__ import annotations

import random
import re
import unittest
from typing import Final

import numpy as np
import pandas as pd
import torch

from common.dataclasses import DataSchema
from common.transformers import ContTf, DtTf
from common.utilities import set_seed
from models.enums import DataType
from models.field_metadata import FieldMetadata


class TestCommon(unittest.TestCase):
    """Collection of unit tests that cover lightweight, shared helpers."""

    RNG_SEED: Final = 2025

    def test_set_seed_affects_all_rngs(self) -> None:
        """Calling set_seed must synchronize all key RNGs"""
        set_seed(self.RNG_SEED)
        random_val_1 = random.random()
        numpy_val_1 = np.random.rand()
        torch_val_1 = torch.rand(1).item()

        set_seed(self.RNG_SEED)
        random_val_2 = random.random()
        numpy_val_2 = np.random.rand()
        torch_val_2 = torch.rand(1).item()

        self.assertEqual(random_val_1, random_val_2)
        self.assertAlmostEqual(numpy_val_1, numpy_val_2)
        self.assertAlmostEqual(torch_val_1, torch_val_2)

    def test_cont_tf_round_trip(self) -> None:
        """ContTf must be a near-lossless bijection (up to tail clipping)"""
        data = pd.Series(np.linspace(-1, 1, 100))
        transformer = ContTf().fit(data)

        transformed = transformer.transform(data)
        self.assertTrue(
            np.isfinite(transformed).all(),
            msg="Forward transform produced non-finite values.",
        )

        recovered = transformer.inverse(transformed)
        self.assertEqual(
            len(recovered),
            len(data),
            msg="Inverse transform changed the sample size.",
        )

        low_q, high_q = data.quantile([0.0025, 0.9975])
        within_bounds = (recovered >= low_q) & (recovered <= high_q)
        self.assertTrue(
            within_bounds.all(),
            msg="Recovered values exceed expected clipping range.",
        )

    def test_dt_tf_inverse_format(self) -> None:
        """DtTf.inverse should return strings in the original format"""
        dt_format = "%Y/%m/%d %H:%M"
        dates = pd.Series(["2025/05/01 00:00", "2025/05/09 23:59", "2025/12/31 12:34"])

        transformer = DtTf(dt_format).fit(dates)
        transformed = transformer.transform(dates)
        self.assertIsInstance(transformed, np.ndarray)

        recovered = transformer.inverse(transformed)
        regex = re.compile(r"^\d{4}/\d{2}/\d{2} \d{2}:\d{2}$")

        for item in recovered:
            self.assertRegex(item, regex)

    def test_data_schema_from_dataframe(self) -> None:
        """DataSchema.from_dataframe should clean and classify columns"""
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
                data_type=DataType.DATETIME,
                datetime_format="%Y-%m-%d",
            ),
            "cat_col": FieldMetadata(data_type=DataType.CATEGORICAL),
            "str_col": FieldMetadata(data_type=DataType.STRING),
            "bool_col": FieldMetadata(data_type=DataType.BOOLEAN),
        }

        schema = DataSchema.from_dataframe(df, meta)

        # After dropna only the first row remains
        self.assertEqual(len(schema.real_df), 1)

        # Column classification
        self.assertListEqual(schema.num_cols, ["int_col", "dec_col"])
        self.assertListEqual(schema.dt_cols, ["dt_col"])
        self.assertSetEqual(set(schema.cat_cols), {"cat_col", "bool_col"})
        self.assertListEqual(schema.str_cols, ["str_col"])


if __name__ == "__main__":
    unittest.main()
