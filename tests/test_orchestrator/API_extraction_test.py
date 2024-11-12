import unittest
import pandas as pd
import numpy as np
from nibelungenbruecke.scripts.utilities.API_sensor_retrieval import API_Request
import os

class TestAPIRequest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up paths and expected results
        cls.csv_path = 'E+080DU_HSN-u-_d10-data-14 May 2024, 17 32 45.csv'
        cls.expected_columns = ["E_plus_080DU_HSN-u-_d10"]

        # Load CSV data for comparison
        if os.path.exists(cls.csv_path):
            cls.DU_data = pd.read_csv(cls.csv_path)
        else:
            cls.skipTest(cls, "CSV file not found: Ensure the test file is in the correct directory.")

        # Initialize API
        cls.api = API_Request()

    def test_fetch_data_from_api(self):
        """Test if API fetches the correct data columns."""
        self.api.body = {
            "startTime": "2024-05-01T08:00:00Z",
            "endTime": "2024-05-01T20:00:00Z",
            "meta_channel": True,
            "columns": self.expected_columns
        }

        output_data = self.api.fetch_data()
        DU_data_api = output_data[self.expected_columns[0]]

        # Check that API data is not empty and matches expected columns
        self.assertIn(self.expected_columns[0], output_data)
        self.assertFalse(DU_data_api.empty, "API data should not be empty.")

    def test_reshape_and_average(self):
        """Test reshaping and averaging of the API data."""
        self.api.body = {
            "startTime": "2024-05-01T08:00:00Z",
            "endTime": "2024-05-14T20:00:00Z",
            "meta_channel": True,
            "columns": self.expected_columns
        }

        output_data = self.api.fetch_data()
        DU_data_array = np.array(output_data[self.expected_columns[0]])

        # Reshape and average calculations
        reshaped_data = DU_data_array.reshape(-1, 10)
        averaged_data = reshaped_data.mean(axis=1)
        averaged_data_list = averaged_data.tolist()

        # Verify shape and type of data
        self.assertEqual(reshaped_data.shape[1], 10, "Reshaped data should have 10 columns.")
        self.assertIsInstance(averaged_data_list, list, "Averaged data should be a list.")

    def test_invalid_columns(self):
        """Test API with invalid columns."""
        self.api.body = {
            "startTime": "2024-05-01T08:00:00Z",
            "endTime": "2024-05-01T20:00:00Z",
            "meta_channel": True,
            "columns": ["Invalid_Column"]
        }

        output_data = self.api.fetch_data()

        # Expect an empty DataFrame for invalid columns
        self.assertTrue(output_data.empty, "Output data should be empty for invalid columns.")

    @classmethod
    def tearDownClass(cls):
        pass  # Cleanup actions if necessary

if __name__ == "__main__":
    unittest.main()
