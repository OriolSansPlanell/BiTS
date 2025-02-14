import pandas as pd

class DataModel:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads CSV data into a Pandas DataFrame."""
        try:
            return pd.read_csv(self.file_path)
        except Exception as e:
            return None

    def get_summary(self, data):
        """Returns summary statistics of the dataset."""
        return data.describe() if data is not None else None
