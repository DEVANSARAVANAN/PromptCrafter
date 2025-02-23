import pandas as pd

def load_dataset(file_path=None):
    if file_path:
        if file_path.name.endswith('.csv'):
            dataset = pd.read_csv(file_path)
        elif file_path.name.endswith('.xlsx'):
            dataset = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    else:
        dataset = pd.read_csv('post_classification_dataset - post_classification_dataset (1).csv')
    return dataset