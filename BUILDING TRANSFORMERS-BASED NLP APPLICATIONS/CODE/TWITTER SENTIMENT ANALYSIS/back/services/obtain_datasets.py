from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "datasets" / "training.1600000.processed.noemoticon.csv"

def get_datasets():
    columns = [
        'target',
        'id',
        'date',
        'flag',
        'user',
        'text'
    ]

    df = pd.read_csv(DATASET_PATH, encoding='latin-1', names=columns)

    df["target"] = df["target"].map({
        0: 0,
        4: 1
    })

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["target"]
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["target"]
    )

    return train_df, validation_df, test_df

if __name__ == "__main__":

    train, validation, test = get_datasets()

    print("Train:", train.shape)
    print("Validation:", validation.shape)
    print("Test:", test.shape)

    print("\nTrain labels:")
    print(train["target"].value_counts())

    print("\nValidation labels:")
    print(validation["target"].value_counts())

    print("\nTest labels:")
    print(test["target"].value_counts())