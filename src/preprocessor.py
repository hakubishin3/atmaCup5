import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocessing_train_test(df, mode="train"):
    data_path = pathlib.Path.cwd() / "data/input/"
    fitting = pd.read_csv(data_path / "fitting.csv")
    df = pd.merge(df, fitting, on="spectrum_id", how="left")
    return df


if __name__ == '__main__':
    train = pd.read_csv("./data/input/train.csv")
    print(preprocessing_train_test(train, "train").shape)

    test = pd.read_csv("./data/input/test.csv")
    print(preprocessing_train_test(test, "test").shape)
