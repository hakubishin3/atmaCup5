import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm


def preprocessing(df, mode="train"):
    data_path = pathlib.Path("/Users/goudashuuhei/Desktop/atmaCup5/data/input/")
    fitting = pd.read_csv(data_path / "fitting.csv")
    df = pd.merge(df, fitting, on="spectrum_id", how="left")
    return df


def make_spectrum_raw_df():
    p_temp = pathlib.Path('./data/input/spectrum_raw')

    spec = []
    for file in p_temp.iterdir():
        spec_df = pd.read_csv(file, sep='\t', header=None)
        spec_df.columns = ["wavelength", "intensity"]
        spec_df["spectrum_filename"] = file.stem + ".dat"
        spec.append(spec_df)

    spec_df = pd.concat(spec, axis=0)
    spec_df.to_csv("./data/input/spectrum.csv", header=True, index=False)


if __name__ == '__main__':
    train = pd.read_csv("./data/input/train.csv")
    print(preprocessing(train, "train").shape)

    test = pd.read_csv("./data/input/test.csv")
    print(preprocessing(test, "test").shape)

    make_spectrum_raw_df()
