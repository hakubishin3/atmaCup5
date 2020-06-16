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


def make_spectrum_raw_df_stack(train, test):

    import os
    def read_spectrum(name):
        path = os.path.join('./data/input/spectrum_raw/', name)

        if not os.path.exists(path):
            raise ValueError(f'{name} is not found at spectrum raw dir {path}')

        return pd.read_csv(path, header=None, sep='\t')

    total = train.append(test).reset_index(drop=True)
    data = total['spectrum_filename'].map(read_spectrum)

    wave_df = pd.DataFrame()

    for k, v in data.items():
        value = v.values[:, 1]
        if len(value) != 512:
            value = np.hstack([value, np.array(np.mean(value))])

        wave_df[k] = value

    wave_df = wave_df.T
    wave_df.index = total['spectrum_filename']
    wave_df.columns = [f"wavelength_{i}" for i in wave_df.columns]
    wave_df = wave_df.reset_index()
    wave_df.to_csv("./data/input/spectrum_stack.csv", header=True, index=False)


if __name__ == '__main__':
    train = pd.read_csv("./data/input/train.csv")
    print(preprocessing(train, "train").shape)

    test = pd.read_csv("./data/input/test.csv")
    print(preprocessing(test, "test").shape)

    make_spectrum_raw_df_stack(train, test)
