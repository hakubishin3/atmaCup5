import sys
import time
import pathlib
import numpy as np
import pandas as pd
from base import Feature
from contextlib import contextmanager
from scipy import signal
from tqdm import tqdm

sys.path.append('../')
from src.utils import get_logger
from src.preprocessor import preprocessing
from sklearn.manifold import TSNE


# =========================================
# === Constants
# =========================================
DATA_DIR = pathlib.Path('../data/input/')
FE_DIR = pathlib.Path('../data/features/')
TRAIN_NAME = 'train.csv'
TEST_NAME = 'test.csv'


# =========================================
# === Settings
# =========================================


# =========================================
# === Functions
# =========================================
@contextmanager
def timer(logger, name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


# =========================================
# === Feature Class 
# =========================================
class EmbeddingIntensityFeatures(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        logger = get_logger(self.__class__.__name__)

        with timer(logger, 'loading data'):
            train = pd.read_csv(DATA_DIR / TRAIN_NAME)
            test = pd.read_csv(DATA_DIR / TEST_NAME)
            spectrum_stack = pd.read_csv(DATA_DIR / "spectrum_stack.csv")

        with timer(logger, 'get intensity features'):
            clf = TSNE(n_components=3)
            X = spectrum_stack.drop(columns=["spectrum_filename"]).values
            print(X.shape)
            z = clf.fit_transform(X)

            result = pd.DataFrame(z, columns=[f"tsne_{i}" for i in range(z.shape[1])])
            fe_cols = result.columns
            result["spectrum_filename"] = spectrum_stack["spectrum_filename"].values

            train = pd.merge(train, result, on="spectrum_filename", how="left")
            test = pd.merge(test, result, on="spectrum_filename", how="left")

            self.train_feature = train[fe_cols]
            self.test_feature = test[fe_cols]


if __name__ == "__main__":
    f = EmbeddingIntensityFeatures(path=FE_DIR)
    f.run().save()
