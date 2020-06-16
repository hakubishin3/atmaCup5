import sys
import time
import pathlib
import numpy as np
import pandas as pd
from base import Feature
from contextlib import contextmanager
from scipy.stats import skew, kurtosis
from scipy import signal
from tqdm import tqdm

sys.path.append('../')
from src.utils import get_logger
from src.preprocessor import preprocessing


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
STATS = [
    np.max,
]

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
class IntensityDiffFeatures(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        logger = get_logger(self.__class__.__name__)

        with timer(logger, 'loading data'):
            train = pd.read_csv(DATA_DIR / TRAIN_NAME)
            test = pd.read_csv(DATA_DIR / TEST_NAME)
            spectrum = pd.read_csv(DATA_DIR / "spectrum.csv")

        with timer(logger, 'get intensity features'):
            spectrum["intensity_diff"] = np.abs(spectrum.groupby(["spectrum_filename"])["intensity"].diff().values)
            agg = spectrum.groupby("spectrum_filename")["intensity_diff"].agg(STATS)
            agg_cols = agg.columns
            train = pd.merge(train, agg.reset_index(), on="spectrum_filename", how="left")
            test = pd.merge(test, agg.reset_index(), on="spectrum_filename", how="left")

            self.train_feature = train[agg_cols]
            self.test_feature = test[agg_cols]


if __name__ == "__main__":
    f = IntensityDiffFeatures(path=FE_DIR)
    f.run().save()
