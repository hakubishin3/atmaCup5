import sys
import time
import pathlib
import pandas as pd
from base import Feature
from contextlib import contextmanager

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
numeric_features = [
    "layout_x",
    "layout_y",
    "pos_x",
    "params0",
    "params1",
    "params2",
    "params3",
    "params4",
    "params5",
    "params6",
    "rms",
    "beta",
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
class NumericFeatures(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        logger = get_logger(self.__class__.__name__)

        with timer(logger, 'loading data'):
            train = pd.read_csv(DATA_DIR / TRAIN_NAME)
            test = pd.read_csv(DATA_DIR / TEST_NAME)

        with timer(logger, 'preprocessing'):
            train = preprocessing(train)
            test = preprocessing(test)

        with timer(logger, 'get numeric features'):
            for col in numeric_features:
                self.train_feature[col] = train[col]
                self.test_feature[col] = test[col]


if __name__ == "__main__":
    f = NumericFeatures(path=FE_DIR)
    f.run().save()
