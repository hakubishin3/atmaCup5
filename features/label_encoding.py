import sys
import time
import pathlib
import pandas as pd
from base import Feature
from contextlib import contextmanager
from sklearn.preprocessing import LabelEncoder

sys.path.append('../')
from src.utils import get_logger


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
categorical_features = [
    # 'chip_id',
    'exc_wl',
    'layout_a',
    "layout_x",
    "layout_y"
]


# =========================================
# === Functions
# =========================================
@contextmanager
def timer(logger, name):
    t0 = time.time()
    yield
    logger.info(f"[{name}] done in {time.time() - t0:.0f} s")


def label_encoding(col, train, test):
    le = LabelEncoder()
    train_label = list(train[col].astype(str).values)
    test_label = list(test[col].astype(str).values)
    total_label = train_label + test_label
    le.fit(total_label)
    train_feature = le.transform(train_label)
    test_feature = le.transform(test_label)

    return train_feature, test_feature


# =========================================
# === Feature Class 
# =========================================
class LabelEncoding(Feature):
    def categorical_features(self):
        return categorical_features

    def create_features(self):
        logger = get_logger(__name__)

        with timer(logger, 'loading data'):
            train = pd.read_csv(DATA_DIR / TRAIN_NAME)
            test = pd.read_csv(DATA_DIR / TEST_NAME)

        with timer(logger, 'label encoding'):
            for col in categorical_features:
                train_result, test_result = label_encoding(col, train, test)
                self.train_feature[col] = train_result
                self.test_feature[col] = test_result


if __name__ == "__main__":
    f = LabelEncoding(path=FE_DIR)
    f.run().save()
