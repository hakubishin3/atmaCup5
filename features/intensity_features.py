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
from scipy.interpolate import UnivariateSpline

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
class IntensityFeatures(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        logger = get_logger(self.__class__.__name__)

        with timer(logger, 'loading data'):
            train = pd.read_csv(DATA_DIR / TRAIN_NAME)
            test = pd.read_csv(DATA_DIR / TEST_NAME)
            spectrum = pd.read_csv(DATA_DIR / "spectrum.csv")

        with timer(logger, 'get intensity features'):
            agg = spectrum.groupby("spectrum_filename")["intensity"].agg(STATS)
            agg_cols = agg.columns
            train = pd.merge(train, agg.reset_index(), on="spectrum_filename", how="left")
            test = pd.merge(test, agg.reset_index(), on="spectrum_filename", how="left")

            self.train_feature = train[agg_cols]
            self.test_feature = test[agg_cols]

        with timer(logger, "get fwhm"):
            spectrum_half = (
                (
                    spectrum.groupby(["spectrum_filename"])["intensity"].max() -
                    spectrum.groupby(["spectrum_filename"])["intensity"].min()
                ) / 2
            )
            spectrum = pd.merge(
                spectrum,
                spectrum_half.rename("half_intensity").reset_index(),
                on="spectrum_filename",
                how="left"
            )

            ratios = [1, 1.5]
            for ratio in ratios:
                col_name = f"fwhm_{ratio}"
                spectrum["intensity_sub_half"] = spectrum["intensity"] - spectrum["half_intensity"] * ratio
                spectrum.loc[spectrum["intensity_sub_half"] < 0, "intensity_sub_half"] = np.nan
                spec_fwhm = spectrum.groupby(["spectrum_filename"])["intensity_sub_half"].apply(lambda x: x.notnull().sum()).rename(col_name)
                train = pd.merge(train, spec_fwhm.reset_index(), on="spectrum_filename", how="left")
                test = pd.merge(test, spec_fwhm.reset_index(), on="spectrum_filename", how="left")
                self.train_feature[col_name] = train[col_name]
                self.test_feature[col_name] = test[col_name]

            self.train_feature["fwhm_diff_1_1.5"] = self.train_feature["fwhm_1"] - self.train_feature["fwhm_1.5"]
            self.test_feature["fwhm_diff_1_1.5"] = self.test_feature["fwhm_1"] - self.test_feature["fwhm_1.5"]

        with timer(logger, "get fwhm v2"):
            spectrum["intensity_sub_half"] = spectrum["intensity"] - spectrum["half_intensity"]

            spectrum_filenames = spectrum["spectrum_filename"].unique()
            result = []
            for spectrum_filename in tqdm(spectrum_filenames):
                tmp = spectrum[spectrum["spectrum_filename"] == spectrum_filename]
                spline = UnivariateSpline(tmp["wavelength"], tmp["intensity_sub_half"], s=0)
                roots = spline.roots()
                dist_list = []
                for j in range(len(roots)//2):
                    left = roots[j*2]
                    right = roots[j*2+1]
                    dist = right - left
                    dist_list.append(dist)

                dists = np.array(dist_list)
                n_peak = len(dists)
                if n_peak > 0:
                    min_dist = np.min(dists)
                    max_dist = np.max(dists)
                    mean_dist = np.mean(dists)
                    median_dist = np.median(dists)
                    max_min_dist = max_dist - min_dist
                    mean_min_dist = mean_dist - min_dist
                else:
                    min_dist = np.nan
                    max_dist = np.nan
                    mean_dist = np.nan
                    median_dist = np.nan
                    max_min_dist = np.nan
                    mean_min_dist = np.nan

                result.append((
                    spectrum_filename,
                    n_peak, min_dist, max_dist,
                    mean_dist, median_dist, max_min_dist,
                    mean_min_dist
                ))

            result = pd.DataFrame(result, columns=[
                "spectrum_filename",
                "n_peak", "min_dist", "max_dist",
                "mean_dist", "median_dist", "max_min_dist",
                "mean_min_dist"
            ])
            train = pd.merge(train, result, on="spectrum_filename", how="left")
            test = pd.merge(test, result, on="spectrum_filename", how="left")

            for col in result.columns:
                self.train_feature[col] = train[col]
                self.test_feature[col] = test[col]


if __name__ == "__main__":
    f = IntensityFeatures(path=FE_DIR)
    f.run().save()
