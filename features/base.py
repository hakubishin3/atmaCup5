import pathlib
import feather
import pandas as pd
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, List, Tuple


class Feature(metaclass=ABCMeta):
    def __init__(self, path='.', prefix='', suffix=''):
        self.path = pathlib.Path(path)
        self.prefix = prefix
        self.suffix = suffix
        self.name = self.__class__.__name__
        self.train_feature = pd.DataFrame()
        self.test_feature = pd.DataFrame()

    def run(self):
        self.create_features()
        prefix = self.prefix + '_' if self.prefix else self.name + '_'
        suffix = '_' + self.suffix if self.suffix else ''
        self.train_feature.columns = prefix + self.train_feature.columns + suffix
        self.test_feature.columns = prefix + self.test_feature.columns + suffix
        return self

    @abstractmethod
    def create_features(self, train: pd.DataFrame, test: pd.DataFrame) -> None:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features() -> List[str]:
        raise NotImplementedError

    def save(self) -> None:
        self.train_feature.reset_index(drop=True).to_feather(self.path / f'{self.name}_train.ftr')
        self.test_feature.reset_index(drop=True).to_feather(self.path / f'{self.name}_test.ftr')


def load_features(config: dict, debug_mode=False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    feathre_path = pathlib.Path(config['dataset']['feature_directory'])

    train_dfs = [feather.read_dataframe(feathre_path / f'{fe}_train.ftr') for fe in config['features']]
    x_train = pd.concat(train_dfs, axis=1)

    if debug_mode:
        return x_train, None
    else:
        test_dfs = [feather.read_dataframe(feathre_path / f'{fe}_test.ftr') for fe in config['features']]
        x_test = pd.concat(test_dfs, axis=1)
        return x_train, x_test
