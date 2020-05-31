import json
import pathlib
import argparse
import pandas as pd
import numpy as np
from src.get_folds import Fold
from src.runner import Runner
from src.utils import get_logger, json_dump, seed_everything
from src.submission import create_submission
from features.base import load_features
from models.model_1d_cnn import Model_1DCNN
from multiprocessing import cpu_count
from tensorflow import keras

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


seed_everything(71)


model_map = {
    '1dcnn': Model_1DCNN
}


def main():
    # =========================================
    # === Settings
    # =========================================
    # Get logger
    logger = get_logger(__name__)
    logger.info('Settings')

    # Get argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='./configs/model_1dcnn_0.json')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logger.info(f'config: {args.config}')
    logger.info(f'debug: {args.debug}')

    # Get config
    config = json.load(open(args.config))
    config.update({
        'args': {
            'config': args.config,
            'debug': args.debug
        }
    })

    if config["model"]["name"] == "lightgbm":
        config["model"]["model_params"]["nthread"] = cpu_count()

    # Create a directory for model output
    model_no = pathlib.Path(args.config).stem
    model_output_dir = (
        pathlib.Path(config['dataset']['output_directory']) / model_no
    )
    if not model_output_dir.exists():
        model_output_dir.mkdir()

    logger.info(f'model_output_dir: {str(model_output_dir)}')
    logger.debug(f'model_output_dir exists: {model_output_dir.exists()}')
    config.update({
        'model_output_dir': str(model_output_dir)
    })

    # =========================================
    # === Loading data
    # =========================================
    logger.info('Loading data')

    # Get train and test
    input_dir = pathlib.Path(config['dataset']['input_directory'])
    train = pd.read_csv(input_dir / 'train.csv')
    test = pd.read_csv(input_dir / 'test.csv')
    spectrum = pd.read_csv(input_dir / 'spectrum_stack.csv')
    wv_cols = [c for c in spectrum.columns if c.find("wavelength_") != -1]

    spectrum[wv_cols] = spectrum[wv_cols].values / np.std(spectrum[wv_cols].values, axis=1, keepdims=True)
    train_spectrum = pd.merge(train, spectrum, on="spectrum_filename", how="left")[wv_cols]
    test_spectrum = pd.merge(test, spectrum, on="spectrum_filename", how="left")[wv_cols]

    # Get target values
    target_column = config['data_type']['target']
    y_train = train[target_column].values

    # =========================================
    # === Loading features
    # =========================================
    logger.info('Loading features')

    # Get features
    x_train, x_test = load_features(config)
    feature_name = x_test.columns
    logger.debug(f'number of features: {len(feature_name)}')

    # =========================================
    # === features preprocess
    # =========================================
    x_total = x_train.append(x_test).reset_index(drop=True)
    remove_features = [c for c in x_total.columns if c.find("layout_x") != -1]
    remove_features += [c for c in x_total.columns if c.find("layout_y") != -1]
    x_total.drop(columns=remove_features, inplace=True)

    x_total = pd.get_dummies(x_total, columns=["LabelEncoding_exc_wl", "LabelEncoding_layout_a"])
    x_total.fillna(0, inplace=True)

    from sklearn.preprocessing import StandardScaler
    numeric_features = [c for c in x_total.columns if c.find("LabelEncoding_") == -1]
    sc = StandardScaler()
    x_total[numeric_features] = sc.fit_transform(x_total[numeric_features])

    x_train = x_total.iloc[:len(train)]
    x_test = x_total.iloc[len(train):].reset_index(drop=True)

    x_train = pd.concat([x_train, train_spectrum], axis=1)
    x_test = pd.concat([x_test, test_spectrum], axis=1)
    logger.debug(f'number of features with spec: {x_train.shape}')

    # =========================================
    # === Train model and predict
    # =========================================
    logger.info('Train model and predict')

    # Get folds
    folds_ids = Fold(
        n_splits=config['cv']['n_splits'],
        shuffle=config['cv']['shuffle'],
        random_state=config['cv']['random_state']
    ).get_stratifiedkfold(x_train, y_train)

    # Train and predict
    model_name = config['model']['name']
    model_cls = model_map[model_name]
    params = config['model']
    runner = Runner(model_cls, params, model_output_dir, f'Train_{model_cls.__name__}')

    oof_preds, evals_result = runner.train_cv(x_train, y_train, folds_ids)
    config.update(evals_result)
    test_preds = runner.predict_cv(x_test)

    # =========================================
    # === Make submission file
    # =========================================
    sub = create_submission(test, test_preds, target_column)
    sub.to_csv(model_output_dir/ 'submission.csv', index=False, header=True)

    # =========================================
    # === Save files
    # =========================================
    save_path = model_output_dir / 'output.json'
    json_dump(config, save_path)


if __name__ == '__main__':
    main()
