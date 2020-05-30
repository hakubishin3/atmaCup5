import json
import pathlib
import argparse
import pandas as pd
from src.get_folds import Fold
from src.runner import Runner
from src.utils import get_logger, json_dump, seed_everything
from src.submission import create_submission
from features.base import load_features
from models.model_lightgbm import Model_LightGBM
from multiprocessing import cpu_count
import lightgbm as lgb 
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


seed_everything(71)

model_map = {
    'lightgbm': Model_LightGBM
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
        '--config', default='./configs/model_0.json')
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
    # === Adversarial Validation
    # =========================================
    logger.info("adversarial validation")
    train_adv = x_train
    test_adv = x_test
    train_adv['target'] = 0
    test_adv['target'] = 1
    train_test_adv = pd.concat([train_adv, test_adv], axis=0, sort=False).reset_index(drop=True)
    target = train_test_adv['target'].values

    train_set, val_set = train_test_split(train_test_adv, test_size=0.33, random_state=71, shuffle=True)
    x_train_adv = train_set[feature_name]
    y_train_adv = train_set['target']
    x_val_adv = val_set[feature_name]
    y_val_adv = val_set['target']
    logger.debug(f'the number of train set: {len(x_train_adv)}')
    logger.debug(f'the number of valid set: {len(x_val_adv)}')

    train_lgb = lgb.Dataset(x_train_adv, label=y_train_adv)
    val_lgb = lgb.Dataset(x_val_adv, label=y_val_adv)
    lgb_model_params = config["adversarial_validation"]["lgb_model_params"]
    lgb_train_params = config["adversarial_validation"]["lgb_train_params"]
    clf = lgb.train(
        lgb_model_params,
        train_lgb,
        valid_sets=[train_lgb, val_lgb],
        valid_names=['train', 'valid'],
        **lgb_train_params
    )

    feature_imp = pd.DataFrame(
        sorted(zip(clf.feature_importance(importance_type='gain'), feature_name)), columns=['value', 'feature']
    )
    plt.figure(figsize=(20, 10))
    sns.barplot(x='value', y='feature', data=feature_imp.sort_values(by='value', ascending=False).head(20))
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.savefig(model_output_dir / "feature_importance_adv.png")

    config.update({
        'adversarial_validation_result': {
            'score': clf.best_score,
            'feature_importances': feature_imp.set_index("feature").sort_values(by="value", ascending=False).head(20).to_dict()["value"]
        }
    })

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
