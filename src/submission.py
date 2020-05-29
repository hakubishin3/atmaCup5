import numpy as np
import pandas as pd


def create_submission(test, test_preds, target_column):
    test[target_column] = test_preds
    return test[[target_column]]
