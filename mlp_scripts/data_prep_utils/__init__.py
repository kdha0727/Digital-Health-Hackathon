from .dataset import *
from .loader import *
from .preprocessor import *

from . import dataset
from . import loader
from . import preprocessor


def get_data_dir():

    from .. import config
    return config.DATA_PATH


def get_raw_data():

    import pandas as pd
    import functools

    csv_loader = functools.partial(pd.read_csv, index_col=0)

    data_dir = get_data_dir()
    clinical_variables = csv_loader(data_dir / 'Clinical_Variables.csv')
    generic_alterations = csv_loader(data_dir / 'Genetic_alterations.csv')
    survival_time_event = csv_loader(data_dir / 'Survival_time_event.csv')
    treatment = csv_loader(data_dir / 'Treatment.csv')

    return clinical_variables, generic_alterations, survival_time_event, treatment


def get_processed_data():

    # import libraries
    import torch
    import pandas as pd
    from .dataset import PandasDataset
    from .preprocessor import ExtrapolationYProcessor, MinMaxScaler
    clinical_variables, generic_alterations, survival_time_event, treatment = get_raw_data()

    # process clinical variables
    clinical_proc = MinMaxScaler(feature_range=(0, 1))
    clinical_variables = pd.DataFrame(
        clinical_proc.fit_transform(clinical_variables),
        columns=clinical_variables.columns
    )

    # process survival time
    y_proc = ExtrapolationYProcessor(feature_range=(0, 1))
    y = y_proc.fit_transform(survival_time_event)

    # concat
    full = pd.concat([treatment, clinical_variables, generic_alterations], axis=1)
    full['time'] = y

    # convert to pytorch dataset
    processed_dataset = PandasDataset(full, label_target='time', dtype=torch.float32)

    # for inverse transform, set processor attributes
    processed_dataset.clinical_proc = clinical_proc
    processed_dataset.y_proc = y_proc

    # return processed dataset
    return processed_dataset
