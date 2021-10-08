import torch
import torch.utils.data

import pandas as pd


class PandasDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataframe: "pd.DataFrame",
            label_target: str,
            drop_nan: bool = True,
            dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        self.label_target = [label_target]
        self.label_input = list(dataframe.columns)
        self.label_input.remove(label_target)
        # Drop NaN
        if drop_nan:
            dataframe = dataframe.drop(dataframe.index[dataframe[label_target] != dataframe[label_target]])
        dataframe.index = range(len(dataframe))
        self.samples = dataframe
        self.dtype = dtype

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> ...:
        row = self.samples.iloc[index]
        x = torch.tensor(row[self.label_input])
        y = torch.tensor(row[self.label_target])
        if self.dtype is not None:
            x, y = x.to(self.dtype), y.to(self.dtype)
        return x, y


__all__ = ['PandasDataset']
