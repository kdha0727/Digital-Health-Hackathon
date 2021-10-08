from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_array
import numpy as np

from typing import cast


def extrapolate(survival_time_event):
    """

    extrapolate survival time whose event is 0 (alive)

    Parameters:

        survival_time_event : data frame.

    """
    if "extrapolated" in survival_time_event.columns:
        return survival_time_event
    extrapolated_survival_time = []
    for row_idx in range(survival_time_event.shape[0]):
        survival_time = survival_time_event.loc[row_idx, "time"]
        if survival_time_event.loc[row_idx, "event"] == 0:
            new_survival_time = survival_time_event.loc[
                (survival_time_event["event"] == 1) &
                (survival_time_event["time"] > survival_time), "time"].mean()
            extrapolated_survival_time.append(new_survival_time)
        else:
            extrapolated_survival_time.append(survival_time)
    survival_time_event["extrapolated"] = extrapolated_survival_time
    # survival_time_event.to_csv('./data/survival_time_extrapolation.csv')
    return survival_time_event


class ExtrapolationYProcessor:
    """

    extrapolate y whose event is 0 (alive), and min-max scale remaining data.

    Parameters:

        feature_range : tuple (min, max), default=(-1, 1)
            Desired range of transformed data.

        copy : bool, default=True
            Set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array).

        clip : bool, default=False
            Set to True to clip transformed values of held-out data to
            provided `feature range`.

    Attributes:

        scaler_ : MinMaxScaler
            backend-scaler which is used in min-max scaling when transform runs.

    Methods:

        fit(time, event)
            Compute data to be used for later scaling.

        transform(time, event)
            Scale features of data according to feature_range.

        fit_transform(time, event)
            Fit to data, then transform it.

    """

    def __init__(self, *, feature_range=(-1, 1), copy=True, clip=False):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, 'scaler_'):
            del self.scaler_

    def fit(self, df):
        """
        fit(time, event)
            Compute data to be used for later scaling.
        """
        self._reset()
        df = extrapolate(df)
        time = df['extrapolated']
        time = self._validate_data(time)
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range, copy=self.copy, clip=self.clip)
        self.scaler_.fit(time.reshape(-1, 1))
        return self

    def transform(self, df):
        """
        transform(time, event)
            Scale features of data according to feature_range.
        """
        df = extrapolate(df)
        time = self._validate_data(df['extrapolated'])
        if self.copy:
            time = time.copy()
        time = self.scaler_.transform(time.reshape(-1, 1)).reshape(-1)
        return time

    def fit_transform(self, df):
        """
        fit_transform(time, event)
            Fit to data, then transform it.
        """
        return self.fit(df).transform(df)

    @staticmethod
    def _validate_data(data):
        return cast(np.ndarray, check_array(data, accept_sparse=True, force_all_finite="allow-nan", ensure_2d=False))


class StandardYProcessor:
    """

    truncate (convert to nan or given value) y whose event is 0 (alive), and min-max scale remaining data.

    Parameters:

        feature_range : tuple (min, max), default=(-1, 1)
            Desired range of transformed data.

        default_value : Number, default=NaN
            Value of truncated elements.

        copy : bool, default=True
            Set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array).

        clip : bool, default=False
            Set to True to clip transformed values of held-out data to
            provided `feature range`.

    Attributes:

        scaler_ : MinMaxScaler
            backend-scaler which is used in min-max scaling when transform runs.

    Methods:

        fit(time, event)
            Compute data to be used for later scaling.

        transform(time, event)
            Scale features of data according to feature_range.

        fit_transform(time, event)
            Fit to data, then transform it.

    """

    def __init__(self, *, feature_range=(-1, 1), default_value=np.nan, copy=True, clip=False):
        self.feature_range = feature_range
        self.default_value = default_value
        self.copy = copy
        self.clip = clip

    def _reset(self):
        if hasattr(self, 'scaler_'):
            del self.scaler_

    def fit(self, time, event):
        """
        fit(time, event)
            Compute data to be used for later scaling.
        """
        self._reset()
        time = self._validate_data(time)
        event = self._validate_data(event)
        self.scaler_ = MinMaxScaler(feature_range=self.feature_range, copy=self.copy, clip=self.clip)
        self.scaler_.fit(time[event != 0].reshape(-1, 1))
        return self

    def transform(self, time, event):
        """
        transform(time, event)
            Scale features of data according to feature_range.
        """
        time = self._validate_data(time)
        event = self._validate_data(event)
        mask = event != 0
        if self.copy:
            time = time.copy()
        time[mask] = self.scaler_.transform(time[mask].reshape(-1, 1)).reshape(-1)
        time[mask == 0] = self.default_value
        return time

    def fit_transform(self, time, event):
        """
        fit_transform(time, event)
            Fit to data, then transform it.
        """
        return self.fit(time, event).transform(time, event)

    @staticmethod
    def _validate_data(data):
        return cast(np.ndarray, check_array(data, accept_sparse=True, force_all_finite="allow-nan", ensure_2d=False))


class TanhYProcessor(StandardYProcessor):
    """

    Min-max scale, followed by applying tanh, y whose event is 1,
    and set remaining data as default value.

    Parameters:

        feature_range : tuple (min, max), default=(-1, 1)
            Desired range of data, which will be used as input of tanh.

        default_value : Number, default=NaN
            Default output of data which is not under tanh transformation.

        copy : bool, default=True
            Set to False to perform inplace row normalization and avoid a
            copy (if the input is already a numpy array).

        clip : bool, default=False
            Set to True to clip transformed values of held-out data to
            provided `feature range`.

    Attributes:

        scaler_ : MinMaxScaler
            backend-scaler which is used in min-max scaling when transform runs.

    Methods:

        fit(time, event)
            Compute data to be used for later scaling.

        transform(time, event)
            Scale features of data.

        fit_transform(time, event)
            Fit to data, then transform it.

    """

    def __init__(self, *, feature_range=(-2, 2), default_value=1, copy=True, clip=False):  # tanh(2) = 0.964..
        super().__init__(feature_range=feature_range, default_value=default_value, copy=copy, clip=clip)

    def transform(self, time, event):
        """
        transform(time, event)
            Scale features of data.
        """
        mask = event != 0
        time[mask] = np.tanh(self.scaler_.transform(time[mask]))
        time[mask == 0] = self.default_value
        return time


__all__ = ['MinMaxScaler', 'StandardYProcessor', 'TanhYProcessor', 'ExtrapolationYProcessor', 'extrapolate']
