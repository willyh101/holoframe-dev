import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from .hf_helpers import _get_hf_group_mean, _handle_xwise_data, _add_xwise

class HoloFrame(pd.DataFrame):
###---warning! don't touch this section. chnages will break it.---###
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return HoloFrame
###---don't change anything above this line because you'll break it---###

    # note: self == df
    def add_trialwise(self, vals, name=None, replace=True, inplace=False):
        """
        Takes an array of trialwise values (ie. power) and adds it to the dataframe
        for all trials.

        Inputs:
            vals (array): trialwise data to add
            name (str): name of the data you're adding, and turns it into the column name
            replace (bool, False): if there is a column of the same name, replace it if True
            inplace (bool, False): do operation in-place

        Returns:
            df (dataframe): your dataframe
        """

        if inplace:
            frame = self
        else:
            frame = self.copy()

        appendthis = _handle_xwise_data(vals, name) # verify/handle the input data

        # whatever you are trying to append should be the same length as the target
        assert frame.trial.nunique() == appendthis.size, 'Number of trials and length of trialwise data must match.'

        frame = _add_xwise(frame, appendthis, 'trial', replace)

        if not inplace:
            return frame


    def add_cellwise(self, vals, name=None, replace=False, inplace=False):
        """
        Takes an array of trialwise values (ie. stimmed or not) and adds it to the dataframe
        for all cells.

        Inputs:
            vals (array): cellwise data to add
            name (str): name of the data you're adding, and turns it into the column name
            replace (bool, False): if there is a column of the same name, replace it if True
            inplace (bool, False): do operation in-place

        Returns:
            df (dataframe): your dataframe
        """

        if inplace:
            frame = self
        else:
            frame = self.copy()

        appendthis = _handle_xwise_data(vals, name) # verify/handle the input data

        assert frame.cell.nunique() == appendthis.size, 'Number of cells and length of cellwise data must match.'

        frame = _add_xwise(frame, appendthis, 'cell', replace)

        if not inplace:
            return frame


    def add_secs(self, frame_rate, inplace=False):
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame['secs'] = frame['time']/frame_rate

        if not inplace:
            return frame


    def meanby(self, conds, on='df', times=None, inplace=False):
        return _get_hf_group_mean(self, conds, on, times, inplace)


    def mbc(self, conds, on='df', times=None, inplace=False):
        cols = ['cell', *conds]
        return _get_hf_group_mean(self, cols, on, times, inplace)


    def mbt(self, conds=None, on='df', times=None, inplace=False):
        cols = ['cell', 'trial', *conds]
        return _get_hf_group_mean(self, cols, on, times, inplace)


    @classmethod
    def from_traces(cls, traces, trialwise_data=None, frame_rate=None):
        """
        Takes a cells x trials x time array and morphs it into a HoloFrame.
        Can add any trialwise data during construction.

        Inputs:
            traces (arr): cell x trial x time array of experiment
            trialwise_data (arr): vector of any trialwise data to append to df

        Returns:
            holoframe (obj): the df/holoframe object
        """

        df = xr.DataArray(traces.T).to_dataset(dim='dim_0').to_dataframe()
        df = df.reset_index(level=['dim_1', 'dim_2'])
        df = pd.melt(df, ('dim_1', 'dim_2'))
        df = df.rename(columns = {'dim_1':'cell', 'dim_2':'trial', 'variable':'time', 'value':'self'})

        hf = cls(df)
        hf = df.add_trialwise(df, trialwise_data)

        if frame_rate is not None:
            hf = hf.add_secs(frame_rate=frame_rate)

        return hf
