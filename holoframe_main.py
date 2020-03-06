import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

def _get_hf_group_unstacked(df, cols, on, times, inplace):
    if times is not None:
        if len(times==4):
            # 4 numbers, so subtract the baseline response
            bdf = df[(df.time > times[0]) & (df.time < times[1])]
            rdf = df[(df.time > times[2]) & (df.time < times[3])]
            grp_df = rdf.groupby(cols)[on].mean().unstack() - bdf.groupby(cols)[on].mean().unstack()
        elif len(times==2):
            # if only 2 numbers, don't baseline
            df = df[(df.time > times[0]) & (df.time < times[1])]
        else: # else it's wrong
            raise IndexError(f'Length of times should either be None (whole trace), 2 (no baselining) or 4 (baselining), not {len(times)}.')
    else: # if you want the whole trace
        grp_df = df.groupby(cols)[on].mean().unstack()

    if not inplace: # returns a copy
        return grp_df.copy()
    else: # returns a view, caution with editing this dataframe!
        print('warning! not returning a copy... expect weird results.')
        return grp_df


class HoloFrame(pd.DataFrame):
###---warning! don't touch this section. chnages will break it.---###
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return HoloFrame
###---don't change anything above this line because you'll break it---###

    # note: self == df
    def add_trialwise(self, vals, name=None, replace=False, inplace=False):
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

        if isinstance(vals, (pd.Series, pd.DataFrame)):
            # if a series or dataframe, no need to do anything
            appendthis = vals
        elif isinstance(vals, (np.ndarray, list)):
            # for passing array-likes, must be 1 dim
            if isinstance(vals, np.ndarray):
                assert hasattr(vals, 'ndim') and vals.ndim == 1, 'Must pass array with one dimension only.'
            # need to have a name to name the col
            if name is not None:
                appendthis = pd.Series(vals, name=name)
            else:
                raise ValueError('Name is required for lists and arrays.')
        elif isinstance(vals, dict):
            # for passing dicts, make them a dataframe
            appendthis = pd.DataFrame(vals)
        else:
            raise TypeError('Must pass list, dict, np.ndarray, pd.Series, or pd.DataFrame only.')

        assert frame.trial.nunique() == appendthis.size, 'Number of trials and length of trialwise data must match.'

        try:
            # join on trials
            frame = frame.join(appendthis, on='trial')
        except ValueError:
            # if col already exists, pandas throws an error
            if replace:
                # drop the cols, method depends on whether joining a df or series
                if isinstance(appendthis, pd.DataFrame):
                    frame = frame.drop(columns=appendthis.columns).join(appendthis, on='trial')
                else:
                    frame = frame.drop(columns=appendthis.name).join(appendthis, on='trial')
            else:
                raise ValueError(f'Trying to append but column(s) already exist. Use replace=True to override.')

        if not inplace:
            return frame


    # def times(self, frame_rate):
    #     self['secs'] = self['time']/frame_rate


    def meanby(self, conds, on='df', times=None, inplace=False):
        return _get_hf_group_unstacked(self, conds, on, times, inplace)


    def mbc(self, conds, on='df', times=None, inplace=False):
        cols = ['cell', *conds]
        return _get_hf_group_unstacked(self, cols, on, times, inplace)


    def mbt(self, conds=None, on='df', times=None, inplace=False):
        cols = ['trial', 'cell', *conds]
        return _get_hf_group_unstacked(self, cols, on, times, inplace)


    @classmethod
    def from_traces(cls, traces, trialwise_data=None):
        """
        Takes a cells x trials x time array and morphs it into a HoloFrame.
        """

        df = xr.DataArray(traces.T).to_dataset(dim='dim_0').to_dataframe()
        df = df.reset_index(level=['dim_1', 'dim_2'])
        df = pd.melt(df, ('dim_1', 'dim_2'))
        df = df.rename(columns = {'dim_1':'cell', 'dim_2':'trial', 'variable':'time', 'value':'self'})

        df = df.add_trialwise(df, trialwise_data, names=None)

        return cls(df)
