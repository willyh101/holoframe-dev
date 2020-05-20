import numpy as np
import pandas as pd

def _get_hf_group_mean(df, cols, on, times, inplace):
    if times is not None:
        if len(times==4):
            # 4 numbers, so subtract the baseline response
            bdf = df[(df.time > times[0]) & (df.time < times[1])]
            rdf = df[(df.time > times[2]) & (df.time < times[3])]
            grp_df = rdf.groupby(cols)[[on]].mean() - bdf.groupby(cols)[[on]].mean()
        elif len(times==2):
            # if only 2 numbers, don't baseline
            df = df[(df.time > times[0]) & (df.time < times[1])]
        else: # else it's wrong
            raise IndexError(f'Length of times should either be None (whole trace), 2 (no baselining) or 4 (baselining), not {len(times)}.')
    else: # if you want the whole trace
        grp_df = df.groupby(cols)[[on]].mean()

    if not inplace: # returns a copy
        return grp_df.copy()
    else: # returns a view, caution with editing this dataframe!
        print('warning! not returning a copy... expect weird results.')
        return grp_df

def _handle_xwise_data(vals, name):
    """
    For handling inputs that append to a HoloFrame. This verifies the input
    and converts them into pandas structures for appending.
    """
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

    return appendthis

def _add_xwise(appendthis, on, replace):
    try:
        # join on trials
        frame = frame.join(appendthis, on=on)
    except ValueError:
        # if col already exists, pandas throws an error
        if replace:
            # drop the cols, method depends on whether joining a df or series
            if isinstance(appendthis, pd.DataFrame):
                frame = frame.drop(columns=appendthis.columns).join(appendthis, on=on)
            else:
                frame = frame.drop(columns=appendthis.name).join(appendthis, on=on)
        else:
            raise ValueError(f'Trying to append but column(s) already exist. Use replace = True to override.')

    return frame
