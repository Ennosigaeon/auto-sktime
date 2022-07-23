import numpy as np
import pandas as pd
from sktime.datatypes._utilities import GET_CUTOFF_SUPPORTED_MTYPES


# Patches https://github.com/alan-turing-institute/sktime/blob/main/sktime/datatypes/_utilities.py#L171
def get_cutoff(
        obj,
        cutoff=0,
        return_index=False,
        reverse_order=False,
        check_input=False,
        convert_input=True,
):
    """Get cutoff = latest time point of time series or time series panel.
    Assumptions on obj are not checked, these should be validated separately.
    Function may return unexpected results without prior validation.
    Parameters
    ----------
    obj : sktime compatible time series data container
        must be of Series, Panel, or Hierarchical scitype
        all mtypes are supported via conversion to internally supported types
        to avoid conversions, pass data in one of GET_CUTOFF_SUPPORTED_MTYPES
    cutoff : int, optional, default=0
        current cutoff, used to offset index if obj is np.ndarray
    return_index : bool, optional, default=False
        whether a pd.Index object should be returned (True)
            or a pandas compatible index element (False)
        note: return_index=True may set freq attribute of time types to None
            return_index=False will typically preserve freq attribute
    reverse_order : bool, optional, default=False
        if False, returns largest time index. If True, returns smallest time index
    check_input : bool, optional, default=False
        whether to check input for validity, i.e., is it one of the scitypes
    convert_input : bool, optional, default=True
        whether to convert the input (True), or skip conversion (False)
        if skipped, function assumes that inputs are one of GET_CUTOFF_SUPPORTED_MTYPES
    Returns
    -------
    cutoff_index : pandas compatible index element (if return_index=False)
        pd.Index of length 1 (if return_index=True)
    Raises
    ------
    ValueError, TypeError, if check_input or convert_input are True
        exceptions from check or conversion failure, in check_is_scitype, convert_to
    """
    from sktime.datatypes import check_is_scitype, convert_to

    # deal with VectorizedDF
    if hasattr(obj, "X"):
        obj = obj.X

    if check_input:
        valid = check_is_scitype(obj, scitype=["Series", "Panel", "Hierarchical"])
        if not valid:
            raise ValueError("obj must be of Series, Panel, or Hierarchical scitype")

    if convert_input:
        obj = convert_to(obj, GET_CUTOFF_SUPPORTED_MTYPES)

    if cutoff is None:
        cutoff = 0

    if len(obj) == 0:
        return cutoff

    # numpy3D (Panel) or np.npdarray (Series)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3:
            cutoff_ind = obj.shape[-1] + cutoff
        if obj.ndim < 3 and obj.ndim > 0:
            cutoff_ind = obj.shape[0] + cutoff
        if reverse_order:
            cutoff_ind = 0
        if return_index:
            return pd.RangeIndex(cutoff_ind - 1, cutoff_ind)
        else:
            return cutoff_ind

    # define "first" or "last" index depending on which is desired
    if reverse_order:
        ix = 0
        agg = min
    else:
        ix = -1
        agg = max

    if isinstance(obj, pd.Series):
        return obj.index[[ix]] if return_index else obj.index[ix]

    # nested_univ (Panel) or pd.DataFrame(Series)
    if isinstance(obj, pd.DataFrame) and not isinstance(obj.index, pd.MultiIndex):
        objcols = [x for x in obj.columns if obj.dtypes[x] == "object"]
        # pd.DataFrame
        if len(objcols) == 0:
            return obj.index[[ix]] if return_index else obj.index[ix]
        # nested_univ
        else:
            if return_index:
                idxx = [x.index[[ix]] for col in objcols for x in obj[col]]
            else:
                idxx = [x.index[ix] for col in objcols for x in obj[col]]
            return max(idxx)

    # pd-multiindex (Panel) and pd_multiindex_hier (Hierarchical)
    if isinstance(obj, pd.DataFrame) and isinstance(obj.index, pd.MultiIndex):
        idx = obj.index
        ####
        # Formerly: series_idx = [obj.loc[x].index.get_level_values(-1) for x in idx.droplevel(-1)]
        ####
        series_idx = [obj.loc[x].index.get_level_values(-1) for x in idx.remove_unused_levels().levels[0]]
        if return_index:
            cutoffs = [x[[-1]] for x in series_idx]
        else:
            cutoffs = [x[-1] for x in series_idx]
        return agg(cutoffs)

    # df-list (Panel)
    if isinstance(obj, list):
        if return_index:
            idxs = [x.index[[ix]] for x in obj]
        else:
            idxs = [x.index[ix] for x in obj]
        return agg(idxs)
