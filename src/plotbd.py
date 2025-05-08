import numpy as _np
from numpy.typing import NDArray as _NDArray
from dataclasses import dataclass as _dataclass
from scipy.sparse import coo_matrix as _coo_matrix
from scipy.spatial.ckdtree import cKDTree as _cKDTree
from tqdm import tqdm as _tqdm

CategorizableType: type = _np.str_ | _np.int_
NumberType: type = int | float

@_dataclass
class SpAnnPoints:
    """
    Spatial Annotated Points.
    
    Attrs:
        coordinates (ndarray): a 2d-array of x, y of locations of points in space.
        
        masks (ndarray): a 1d-array of labels of points. Points of different labels are
        considered to be of different objects.
    """
    coordinates: _np.ndarray
    masks: _NDArray[CategorizableType]
    def __post_init__(self):
        assert len(self.masks) == self.coordinates.shape[0]
        assert self.coordinates.shape[1] == 2
        self._n_points: int = self.coordinates.shape[0]
        self._xrange: tuple[NumberType] = (
            self.coordinates[:,0].min(),
            self.coordinates[:,0].max(),
        )
        self._yrange: tuple[NumberType] = (
            self.coordinates[:,1].min(),
            self.coordinates[:,1].max(),
        )
        return

def get_boundaries(
    ann_points: SpAnnPoints,
    nbhd_radius: NumberType | None = None,
    verbose: bool = True,
) -> _NDArray[_np.bool_]:
    """
    Get boundaries of many objects.

    Args:
        ann_points (SpAnnPoints): points in 2d-space with annotations of their belonging objects.
         
        nbhd_radius (NumberType | None, optional): the range within which points are considered as 
        neighbors. If neighbors of a point include different object annotations, this point
        is regarded as a boundary point. If `None`, roughly estimates the radius by calculating
        the density of the points.

    Return:
        ndarray: 1d-array of booleans with shape corresponding to `ann_points`. If a point is
        boundary point, its corresponding entry is `True`, otherwise `False`.
    """
    if nbhd_radius is None:
        area: NumberType = (
                ann_points._xrange[1] - ann_points._xrange[0]
            ) * (
                ann_points._yrange[1] - ann_points._yrange[0]
            )
        assert area > 0
        # Area occupied by a point on average
        inv_density: NumberType = area / ann_points._n_points
        # Estimate nbhd_radius according to S=pi*r^2 -> r = sqrt(S/pi)
        nbhd_radius: NumberType = 2 * _np.sqrt(inv_density/_np.pi)
        # Increase it a bit
        nbhd_radius *= 1.05
    assert isinstance(nbhd_radius, NumberType)

    # Construct spatial graph
    ckdtree: _cKDTree = _cKDTree(
        data=ann_points.coordinates,
    )
    dist_matrix: _coo_matrix = ckdtree.sparse_distance_matrix(
        other=ckdtree,
        max_distance=nbhd_radius,
        p=2,
        output_type="coo_matrix",
    )
    boundary_masks: _NDArray[_np.bool_] = _np.zeros(
        shape=(ann_points._n_points,),
        dtype=bool,
    )
    # Just implement a for-loop so far
    if verbose:
        _itor = _tqdm(
            range(dist_matrix.shape[0]),
            desc='Finding boundaries',
            ncols=60,
        )
    else:
        _itor = range(dist_matrix.shape[0])
    for i_point in _itor:
        nbors: _NDArray[_np.int_] = dist_matrix.getrow(i_point).nonzero()[1]
        if len(nbors)==0:
            continue
        masks_nbors_unique: _NDArray[CategorizableType] = _np.unique(ann_points.masks[nbors])
        if len(masks_nbors_unique) > 1:
            boundary_masks[i_point] = True
            continue
        if masks_nbors_unique[0] != ann_points.masks[i_point]:
            boundary_masks[i_point] = True
            continue
    return boundary_masks

def plot_boundaries(
    coordinates: _np.ndarray,
    boundaries_flags: _NDArray[_np.bool_],
):
    import seaborn as sns
    return sns.scatterplot(
        x=coordinates[:,0][boundaries_flags],
        y=coordinates[:,1][boundaries_flags],
    )
        

