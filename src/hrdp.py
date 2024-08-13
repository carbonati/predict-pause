from __future__ import annotations
import json
from typing import Any, Dict, List, Tuple, Union

from pathlib import Path
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns

DEFAULT_BEAT_TYPES = ["N", "V", "A"]


def get_beat_hr_and_alphas(
    X: np.ndarray,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Returns HRs and alpha density values of each beat

    Parameters
    ----------
    X : np.ndarray
        HRDP matrix of shape (channel, HR, timestamp)

    Returns
    --------
    dict
        Mapping of each beat type to list of HRs at each timestamp
    dict
        Mapping of each beat type to the counts of each HR at each timestamp
    """
    beat_to_hrs = {}
    beat_to_alphas = {}
    for ch in range(X.shape[0]):
        X_ch = X[ch, :, :]
        hrs = []
        alphas = []
        # for each timestamp, add all HRs and counts at each HR
        for ts in range(X_ch.shape[1]):
            hr_mask = X_ch[:, ts] > 0
            hrs.append(list(np.where(hr_mask)[0] + 1))
            alphas.append(list(X_ch[hr_mask, ts]))
        beat_to_hrs[ch] = hrs
        beat_to_alphas[ch] = alphas
    return beat_to_hrs, beat_to_alphas


def plot_hrdp(
    X: np.ndarray,
    use_alpha: bool = False,
    figsize: Tuple[int, int] = (20, 8),
    markersize_map: int = None,
    alpha_map: Dict[int, int] = None,
    pal: Union[sns.palettes._ColorPalette, List[Tuple[int, int, int]]] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    if pal is None:
        pal = sns.color_palette()
        # (black, red, blue)
        pal = [(0, 0, 0), pal[3], pal[0]]
    if markersize_map is None:
        markersize_map = {0: 2, 1: 10, 2: 10}  # N  # V  # A
    if alpha_map is None:
        alpha_map = {0: 0.5, 1: 0.5, 2: 0.7}

    # compute HRs at every timestamp for each beat
    beat_to_hrs, beat_to_alphas = get_beat_hr_and_alphas(X)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

    for ch in beat_to_hrs:
        hrs = beat_to_hrs[ch]
        alphas = beat_to_alphas[ch]
        color = pal[ch]
        markersize = markersize_map[ch]
        for i, hr_values in enumerate(hrs):
            if use_alpha:
                rgba = [color + (alphas[i][j],) for j in range(len(hr_values))]
                ax.scatter(
                    [i + 1] * len(hr_values),
                    hr_values,
                    color=rgba,
                    label=ch,
                    s=markersize,
                )
            else:
                ax.plot(
                    [i + 1] * len(hr_values),
                    hr_values,
                    ".",
                    color=color,
                    label=ch,
                    markersize=markersize,
                    alpha=alpha_map[ch],
                )

    plt.tight_layout()
    plt.ylim(0, X.shape[1])
    plt.xlim(0, X.shape[2])
    sns.despine()

    return ax


class HRDP:
    """Heart rate density plot (HRDP) object

    3-dimensional representation of the instantaneous heart rate during a Holter recording.
    The x-axis represents time; the y-axis represents heart rate; and the z-axis consists
    of three channels corresponding to the beat classification, which by default are:
        N - Normal
        V - premature ventricular complex
        A - supraventricular premature complex

    Parameters
    ----------
    data : np.ndarray
        HRDP matrix of shape (num_beat_types, max_hr, duration // resolution)
    metadata : dict
        Metadata storing holter 'uuid', 'start', and 'end'
    beat_types : list (default=['N', 'V', 'A'])
        Order of beat classification
    resolution : int (default=1)
        Number of seconds per pixel (zoom-level)

    Attributes
    ----------
    data : np.ndarray
        HRDP matrix
    uuid : str
        UUID of the Holter
    start : float
        Timestamp at the start of the HRDP
    end : float
        Timestamp at the end of the HRDP
    duration : float
        Duration (s) of the HRDP
    window_size : int
        Size of the HRDP at the current resolution (zoom-level).
        The "x-axis"
    shape : tuple
        Shape of the object
    """

    def __init__(
        self,
        data: np.ndarray,
        metadata: Dict[str, Any],
        beat_types: List[str] = None,
        resolution: int = 1,
    ):
        self._data = data
        self.beat_types = beat_types or DEFAULT_BEAT_TYPES
        self.metadata = metadata
        self.resolution = resolution

        self._uuid = self.metadata["uuid"]
        self._start = self.metadata["start"]
        self._end = self.metadata["end"]
        self._duration = self.end - self.start
        self._window_size = self.data.shape[2]

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def start(self) -> float:
        return self._start

    @property
    def end(self) -> float:
        return self._end

    @property
    def duration(self) -> float:
        return self._duration

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._data.shape

    def __getitem__(self, key) -> HRDP:
        start, end = self.start, self.end
        if isinstance(key, (int, slice)):
            key = key if isinstance(key, slice) else slice(key)
            X = self._data[key]
            beat_types = self.beat_types[key]
        elif isinstance(key, str):
            X = self._data[self.beat_types.index(key)][None]
            beat_types = [key]
        elif isinstance(key, tuple) and len(key) < 4:
            beat_types = self.beat_types[key[0]]
            if len(key) == 3:
                # slice by duration (in seconds)
                start = key[2].start or start
                end = key[2].stop or end
                X = self._data[
                    key[0],
                    key[1],
                    int(start / self.resolution): int(end / self.resolution),
                ]
            else:
                X = self._data[key]
        elif isinstance(key, tuple) and len(key) > 3:
            raise IndexError(
                "too many indices for HRDP: HRDP is 2-dimensional"
                f", but {len(key)} were indexed"
            )
        else:
            raise IndexError(f"Unrecognized index of type {type(key)}: {key}")

        hrdp = HRDP(
            X,
            metadata={**self.metadata, "start": start, "end": end},
            beat_types=beat_types,
            resolution=self.resolution,
        )
        return hrdp

    def __str__(self) -> str:
        format_str = (
            f"{self.__class__.__name__}(start={self.start:.2f}, end={self.end:.2f}, "
            f"r={self.resolution}, beat_types={self.beat_types})"
        )
        return format_str

    def __repr__(self) -> str:
        return self.__str__()

    def __array__(self) -> np.ndarray:
        return np.asarray(self._data)

    @classmethod
    def from_disk(
        cls,
        working_dir: Union[str, Path],
        beat_types: List[str] = None,
        resolution: int = 1,
        hr_dim: int = 300,
        hr_bins: List[int] = None,
    ) -> HRDP:
        beat_types = beat_types or DEFAULT_BEAT_TYPES

        # load metadata
        working_dir = Path(working_dir)
        with open(working_dir / "metadata.json") as f:
            metadata = json.load(f)

        data = []
        for beat_type in beat_types:
            sparse_hrdp = sparse.load_npz(working_dir / f"hrdp_{beat_type}.npz")
            # bin HR values for each resolution (seconds per pixel)
            sparse_hrdp.col = np.round(sparse_hrdp.col / resolution).astype(int)
            # bin/map HR values
            if hr_bins:
                sparse_hrdp.row = np.digitize(sparse_hrdp.row, hr_bins)

            sparse_hrdp.resize(
                (hr_dim, int(np.ceil(sparse_hrdp.shape[1] / resolution)))
            )
            data.append(sparse_hrdp.toarray())

        X = np.stack(data).astype(np.float32)
        hrdp = cls(X, metadata=metadata, beat_types=beat_types, resolution=resolution)
        return hrdp

    @classmethod
    def from_sparse(
        cls,
        data: Dict[str, sparse.coo_matrix],
        metadata: Dict[str, Any],
        resolution: int = 1,
        hr_dim: int = 300,
        hr_bins: List[int] = None,
    ) -> HRDP:
        X = []
        for sparse_hrdp in data.values():
            # bin HR values for each resolution (seconds per pixel)
            sparse_hrdp = sparse_hrdp.copy()
            sparse_hrdp.col = np.round(sparse_hrdp.col / resolution).astype(int)
            # bin/map HR values
            if hr_bins:
                sparse_hrdp.row = np.digitize(sparse_hrdp.row, hr_bins)

            sparse_hrdp.resize(
                (hr_dim, int(np.ceil(sparse_hrdp.shape[1] / resolution)))
            )
            X.append(sparse_hrdp.toarray())

        X = np.stack(X).astype(np.float32)
        hrdp = cls(
            X, metadata=metadata, beat_types=list(data.keys()), resolution=resolution
        )
        return hrdp

    def plot(self, *args, **kwargs) -> plt.Axes:
        return plot_hrdp(self.data, *args, **kwargs)
