import json
from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from src.hrdp import HRDP

BEAT_ORDER = ["N", "V", "A"]
MAX_HR = 300


def pad_3d(x: np.ndarray, c: int, h: int, w: int, value: int = 0) -> np.ndarray:
    ic, ih, iw = x.shape

    cp_left = (c - ic) // 2
    cp_right = c - cp_left - ic

    hp_left = (h - ih) // 2
    hp_right = h - hp_left - ih

    wp_left = (w - iw) // 2
    wp_right = w - wp_left - iw

    return np.pad(
        x,
        pad_width=((cp_left, cp_right), (hp_left, hp_right), (wp_left, wp_right)),
        constant_values=value,
    )


class HRDPDataset(Dataset):
    """Heart rate density plot (HRDP) torch dataset

    Parameters
    ----------
    df_cohort : pd.DataFrame
        Cohort table
    target_name : str
        Name of the target variable in `df_cohort`
    hrdp_dir : Path
        Path to the HRDP directory with uuid hrdps
    resolution : int (default=36)
        Resolution. Number of seconds per pixel. A resolution of 1 will generate
        a HR-plot (1 column per second), while a resolution of 60 will generate
        1 column per minute.
    crop_size : int (default=1200)
        Dimension of the x-axis.
    hr_dim : int (default=300)
        Maximum HR. Dimension of the y-axis.
    recording_bounds : (default=(0, 86400)
        Timestamps of the onset and offset from the recording that can be used to
        build the HRDP. Any monitoring recorded before or after the bounds will
        be excluded.
        By default all monitoring from the first 24 hours will be used.
    min_input_duration : int (default=None)
        Minimum duration (in seconds) of a HRDP to be used. Any samples with an
        HRDP duration less than `min_input_duration` will excluded when preparing the dataset.
        Defaults to `input_duration`.
        All samples `min_input_duration > x > input_duration` will be 0 padded
        evenly on each side.
    norm_method : str (default=None)
        Normalization method
    img_stats : dict, str, Path (default=None)
        Dictionary or path to a JSON file with a mapping of each beat type to
        'mean' and 'std' used to standardize the HRDP.
    hr_bins : list, np.ndarray
        Heart rate binning values. Used to aggregate beats with an HR interval
    beat_order : list
        Name and ordering of beat types (channels) to use.
        Defaults to ["N", "V", "A"]
    augmentor : callable
        Augmentor applied to the HRDP. Applied after normalization if `norm_method`
        is not None.
    aux_features : list
        List of auxilary features in `df_cohort` to provide with the HRDP
    return_onsets : bool (default=False)
        Boolean whether to return onset timestamps with the input and output
    seed : int
        Random state.
    """

    def __init__(
        self,
        df_cohort: pd.DataFrame,
        target_name: str,
        hrdp_dir: Path,
        resolution: int = 36,
        crop_size: int = 1200,
        hr_dim: int = 300,
        recording_bounds: Tuple[int, int] = (0, 86400),
        min_input_duration: int = None,
        beat_order: List[str] = None,
        norm_method: str = None,
        hr_bins: List[int] = None,
        augmentor: Callable = None,
        aux_features: List[str] = None,
        return_onsets: bool = False,
        seed: int = None,
    ):
        self.df_cohort = df_cohort
        self.hrdp_dir = hrdp_dir
        self.target_name = target_name
        self.crop_size = crop_size
        self.resolution = resolution
        self.hr_dim = hr_dim
        self.recording_bounds = recording_bounds
        self.min_input_duration = min_input_duration or int(
            1 * self.crop_size * self.resolution
        )
        self.beat_order = beat_order or BEAT_ORDER
        self.norm_method = norm_method
        self.augmentor = augmentor
        self.aux_features = aux_features
        self.return_onsets = return_onsets
        self.seed = seed
        self.hrdp_duration = self.crop_size * self.resolution

        # global variables
        self.uuids = None
        self.uuid_timestamps = None
        self._uuid_to_bounds = {}
        self._num_channels = len(self.beat_order)
        self._max_value = 5 * self.resolution

        self._set_uuids()
        self.targets = self.get_targets()

        # set auxilary features
        if self.aux_features is not None:
            self.uuid_to_features = self.get_metadata()
        else:
            self.uuid_to_features = None

        # set state for reproducibility
        self.random_state = np.random.RandomState(self.seed)

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        X, onsets = self.get_hrdp(uuid)
        y = self.get_target(uuid)

        if self.uuid_to_features is not None:
            X = (X, np.array(self.uuid_to_features[uuid]))

        if self.return_onsets:
            return X, y, onsets
        else:
            return X, y

    def _set_uuids(self):
        uuids = self.df_cohort["uuid"]
        for uuid in tqdm(uuids, total=len(uuids), desc="Preparing dataset"):
            with open(self.hrdp_dir / uuid / "metadata.json") as f:
                metadata = json.load(f)
            start = metadata["start"]
            end = metadata["end"]
            hrdp_duration = end - start
            if hrdp_duration < self.min_input_duration:
                continue

            start = max(self.recording_bounds[0], start)
            end = min(start + self.recording_bounds[1], end)
            self._uuid_to_bounds[uuid] = [start, end]

        self.uuids = list(self._uuid_to_bounds.keys())

    def get_hrdp(self, uuid: str, onset_timestamp: float = None) -> np.ndarray:
        uuid_bounds = self._uuid_to_bounds[uuid]
        hrdp_duration = int(uuid_bounds[1] - uuid_bounds[0])
        if onset_timestamp is None:
            onset_timestamp = self.random_state.randint(
                hrdp_duration - self.min_input_duration + 1
            )

        hrdp = HRDP.from_disk(
            self.hrdp_dir / uuid,
            beat_types=self.beat_order,
            resolution=self.resolution,
            hr_dim=self.hr_dim,
        )

        # crop the hrdp
        offset_timestamp = onset_timestamp + self.hrdp_duration
        hrdp = hrdp[:, :, onset_timestamp:offset_timestamp]
        return self.preprocess(hrdp.data), (onset_timestamp, offset_timestamp)

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        # pad the HRDP with 0s if too small
        x = pad_3d(x, self._num_channels, self.hr_dim, self.crop_size)

        # normalization
        if self.norm_method == "normalize":
            x = x / self._max_value
        elif self.norm_method == "resolution":
            x = x / self.resolution  # bounded [0, 5]
        elif self.norm_method == "t_sum_to_1":
            x = x / (x.sum(0, keepdims=True) + 1e-6)
        elif self.norm_method == "hr_sum_to_1":
            x = x / (x.sum(1, keepdims=True) + 1e-6)

        # augmentation
        if self.augmentor:
            x = self.augmentor(x)
        return x

    def get_target(self, uuid: str) -> int:
        return self.df_cohort.query("uuid == @uuid")[self.target_name].item()

    def get_targets(self) -> List[int]:
        return [self.get_target(uuid) for uuid in self.uuids]

    def get_metadata(
        self, aux_features: List[str] = None
    ) -> Dict[str, List[Union[int, float]]]:
        """Returns a dict mapping each uuid to its corresponding auxilary features"""
        aux_features = aux_features or self.aux_features
        return (
            self.df_cohort.set_index("uuid")[aux_features]
            .apply(lambda x: x.tolist(), axis=1)
            .to_dict()
        )

    def reset_state(self, seed: int = None) -> None:
        self.random_state = np.random.RandomState(seed)
