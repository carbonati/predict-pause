import os
import argparse
import datetime
import json
from typing import Dict, List, Tuple, Union
from pathlib import Path
from tqdm.notebook import tqdm
from scipy import sparse
import numpy as np
import pandas as pd

NORMAL_TAGS = ["N"]
ATRIAL_TAGS = ["A"]
VENTRICULAR_TAGS = ["V"]


def has_tags(x: np.ndarray, tags: List[str]) -> np.ndarray:
    return np.array([v in tags for v in x])


def generate_hrdp(
    uuid: str, onset_dir: Path, height: int = 300, resolution: int = 1
) -> Tuple[Dict[str, sparse.coo_matrix], Dict[str, Union[int, str]]]:
    """Generates an HRDP sparse matrix from QRS onsets"""
    # load qrs
    df_onset = pd.read_csv(onset_dir / f"{uuid}.csv")
    qrs_onsets = np.array(df_onset["onset_s"])
    qrs_tags = np.array(df_onset["wave_type"])

    # compute HR
    x_hr = np.minimum(60 / (np.diff(qrs_onsets) + 1e-6), 299).astype(int)
    x_hr = np.round(x_hr * height / 300).astype(int)

    # set onsets to start at 0
    start = qrs_onsets[0]
    end = qrs_onsets[-1]
    qrs_onsets = qrs_onsets - start

    width = int(qrs_onsets[-1] / resolution + 1)

    # map timestamps to an index along the x-axis
    x_ts = qrs_onsets.astype(int)[1:]

    # coo sparse format: data (row index, column index)
    wave_to_hrdp = {}
    wave_to_hrdp["all"] = sparse.coo_matrix(
        (np.ones(len(x_hr)), (x_hr, x_ts)), shape=(height, width)
    )

    # sinus
    x_is_sinus = has_tags(qrs_tags, NORMAL_TAGS)[1:]
    wave_to_hrdp["N"] = sparse.coo_matrix(
        (np.ones(x_is_sinus.sum()), (x_hr[x_is_sinus], x_ts[x_is_sinus])),
        shape=(height, width),
    )

    # ventricular
    x_is_veb = has_tags(qrs_tags, VENTRICULAR_TAGS)[1:]
    wave_to_hrdp["V"] = sparse.coo_matrix(
        (np.ones(x_is_veb.sum()), (x_hr[x_is_veb], x_ts[x_is_veb])),
        shape=(height, width),
    )

    # supraventricular
    x_is_psvc = has_tags(qrs_tags, ATRIAL_TAGS)[1:]
    wave_to_hrdp["A"] = sparse.coo_matrix(
        (np.ones(x_is_psvc.sum()), (x_hr[x_is_psvc], x_ts[x_is_psvc])),
        shape=(height, width),
    )

    # metadata
    metadata = {"uuid": uuid, "start": start, "end": end, "duration": end - start}

    return wave_to_hrdp, metadata


def main() -> None:
    """Generates cohort HRDPs"""
    start_dt = datetime.datetime.utcnow()
    print(f"Preparing inputs for HRDP model @ {str(start_dt)}")

    args = make_parser().parse_args()
    working_dir = Path(args.working_dir)
    height = args.height
    resolution = args.resolution

    onset_dir = working_dir / "qrs_onsets"
    output_dir = working_dir / f"hrdp_{height}h_{resolution}r"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving HRDPs to {output_dir}")

    uuids = [filename.split(".")[0] for filename in os.listdir(onset_dir)]
    for uuid in tqdm(uuids, total=len(uuids), desc="Generating HRDPs"):
        wave_to_hrdp, metadata = generate_hrdp(
            uuid, onset_dir, height=height, resolution=resolution
        )
        uuid_dir = output_dir / uuid
        uuid_dir.mkdir(exist_ok=True)

        sparse.save_npz(uuid_dir / "hrdp_all.npz", wave_to_hrdp["all"])
        sparse.save_npz(uuid_dir / "hrdp_N.npz", wave_to_hrdp["N"])
        sparse.save_npz(uuid_dir / "hrdp_V.npz", wave_to_hrdp["V"])
        sparse.save_npz(uuid_dir / "hrdp_A.npz", wave_to_hrdp["A"])
        with open(uuid_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

    end_dt = datetime.datetime.utcnow()
    runtime = round((end_dt - start_dt).total_seconds(), 2)
    print(f"Finished generation HRDPs @ {str(end_dt)} - Runtime : {runtime}s")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generates HRDPs from QRS onsets")
    parser.add_argument(
        "--working_dir",
        help="Path to snorcam versioned dataset",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        help="Temporal resolution of the HRDP. Number of seconds per pixel",
    )
    parser.add_argument("--height", type=int, default=300, help="Maximum heart rate")
    return parser


if __name__ == "__main__":
    main()
