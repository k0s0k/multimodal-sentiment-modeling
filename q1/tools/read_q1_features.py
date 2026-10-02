"""Read every defined Q1 feature through the portable CSV index; no GPU required."""

import argparse, csv, hashlib
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=Path("features"))
    a = p.parse_args()
    a.features = a.features.resolve()
    with (a.features / "samples_index.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100 and len({r["sample_key"] for r in rows}) == 100
    for row in rows:
        for branch, dim in [
            ("text", 1024),
            ("audio", 1024),
            ("acoustic", 25),
            ("face", 44),
            ("video", 768),
        ]:
            path = (a.features / row[branch + "_path"]).resolve()
            assert path.is_relative_to(a.features), path
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == row[branch + "_sha256"]
            )
            with np.load(path, allow_pickle=False) as z:
                x = z["features"]
                mask = (
                    z["valid_mask"]
                    if branch in ["text", "audio", "acoustic"]
                    else z["valid"]
                )
                time = (
                    z["time_s"]
                    if branch in ["text", "audio", "acoustic"]
                    else z["times"]
                )
                assert (
                    x.shape == (int(row[branch + "_rows"]), dim)
                    and x.dtype == np.float32
                    and np.isfinite(x).all()
                )
                assert (
                    len(mask) == len(x)
                    and time.shape == (len(x), 2)
                    and (time.dtype == np.float64)
                )
                assert (
                    mask.shape in ((len(x),), (len(x), dim))
                    and np.isin(mask, [0, 1]).all()
                )
                time = time.copy()
                if branch == "video":
                    absent = np.all(time == -1, axis=1)
                    if absent.any():
                        assert not np.any(mask[absent]) and np.all(
                            z["coverage"][absent] == 0
                        ), "Unknown video time must be invalid with zero coverage"
                    time[absent] = np.nan
                finite = np.isfinite(time).all(axis=1)
                unknown = np.isnan(time).all(axis=1)
                assert np.all(
                    finite | unknown
                ), "Time must be a finite pair or an unknown NaN pair"
                assert np.all(time[finite, 0] >= 0) and np.all(
                    time[finite, 1] > time[finite, 0]
                )
                assert np.all(
                    time[finite, 1] <= float(row["duration_s"]) + 1 / 16000 + 1e-07
                )
                valid = np.any(mask, axis=1) if mask.ndim == 2 else mask.astype(bool)
                if branch != "text":
                    assert np.all(
                        ~valid | finite
                    ), "Valid nontext feature has no physical time"
    print(
        "PASS: 100 samples, 500 complete feature arrays, hashes, shapes, masks and clocks readable."
    )


if __name__ == "__main__":
    main()
