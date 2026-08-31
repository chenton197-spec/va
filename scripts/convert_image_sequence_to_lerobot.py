#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TASK = (
    "Grasp the edge of the white shoe sole in the blue box with the left gripper, "
    "lift it into the air and hold still, then grasp the sole from the side with the "
    "right arm, open the left gripper, and place the sole down with the right arm. "
    "After placing, keep the right arm still. Repeat this cycle: the left arm keeps "
    "grasping the next sole from the blue box."
)
STATE_NAMES = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "R6",
    "left_gripper",
    "right_gripper",
]
CAMERAS = ("head", "left_hand", "right_hand")


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _vec(values: list) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _grip(values: list) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1, 1)


def _stats(arr: np.ndarray) -> dict[str, Any]:
    a = arr.astype(np.float64, copy=False)
    return {
        "min": a.min(axis=0).astype(np.float32).tolist(),
        "max": a.max(axis=0).astype(np.float32).tolist(),
        "mean": a.mean(axis=0).astype(np.float32).tolist(),
        "std": a.std(axis=0).astype(np.float32).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).astype(np.float32).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).astype(np.float32).tolist(),
        "count": [int(a.shape[0])],
    }


def _hardlink_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["cp", "-a", "--link", str(src), str(dst)], check=True)
    except subprocess.CalledProcessError:
        subprocess.run(["cp", "-a", str(src), str(dst)], check=True)


def _image_col(paths: list[str]) -> pa.Array:
    return pa.array(
        [{"bytes": None, "path": p} for p in paths],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )


def convert(src: Path, dst: Path, task: str) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if dst == src:
        raise SystemExit("dst must differ from src")
    info = _load_json(src / "meta" / "info.json")
    chunks_size = int(info["chunks_size"])
    data_tmpl = info["data_path"]
    episodes = _load_jsonl(src / "meta" / "episodes.jsonl")
    if not episodes:
        raise SystemExit(f"no episodes.jsonl under {src}")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    src_images = src / "images"
    if not src_images.is_dir():
        raise SystemExit(f"missing {src_images}")
    _hardlink_tree(src_images, dst / "images")

    new_episodes: list[dict[str, Any]] = []
    new_stats: list[dict[str, Any]] = []
    total_frames = 0
    kept = 0

    for ep in episodes:
        ep_id = int(ep["episode_index"])
        chunk = ep_id // chunks_size
        rel = data_tmpl.format(episode_chunk=chunk, episode_index=ep_id)
        pq_src = src / rel
        if not pq_src.is_file():
            print(f"skip missing parquet episode {ep_id}")
            continue
        table = pq.read_table(pq_src)
        data = table.to_pydict()
        t = table.num_rows
        state = np.concatenate(
            [
                _vec(data["observation.state"]),
                _grip(data["observation.left_gripper"]),
                _grip(data["observation.right_gripper"]),
            ],
            axis=1,
        )
        action = np.concatenate([state[1:], state[-1:]], axis=0)
        img_arrays = {}
        for cam in CAMERAS:
            key = f"observation.images.{cam}"
            col = data[key]
            paths = [row["path"] if isinstance(row, dict) else row[0] for row in col]
            img_arrays[key] = _image_col(paths)

        arrays: dict[str, pa.Array] = {
            "index": pa.array(np.asarray(data["index"], dtype=np.int64), type=pa.int64()),
            "episode_index": pa.array(
                np.asarray(data["episode_index"], dtype=np.int64), type=pa.int64()
            ),
            "frame_index": pa.array(
                np.asarray(data["frame_index"], dtype=np.int64), type=pa.int64()
            ),
            "timestamp": pa.array(
                np.asarray(data["timestamp"], dtype=np.float32), type=pa.float32()
            ),
            "task_index": pa.array(np.zeros(t, dtype=np.int64), type=pa.int64()),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), 16
            ),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), 16
            ),
        }
        arrays.update(img_arrays)
        out_table = pa.table(arrays)
        pq_dst = dst / rel
        pq_dst.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out_table, pq_dst)

        new_episodes.append({"episode_index": ep_id, "tasks": [task], "length": t})
        new_stats.append(
            {
                "episode_index": ep_id,
                "stats": {
                    "action": _stats(action),
                    "observation.state": _stats(state),
                },
            }
        )
        total_frames += t
        kept += 1
        print(f"episode {ep_id} {t} frames")

    features = {
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "action": {"dtype": "float32", "shape": [16], "names": STATE_NAMES},
        "observation.state": {"dtype": "float32", "shape": [16], "names": STATE_NAMES},
    }
    for cam in CAMERAS:
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        }

    out_info = {
        "codebase_version": "v2.1",
        "robot_type": "hcx_dual_arm",
        "total_episodes": kept,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": 1,
        "chunks_size": chunks_size,
        "fps": int(info["fps"]),
        "splits": {"train": f"0:{kept}"},
        "data_path": data_tmpl,
        "features": features,
    }
    _write_json(dst / "meta" / "info.json", out_info)
    _write_jsonl(dst / "meta" / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_jsonl(dst / "meta" / "episodes.jsonl", new_episodes)
    _write_jsonl(dst / "meta" / "episodes_stats.jsonl", new_stats)
    print(f"wrote {dst} episodes={kept} frames={total_frames}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("/home/casbot/ct/va/data/openarm_hcx_dual_arm_with_out_room_s"),
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("/home/casbot/ct/va/data/openarm_hcx_dual_arm_pi05"),
    )
    p.add_argument("--task", type=str, default=TASK)
    args = p.parse_args()
    convert(args.src, args.dst, args.task)


if __name__ == "__main__":
    main()
