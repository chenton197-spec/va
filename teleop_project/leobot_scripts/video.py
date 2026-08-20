"""LeRobot v2.x compatible episode video encoding."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from av.video.stream import VideoStream


def encode_episode_video(image_dir: Path, output_path: Path, fps: int) -> None:
    """Encode sequential PNG frames with the historical v2.x defaults.

    The output is AV1 in an MP4 container: ``libsvtav1``, ``yuv420p``, GOP 2,
    CRF 30, and no fast-decode tuning. PyAV stays an optional dependency until
    a camera source is actually registered.
    """

    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Video collection requires the 'av' and 'Pillow' packages. "
            "Install leobot_scripts/requirements-video.txt."
        ) from exc

    images = sorted(image_dir.glob("frame_*.png"))
    if not images:
        raise ValueError(f"No PNG frames available for video encoding in {image_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(images[0]) as first:
        width, height = first.size

    options = {"g": "2", "crf": "30"}
    with av.open(str(output_path), "w") as container:
        # PyAV's ``str`` overload includes audio and subtitle streams even
        # though libsvtav1 is always a video encoder.
        stream = cast("VideoStream", container.add_stream("libsvtav1", fps, options=options))
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height
        for image_path in images:
            with Image.open(image_path) as image:
                frame = av.VideoFrame.from_image(image.convert("RGB"))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    if not output_path.is_file():
        raise RuntimeError(f"PyAV did not create {output_path}")
