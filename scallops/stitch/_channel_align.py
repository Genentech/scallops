"""Per-channel alignment correction, applied while fusing stitched tiles.

Each corrected channel is aligned to the reference channel (the stitch alignment channel, ``--align-channel``) with
a magnification ``m`` about the tile centre and a translation ``(ty, tx)``, measured in raw tile coordinates::

    corrected(y, x) = raw(c + (1 + m) * ((y, x) - c) + (ty, tx)),  c = (H / 2, W / 2)

The alignment is composed with the radial (barrel) correction into one coordinate map per channel, so every channel
is resampled exactly once. When the correction is requested, every output channel (corrected or not) is resampled
with a quintic B-spline, which keeps about 95% of fine detail where the bilinear radial correction keeps about half.
Without the correction, stitching uses the existing bilinear radial correction unchanged.

The coordinate map of a channel depends only on the tile shape, the radial k and the channel's parameters, so it is
computed once per stitched image and reused for every tile.
"""

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from typing import Literal

import fsspec
import numpy as np
from scipy import ndimage as ndi
from skimage.util import img_as_float

logger = logging.getLogger("scallops")

PARAMS_FORMAT = "scallops-channel-alignment"
PARAMS_VERSION = 1
ALL_TOKENS = ("all", "*", "true")
SPLINE_ORDER = 5
RESAMPLING = (
    "one warp per channel (alignment composed with the radial correction), quintic B-spline "
    "(scipy.ndimage.map_coordinates, order 5, prefiltered), constant border"
)


def channel_key(name) -> str:
    """Normalised channel name for matching: text (numbers are converted), surrounding spaces removed, lower case."""
    return str(name).strip().lower()


def parse_correction_channels(
    values: Sequence[str | int] | bool | str | None,
) -> list[int] | Literal["all"] | None:
    """Parse ``--alignment-correction-channels``.

    :param values: Channel indices, or one of ``all``, ``*``, ``true`` (case-insensitive) or ``True`` for every
        channel except the reference. ``None`` or empty means no correction.
    :return: Sorted unique indices, ``"all"``, or ``None``.
    """
    if values is None or values is False:
        return None
    if values is True:
        return "all"
    if isinstance(values, (str, int)):
        values = [values]
    values = [str(v).strip() for v in values]
    if len(values) == 0:
        return None
    if any(v.lower() in ALL_TOKENS for v in values):
        if len(values) > 1:
            raise ValueError(
                "--alignment-correction-channels: use either channel indices or one of "
                f"{', '.join(ALL_TOKENS)}, not both."
            )
        return "all"
    try:
        indices = [int(v) for v in values]
    except ValueError:
        raise ValueError(
            "--alignment-correction-channels must be channel indices (0-based) or one of "
            f"{', '.join(ALL_TOKENS)}; got {' '.join(values)}."
        ) from None
    if any(i < 0 for i in indices):
        raise ValueError(
            "--alignment-correction-channels: channel indices must be >= 0."
        )
    return sorted(set(indices))


def load_params(path: str) -> dict:
    """Read and validate a channel alignment parameter file.

    :param path: Local path or fsspec URL of the JSON file.
    :return: Dict with ``path``, ``sha256``, ``tile_shape``, ``reference_channel`` (or ``None``) and ``channels``,
        mapping each channel's :func:`channel_key` to ``dict(name, m, ty, tx, same_as)``, with ``same_as`` resolved.
    """
    fs, _ = fsspec.url_to_fs(path)
    with fs.open(path, "rb") as f:
        raw = f.read()
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Channel alignment parameter file {path} is not valid JSON: {e}"
        ) from None

    def fail(msg):
        raise ValueError(f"Channel alignment parameter file {path}: {msg}")

    if not isinstance(doc, dict):
        fail("must contain a JSON object.")
    if doc.get("format") != PARAMS_FORMAT:
        fail(f'"format" must be "{PARAMS_FORMAT}".')
    if doc.get("version") != PARAMS_VERSION:
        fail(f'"version" must be {PARAMS_VERSION}.')
    tile_shape = doc.get("tile_shape")
    if (
        not isinstance(tile_shape, list)
        or len(tile_shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in tile_shape)
    ):
        fail('"tile_shape" must be [height, width] in pixels.')
    reference = doc.get("reference_channel")
    if reference is not None:
        if isinstance(reference, (dict, list, bool)):
            fail('"reference_channel" must be a channel name.')
        reference = str(reference).strip()
    entries = doc.get("channels")
    if not isinstance(entries, dict) or len(entries) == 0:
        fail('"channels" must map channel names to parameters.')

    by_key = {}
    for name, entry in entries.items():
        key = channel_key(name)
        if key in by_key:
            fail(f'channel names must be unique (case-insensitive): "{name}".')
        if not isinstance(entry, dict):
            fail(f'entry for "{name}" must be an object.')
        by_key[key] = (name, entry)

    channels = {}
    for key, (name, entry) in by_key.items():
        source_name = None
        if "same_as" in entry:
            source_name = entry["same_as"]
            if set(entry) != {"same_as"}:
                fail(
                    f'entry for "{name}" has "same_as", so it cannot also have values.'
                )
            if (
                isinstance(source_name, (dict, list, bool))
                or channel_key(source_name) not in by_key
            ):
                fail(f'"{name}": "same_as" must name another channel in the file.')
            source_name = str(source_name).strip()
            target = by_key[channel_key(source_name)][1]
            if "same_as" in target:
                fail(
                    f'"{name}": "same_as" must point to an entry with values, not another "same_as".'
                )
            entry = target
        values = {}
        for v in ("m", "ty", "tx"):
            x = entry.get(v)
            if (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
            ):
                fail(f'"{name}": "{v}" must be a finite number.')
            values[v] = float(x)
        channels[key] = dict(name=name, same_as=source_name, **values)
    return dict(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        tile_shape=tuple(tile_shape),
        reference_channel=reference,
        channels=channels,
    )


def resolve_channel_alignment(
    correction_channels: list[int] | Literal["all"],
    params: dict | None,
    channel_order: Sequence[str] | None,
    n_channels: int,
    reference_channel: int,
    output_channels: Sequence[int] | None,
    tile_shape: tuple[int, int],
) -> tuple[dict[int, dict], list[int]]:
    """Work out the parameters to apply to each channel of a stitched image.

    :param correction_channels: Parsed ``--alignment-correction-channels``.
    :param params: Loaded parameter file (:func:`load_params`), or ``None`` (every channel is estimated).
    :param channel_order: ``--alignment-correction-channel-order``: one parameter-file name per input channel
        (``-`` for a channel with no entry).
    :param n_channels: Number of input channels.
    :param reference_channel: Reference channel index (the stitch alignment channel).
    :param output_channels: Output channels, or ``None`` for all.
    :param tile_shape: Tile (height, width) in pixels.
    :return: ``(resolved, to_estimate)``: for each channel with parameter-file values, its input index ->
        ``dict(name, m, ty, tx, same_as, source)``; and the input indices of the requested channels to estimate from
        the images (no parameter file, or no entry in it).
    """
    if channel_order is None or len(channel_order) == 0:
        raise ValueError(
            "--alignment-correction-channel-order is required with --alignment-correction-channels: give the "
            "parameter-file name of every input channel, in input order (use - for a channel with no entry)."
        )
    if len(channel_order) != n_channels:
        raise ValueError(
            f"--alignment-correction-channel-order has {len(channel_order)} names, but the images have "
            f"{n_channels} channels: give one name per input channel, in input order."
        )
    channel_order = [str(n).strip() for n in channel_order]
    if not 0 <= reference_channel < n_channels:
        raise ValueError(
            f"Reference channel {reference_channel} (--align-channel) is out of range for {n_channels} channels."
        )
    outputs = (
        list(range(n_channels)) if output_channels is None else list(output_channels)
    )
    if correction_channels == "all":
        requested = [c for c in outputs if c != reference_channel]
    else:
        out_of_range = [c for c in correction_channels if c >= n_channels]
        if out_of_range:
            raise ValueError(
                f"--alignment-correction-channels: {', '.join(map(str, out_of_range))} out of range for "
                f"{n_channels} channels."
            )
        requested = []
        for c in correction_channels:
            if c == reference_channel:
                logger.warning(
                    f"Channel {c} is the alignment reference (--align-channel); it is not corrected."
                )
            elif c not in outputs:
                logger.warning(
                    f"Channel {c} is not in --output-channels; it is not corrected."
                )
            else:
                requested.append(c)

    if params is None:
        if requested:
            logger.info(
                "No --alignment-correction-params: estimating the alignment of channel(s) "
                f"{', '.join(f'{c} ({channel_order[c]})' for c in requested)} from the images of each well."
            )
        return {}, requested
    if tuple(params["tile_shape"]) != tuple(tile_shape):
        raise ValueError(
            f"Parameter file {params['path']} is for tiles of {params['tile_shape'][0]} x "
            f"{params['tile_shape'][1]} px, but these tiles are {tile_shape[0]} x {tile_shape[1]} px."
        )
    reference_name = channel_order[reference_channel]
    if params["reference_channel"] is not None and channel_key(
        reference_name
    ) != channel_key(params["reference_channel"]):
        raise ValueError(
            f"Parameter file {params['path']} is relative to reference channel "
            f'"{params["reference_channel"]}", but the reference channel (--align-channel {reference_channel}) '
            f'is "{reference_name}" in --alignment-correction-channel-order.'
        )

    resolved = {}
    missing = []
    for c in requested:
        name = channel_order[c]
        entry = params["channels"].get(channel_key(name)) if name != "-" else None
        if entry is None:
            missing.append(c)
            continue
        resolved[c] = dict(
            name=name,
            m=entry["m"],
            ty=entry["ty"],
            tx=entry["tx"],
            same_as=entry["same_as"],
            source="params_file",
        )
    if missing:
        logger.warning(
            f"No entry in parameter file {params['path']} for channel(s) "
            f"{', '.join(f'{c} ({channel_order[c]})' for c in missing)}: estimating their alignment from the images "
            "of each well."
        )
    return resolved, missing


def alignment_coords(
    shape: tuple[int, int],
    radial_correction_k: float | None,
    m: float = 0.0,
    ty: float = 0.0,
    tx: float = 0.0,
) -> np.ndarray | None:
    """Raw-tile coordinates sampled for each output pixel: the radial map, then the alignment.

    ``s(p) = c + (1 + m) * (R(p) - c) + (ty, tx)``, where ``R`` is the radial correction's inverse map
    (:func:`scallops.stitch._radial._radial_mapping`, same centre ``c = shape / 2``).

    :return: ``(2, H, W)`` float32 array of (y, x) coordinates, or ``None`` if the map is the identity.
    """
    k = radial_correction_k or 0.0
    if k == 0 and m == 0 and ty == 0 and tx == 0:
        return None
    h, w = shape
    cy, cx = h / 2, w / 2
    dy = np.arange(h, dtype=np.float64)[:, None] - cy
    dx = np.arange(w, dtype=np.float64)[None, :] - cx
    f = (1 + m) * (1 + k * (dy**2 + dx**2))
    coords = np.empty((2, h, w), dtype=np.float32)
    coords[0] = cy + dy * f + ty
    coords[1] = cx + dx * f + tx
    return coords


def channel_coords(
    shape: tuple[int, int],
    radial_correction_k: float | None,
    channels: Sequence[int],
    channel_alignment: dict[int, dict],
) -> dict[int, np.ndarray | None]:
    """Coordinate map of each channel; channels without alignment share the radial-only map."""
    radial_only = alignment_coords(shape, radial_correction_k)
    result = {}
    for c in channels:
        a = channel_alignment.get(c)
        result[c] = (
            radial_only
            if a is None
            else alignment_coords(shape, radial_correction_k, a["m"], a["ty"], a["tx"])
        )
    return result


def edge_band_width(
    shape: tuple[int, int],
    radial_correction_k: float | None,
    m: float = 0.0,
    ty: float = 0.0,
    tx: float = 0.0,
) -> tuple[int, int]:
    """Width (px) of the band at the tile edge that resampling fills with zeros (output pixels sampled from outside
    the tile), as rows (y) and columns (x), the wider of the two sides of each axis."""
    coords = alignment_coords(shape, radial_correction_k, m, ty, tx)
    if coords is None:
        return 0, 0
    h, w = shape
    rows = max(
        int((coords[0] < 0).any(axis=1).sum()),
        int((coords[0] > h - 1).any(axis=1).sum()),
    )
    cols = max(
        int((coords[1] < 0).any(axis=0).sum()),
        int((coords[1] > w - 1).any(axis=0).sum()),
    )
    return rows, cols


def check_stage_positions_only(
    channel_alignment: dict[int, dict],
    shape: tuple[int, int],
    radial_correction_k: float | None,
    image_key: str,
) -> None:
    """Stop when tiles were placed at their stage positions and a corrected channel would leave a band of missing
    pixels: without tile alignment, fusion doesn't crop, so the bands would stay in the stitched image.

    :raises ValueError: Naming the channels that would leave bands.
    """
    banded = [
        f"{c} ({a['name']})"
        for c, a in sorted(channel_alignment.items())
        if any(edge_band_width(shape, radial_correction_k, a["m"], a["ty"], a["tx"]))
    ]
    if banded:
        raise ValueError(
            f"Tiles of {image_key} could not be aligned (no tile pairs overlap enough) and were placed at their "
            f"stage positions, so nothing is cropped and correcting channel(s) {', '.join(banded)} would leave bands "
            "of missing pixels at the tile edges. Remove only these channel(s) from --alignment-correction-channels; "
            "the other requested channels leave no bands and can still be corrected."
        )


def fusion_crop(
    channel_alignment: dict[int, dict],
    shape: tuple[int, int],
    radial_correction_k: float | None,
    crop: tuple[int, int],
    max_crop: tuple[int | None, int | None],
    image_key: str,
) -> tuple[tuple[int, int], dict]:
    """Fusion crop that removes the zero bands the corrected channels leave at the tile edges.

    :param crop: SCALLOPS's fusion crop (y, x).
    :param max_crop: Per axis, the largest crop that opens no gaps between tiles (half the smallest tile overlap);
        ``None`` where any crop opens no gaps.
    :return: The crop to use (each axis at least the widest band), and a record of the values.
    :raises ValueError: If a channel's band is wider than both ``crop`` and ``max_crop`` on an axis: the tile overlap
        is too small for the correction.
    """
    allowed = [
        None if max_crop[i] is None else max(crop[i], max_crop[i]) for i in range(2)
    ]
    bands = {
        c: edge_band_width(shape, radial_correction_k, a["m"], a["ty"], a["tx"])
        for c, a in channel_alignment.items()
    }
    too_wide = []
    for c, b in sorted(bands.items()):
        over = [i for i in range(2) if allowed[i] is not None and b[i] > allowed[i]]
        if over:
            i = max(over, key=lambda i: b[i] - allowed[i])
            too_wide.append(
                f"Tile overlap is too small to correct channel {c} ({channel_alignment[c]['name']}) of {image_key}: "
                f"its alignment leaves a {b[i]} px band of missing pixels at the tile edge, but the overlap allows "
                f"cropping only {allowed[i]} px. Leave the channel out of --alignment-correction-channels or give "
                "smaller parameters for it."
            )
    if too_wide:
        raise ValueError("\n".join(too_wide))
    band = tuple(max([b[i] for b in bands.values()], default=0) for i in range(2))
    used = (max(crop[0], band[0]), max(crop[1], band[1]))
    if used != tuple(crop):
        logger.info(
            f"Crop for writing increased to {used} to remove the zero bands of the channel alignment correction "
            f"(widest band {band} px)."
        )
    return used, dict(
        scallops=list(crop), edge_band=list(band), used=list(used), max=list(max_crop)
    )


def warp_image(
    image: np.ndarray, coords: np.ndarray | None, cval: float = 0.0
) -> np.ndarray:
    """Resample a 2-D image at ``coords`` with a quintic B-spline.

    Follows :func:`scallops.stitch._radial.radial_correct`'s conventions: the result is a float image (integer
    input is scaled as by :func:`skimage.util.img_as_float`), points outside the image get ``cval``, and values are
    clipped to the range of the input and ``cval``.
    """
    image = img_as_float(image)
    if coords is None:
        return image.copy()
    out = ndi.map_coordinates(
        image,
        coords,
        order=SPLINE_ORDER,
        mode="constant",
        cval=cval,
        prefilter=True,
    )
    lo = min(float(image.min()), cval)
    hi = max(float(image.max()), cval)
    return np.clip(out, lo, hi, out=out)


def _file_channel_names(path: str, n_channels: int = 1) -> list[str | None]:
    """Channel names recorded in an image file's metadata, if any (IXM/MetaXpress ``_IllumSetting_``, else OME), for
    the file's first ``n_channels`` channels. The file is opened once."""
    import re

    from scallops.io import _create_image

    img = _create_image(path)
    try:
        attrs = img.xarray_dask_data.attrs
        xml = attrs.get("unprocessed", {}).get(270) if isinstance(attrs, dict) else None
        if isinstance(xml, str):
            m = re.search(r'<prop id="_IllumSetting_"[^>]*value="([^"]*)"', xml)
            if m:
                return [m.group(1)] * n_channels
    except Exception:  # noqa: BLE001 - metadata is optional
        pass
    result: list[str | None] = [None] * n_channels
    try:
        names = img.channel_names or []
        for i in range(min(n_channels, len(names))):
            if not str(names[i]).startswith("Channel:"):  # bioio placeholder
                result[i] = str(names[i])
    except Exception:  # noqa: BLE001
        pass
    return result


def metadata_channel_names(
    file_list: Sequence[str], attrs: dict | None, n_channels: int
) -> list[str | None]:
    """The image metadata's name of each input channel of one tile (``None`` where the metadata has none).

    Only used to cross-check ``--alignment-correction-channel-order``; it never decides which parameters apply.
    """
    names: list[str | None] = [None] * n_channels
    try:
        meta = attrs.get("file_metadata") if attrs else None
        if meta and len(file_list) > 1 and "c" in meta[0]:  # one file per channel
            seen = []
            for path, md in zip(file_list, meta):
                if md["c"] not in seen:
                    seen.append(md["c"])
                    if len(seen) <= n_channels:
                        names[len(seen) - 1] = _file_channel_names(path)[0]
        else:
            names = _file_channel_names(file_list[0], n_channels)
    except Exception as e:  # noqa: BLE001 - metadata is optional
        logger.debug(f"Could not read channel names from the image metadata: {e}")
    return names


def check_channel_order(
    channel_order: Sequence[str], metadata_names: Sequence[str | None]
) -> None:
    """Warn where --alignment-correction-channel-order and the image metadata name a channel differently."""
    for c, (given, meta) in enumerate(zip(channel_order, metadata_names)):
        if (
            meta is not None
            and str(given).strip() != "-"
            and channel_key(given) != channel_key(meta)
        ):
            logger.warning(
                f"Channel {c}: --alignment-correction-channel-order says '{given}' but the image metadata says "
                f"'{meta}'. Check the order (the names may just be written differently)."
            )


def record(
    resolved: dict[int, dict],
    params: dict | None,
    channel_order: Sequence[str],
    channel_names: Sequence[str] | None,
    reference_channel: int,
    radial_correction_k: float | None,
    metadata_names: Sequence[str | None] | None = None,
    estimation: dict | None = None,
) -> dict:
    """JSON-serializable record of the alignment applied to a stitched image.

    :param resolved: Input channel index -> ``dict(name, m, ty, tx, same_as, source[, estimate])``.
    :param estimation: Well-level information of :func:`scallops.stitch._channel_align_estimate.estimate_well`
        (settings, tile pool), if any channel was estimated.
    """
    meta = (
        list(metadata_names)
        if metadata_names is not None
        else [None] * len(channel_order)
    )
    return dict(
        model="corrected(y, x) = raw(c + (1 + m) * ((y, x) - c) + (ty, tx)); c = (H/2, W/2); px; "
        "relative to the reference channel; composed with the radial correction",
        resampling=RESAMPLING,
        radial_correction_k=radial_correction_k,
        reference_channel=dict(
            index=reference_channel,
            name=channel_order[reference_channel],
            channel_name=channel_names[reference_channel] if channel_names else None,
            metadata_name=meta[reference_channel],
        ),
        params_file=dict(path=params["path"], sha256=params["sha256"])
        if params
        else None,
        channel_identification=[
            dict(
                index=c,
                name=channel_order[c],
                channel_name=channel_names[c] if channel_names else None,
                metadata_name=meta[c],
            )
            for c in range(len(channel_order))
        ],
        channels=[
            dict(
                index=c,
                name=a["name"],
                channel_name=channel_names[c] if channel_names else None,
                metadata_name=meta[c],
                m=a["m"],
                ty=a["ty"],
                tx=a["tx"],
                same_as=a.get("same_as"),
                source=a["source"],
                estimate=a.get("estimate"),
            )
            for c, a in sorted(resolved.items())
        ],
        estimation=estimation,
    )


def write_record(path: str, data: dict) -> None:
    """Write a record as one JSON document in a single write."""
    fs, path = fsspec.url_to_fs(path)
    text = json.dumps(data, indent=2)
    with fs.open(path, "w") as f:
        f.write(text)
