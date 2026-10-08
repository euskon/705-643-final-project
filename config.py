"""Single place for every knob of the Horseshoe Canyon Ranch proof of concept.

All distances are in metres in UTM zone 15N (EPSG:32615). The USGS EPT tiles
are stored in Web Mercator (EPSG:3857), whose "metres" are stretched by
1 / cos(36 deg) ~= 1.24 at this latitude, so we always reproject before doing
any geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
PRETRAINED_DIR = PROJECT_DIR / "pretrained"

WGS84 = "EPSG:4326"
EPT_CRS = "EPSG:3857"
WORK_CRS = "EPSG:32615"  # UTM 15N, true metres around Jasper, AR

FT_TO_M = 0.3048

# USGS 3DEP Entwine Point Tile datasets that cover the ranch (from
# https://github.com/hobuinc/usgs-lidar boundaries). The 2017 North Corridor
# collection is newer and denser; the 2010 Buffalo River one is a fallback.
EPT_URLS = {
    "AR_North_Corridor_2017": (
        "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/"
        "USGS_LPC_AR_North_Corridor_B1_2017_LAS_2018/ept.json"
    ),
    "AR_BuffaloRiver_2010": (
        "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/AR_BuffaloRiver_2010/ept.json"
    ),
}

# Mountain Project area id of "Horseshoe Canyon Ranch" (North-Central Arkansas)
MP_AREA_ID = 105903004
# Route pins that sit exactly on the ranch's own area pin were never given a
# real wall location by their submitters, so they cannot be placed on rock
MP_DEFAULT_PIN = (36.0118, -93.2922)

# LAS classes dropped while streaming: 7 = low noise, 18 = high noise
NOISE_CLASSES = (7, 18)


@dataclass
class RegionConfig:
    """Lat/lon bounding box of the study area and where its LiDAR comes from."""

    name: str = "horseshoe_canyon_ranch"
    # (min_lon, min_lat, max_lon, max_lat) - hugs the 33 MP wall pins with a
    # ~150 m margin so the contextual negatives have room to land
    bbox_lonlat: tuple[float, float, float, float] = (
        -93.2995,
        35.9985,
        -93.2825,
        36.0135,
    )
    ept_name: str = "AR_North_Corridor_2017"
    # the EPT is fetched in square tiles of this size so no single HTTP
    # request has to return the whole region
    tile_size_m: float = 250.0

    @property
    def ept_url(self) -> str:
        return EPT_URLS[self.ept_name]

    @property
    def cache_dir(self) -> Path:
        return DATA_DIR / self.name


@dataclass
class LabelConfig:
    """How Mountain Project routes become climbable / non-climbable patches."""

    # local-PCA neighbourhood used for surface normals
    normal_k: int = 12
    # normals are computed on a copy thinned to one point per voxel of this
    # size, so their scale doesn't depend on point density (flight-line
    # overlap strips have ~2x the points and otherwise show as fake steep
    # stripes); None disables the thinning
    normal_voxel_m: float | None = 0.5
    # a point is "steep" (part of a cliff face) above this slope in degrees
    steep_slope_deg: float = 60.0
    # the 2017 collection has no vegetation class (everything is 1 or 2) and
    # ~58% of steep-normal points are multi-return, i.e. tree canopy; a laser
    # pulse that hits bare rock returns once, so only count single returns as
    # cliff
    steep_single_return_only: bool = True
    # MP wall pins are ~10-30 m off; look this far for the cliff line
    snap_search_radius_m: float = 30.0
    # minimum spacing between neighbouring route cylinders on one wall
    route_spacing_m: float = 3.0
    # radius of the cylinder whose points belong to a route (the label region)
    route_radius_m: float = 2.5
    # a route cylinder with fewer steep points than this inside its height
    # range is "unseen by the LiDAR" and is dropped (and reported)
    min_steep_points: int = 8
    # how far to step away from a route, along the wall normal, for the
    # contextual negatives (cliff top plateau and talus at the base)
    stepback_m: tuple[float, ...] = (12.0, 20.0)
    # random negatives must be at least this far from every MP wall pin and
    # every positive (unknown band around crags is ignored, not labelled 0)
    ignore_buffer_m: float = 40.0
    # random negatives that are mostly cliff are probably unlisted rock rather
    # than true negatives, so they are ignored too
    max_steep_fraction_random_neg: float = 0.25
    n_random_negatives: int = 300
    # drop routes whose pin is the ranch's default pin (see MP_DEFAULT_PIN)
    drop_default_pin_routes: bool = True


@dataclass
class PatchConfig:
    """The model input: a vertical column of points around a GPS location."""

    # horizontal radius of the input column (the label cylinder is 2.5 m)
    radius_m: float = 5.0
    # every patch is resampled to exactly this many points
    num_points: int = 1024
    # patches with fewer raw points than this are skipped
    min_points: int = 64
    # "unit_sphere" matches the ModelNet40 pretraining (centre + scale to
    # radius 1); "fixed" keeps metric scale by dividing by `fixed_scale_m`
    normalize: str = "unit_sphere"
    fixed_scale_m: float = 15.0


@dataclass
class SplitConfig:
    """Spatial split: whole walls go to one split, negatives follow their wall."""

    val_frac: float = 0.2
    test_frac: float = 0.2
    seed: int = 42


@dataclass
class ExperimentConfig:
    region: RegionConfig = field(default_factory=RegionConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    patch: PatchConfig = field(default_factory=PatchConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
