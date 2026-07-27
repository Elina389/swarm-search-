"""
Fetch real-world obstacle layouts (buildings, water, forest, etc.) from
OpenStreetMap and rasterize them onto a grid_size x grid_size cell grid,
for use as SwarmCoverageEnv obstacles.

This lets the swarm search a real place -- a campus, a park, a city block --
instead of a randomly scattered grid.

Requires: osmnx, geopandas, shapely (installed together via `pip install osmnx`).

Usage:
    from osm_obstacles import obstacles_from_bbox

    # bbox is (west, south, east, north) in lon/lat degrees, EPSG:4326
    obstacles, cell_bounds = obstacles_from_bbox(
        bbox=(-122.259, 37.870, -122.253, 37.875),
        grid_size=20,
    )
"""
import hashlib
import json
import os

# Union of these tags is downloaded -- a cell becomes an obstacle if it
# overlaps ANY feature matching ANY of these tags.
DEFAULT_TAGS = {
    "building": True,
    "natural": ["water"],
    "landuse": ["forest", "wood"],
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".osm_cache")


def _cache_key(bbox, tags):
    payload = json.dumps({"bbox": bbox, "tags": tags}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _fetch_features(bbox, tags):
    """Download raw OSM feature geometries for bbox, with on-disk caching.

    Overpass (OSM's query API) is rate-limited and can be slow, so repeated
    runs against the same area should hit the cache instead of re-downloading.
    """
    import geopandas as gpd

    os.makedirs(_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(_CACHE_DIR, f"{_cache_key(bbox, tags)}.geojson")

    if os.path.exists(cache_path):
        return gpd.read_file(cache_path)

    import osmnx as ox

    gdf = ox.features_from_bbox(bbox, tags=tags)
    if not gdf.empty:
        # keep only geometry -- OSM tag columns can contain lists/mixed types
        # that don't round-trip cleanly through GeoJSON
        gdf = gdf[["geometry"]]
        gdf.to_file(cache_path, driver="GeoJSON")
    return gdf


def obstacles_from_bbox(bbox, grid_size, tags=None, use_cache=True, min_overlap=0.3):
    """
    Rasterize real OSM features within `bbox` onto a grid_size x grid_size grid.

    Parameters
    ----------
    bbox : tuple(west, south, east, north)
        Bounding box in lon/lat degrees (EPSG:4326).
    grid_size : int
        Number of rows/cols in the target grid.
    tags : dict, optional
        OSM tag filter passed to osmnx (defaults to buildings + water + forest).
    use_cache : bool
        Reuse a cached download of the same bbox/tags instead of re-querying
        the Overpass API every time.
    min_overlap : float, default 0.3
        Fraction of a cell's area that must be covered by real-world features
        before the cell counts as an obstacle. At city scale, individual
        buildings are much smaller than a grid cell, so a plain
        intersects() test would mark nearly every cell as blocked even in
        open areas (a cell can "touch" a building without being mostly
        building). Requiring a meaningful overlap fraction instead reflects
        how much of the cell is actually unusable.

    Returns
    -------
    obstacles : set of (row, col)
        Grid cells whose real-world feature overlap exceeds min_overlap.
        Row 0 is the northern edge of the bbox, consistent with how the grid
        gets rendered.
    cell_bounds : callable(row, col) -> (west, south, east, north)
        Lon/lat bounds of a given grid cell -- useful for mapping drone
        positions back onto a real map (e.g. for a Leaflet/basemap overlay).
    """
    import numpy as np
    from shapely.geometry import box as shapely_box
    from shapely import STRtree, union_all

    tags = tags or DEFAULT_TAGS
    west, south, east, north = bbox

    if use_cache:
        gdf = _fetch_features(bbox, tags)
    else:
        import osmnx as ox
        gdf = ox.features_from_bbox(bbox, tags=tags)

    lon_step = (east - west) / grid_size
    lat_step = (north - south) / grid_size

    def cell_bounds(r, c):
        cell_north = north - r * lat_step
        cell_south = cell_north - lat_step
        cell_west = west + c * lon_step
        cell_east = cell_west + lon_step
        return (cell_west, cell_south, cell_east, cell_north)

    obstacles = set()
    if gdf is not None and not gdf.empty:
        # Building a single union of every feature in the bbox and then
        # intersecting that (huge, high-vertex-count) geometry against each
        # grid cell one at a time is O(cells x total_vertices) and grinds to
        # a halt once a city-scale query returns tens of thousands of
        # buildings. Instead, index individual geometries with an STRtree so
        # each cell only ever looks at the handful of features actually near
        # it, then unions/intersects just that small subset.
        geoms = np.asarray(gdf.geometry.values)
        tree = STRtree(geoms)

        for r in range(grid_size):
            for c in range(grid_size):
                cell = shapely_box(*cell_bounds(r, c))
                candidate_idx = tree.query(cell, predicate="intersects")
                if len(candidate_idx) == 0:
                    continue
                nearby = union_all(geoms[candidate_idx])
                overlap = nearby.intersection(cell).area
                if overlap / cell.area >= min_overlap:
                    obstacles.add((r, c))

    return obstacles, cell_bounds
