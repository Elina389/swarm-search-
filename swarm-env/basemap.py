"""
Fetch a real basemap image (street/satellite tiles) for a lon/lat bounding
box, no API key required -- pulls free OpenStreetMap (or other provider)
tiles via contextily. This is purely visual: it goes *behind* the existing
SwarmCoverageEnv grid rendering in visualize_swarm.py, it doesn't touch
obstacles/coverage logic at all (see osm_obstacles.py for that).

Requires: contextily (pip install contextily).
"""
import hashlib
import os

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".osm_cache")


def get_basemap(bbox, source=None, use_cache=True):
    """
    Fetch a real basemap image for bbox=(west, south, east, north).

    Parameters
    ----------
    bbox : tuple(west, south, east, north)
        Lon/lat bounding box (EPSG:4326) -- same bbox you'd pass to
        SwarmCoverageEnv(bbox=...) / osm_obstacles.obstacles_from_bbox.
    source : contextily provider, optional
        Defaults to OpenStreetMap's standard street map. Try
        `contextily.providers.Esri.WorldImagery` for satellite instead.
    use_cache : bool
        Reuse a cached download of the same bbox/source (tiles servers are
        rate-limited and this is purely cosmetic, no need to refetch).

    Returns
    -------
    img : numpy.ndarray, shape (H, W, 4)
        RGBA image ready to hand to matplotlib's imshow.
    """
    import contextily as cx
    import numpy as np

    os.makedirs(_CACHE_DIR, exist_ok=True)
    key = hashlib.sha1(f"{bbox}-{source}".encode()).hexdigest()[:16]
    cache_path = os.path.join(_CACHE_DIR, f"basemap_{key}.npy")

    if use_cache and os.path.exists(cache_path):
        return np.load(cache_path)

    west, south, east, north = bbox
    provider = source or cx.providers.OpenStreetMap.Mapnik
    img, _extent = cx.bounds2img(west, south, east, north, ll=True, source=provider)

    if use_cache:
        np.save(cache_path, img)
    return img
