import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import cv2
from tqdm import tqdm
import sys
# Reuse road loading and community parsing from the existing visualizer
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from unified_community_visualization import UnifiedCommunityVisualizer
from shapely.geometry import box as shapely_box
import geopandas as gpd


DEFAULT_IMG_SIZE = 1024  # 1024 x 1024
DEFAULT_RADIUS_M = 500


def _init_square_fig(ax_size_px: int) -> Tuple[Figure, Axes]:
    """Create a white 1024x1024 canvas with equal aspect and no axes.

    We avoid bbox_inches/tight_layout to preserve exact pixel size and aspect.
    """
    inches = ax_size_px / 100.0  # dpi=100 → pixels = inches * 100
    fig = plt.figure(figsize=(inches, inches), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])  # full canvas
    ax.set_facecolor("white")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig, ax


def _classify_road_style(highway: str, colors: Dict[str, str]) -> Tuple[str, float, float]:
    hwy = (highway or "").lower()
    if any(t in hwy for t in ["motorway", "trunk", "primary", "secondary"]):
        return colors["road_major"], 2.5, 0.9
    if any(t in hwy for t in ["tertiary", "unclassified", "residential", "living_street"]):
        return colors["road_minor"], 1.5, 0.8
    return colors["road_other"], 1.2, 0.9


def draw_roads(ax: Axes, roads: List[Dict], colors: Dict[str, str]) -> None:
    for seg in roads:
        coords = seg["coords"]
        color, lw, alpha = _classify_road_style(seg.get("highway", ""), colors)
        if len(coords) < 2:
            continue
        xs, ys = zip(*coords)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", zorder=1)


def draw_buildings(ax: Axes, context_blocks: List[Dict], fill_color: str) -> None:
    for blk in context_blocks:
        for bldg in blk.get("buildings", []):
            verts = bldg.get("vertices", [])
            if not verts:
                continue
            rel = [(x, y) for x, y, _ in verts]
            if len(rel) < 3:
                continue
            px = [p[0] for p in rel] + [rel[0][0]]
            py = [p[1] for p in rel] + [rel[0][1]]
            ax.fill(px, py, color=fill_color, alpha=0.95, zorder=4)


def render_pair(viz: UnifiedCommunityVisualizer,
                community_id: str,
                radius_m: int = DEFAULT_RADIUS_M,
                building_fill_color: str = "#1E90FF") -> Tuple[np.ndarray, np.ndarray]:
    """Render control (roads only) and target (roads + buildings) images as numpy arrays (H,W,3).

    - White background
    - Exact 1024x1024
    - Equal aspect, no distortion
    - Road color scheme: major=red, minor=blue, other=gray (taken from visualizer colors)
    - Buildings filled with provided blue color
    """
    data = viz.load_community(community_id)
    cx, cy = data["source_info"]["center_utm"]
    city = data["source_info"]["city"]
    # Fast road loading with spatial pre-filter to avoid scanning entire shapefile
    roads = fast_load_roads(viz.road_data_dir, city, (cx, cy), radius_m, viz)

    # Control (roads only)
    fig_c, ax_c = _init_square_fig(DEFAULT_IMG_SIZE)
    # bounds: fixed symmetric square to radius
    margin = 0.0
    ax_c.set_xlim(-radius_m - margin, radius_m + margin)
    ax_c.set_ylim(-radius_m - margin, radius_m + margin)
    draw_roads(ax_c, roads, viz.colors)
    fig_c.canvas.draw()
    cw, ch = fig_c.canvas.get_width_height()
    control_rgba = np.frombuffer(fig_c.canvas.buffer_rgba(), dtype=np.uint8)
    control_rgba = control_rgba.reshape((ch, cw, 4))
    control = control_rgba[..., :3].copy()
    plt.close(fig_c)

    # Target (roads + buildings)
    fig_t, ax_t = _init_square_fig(DEFAULT_IMG_SIZE)
    ax_t.set_xlim(-radius_m - margin, radius_m + margin)
    ax_t.set_ylim(-radius_m - margin, radius_m + margin)
    draw_roads(ax_t, roads, viz.colors)
    draw_buildings(ax_t, data["community_context"]["context_blocks"], building_fill_color)
    fig_t.canvas.draw()
    tw, th = fig_t.canvas.get_width_height()
    target_rgba = np.frombuffer(fig_t.canvas.buffer_rgba(), dtype=np.uint8)
    target_rgba = target_rgba.reshape((th, tw, 4))
    target = target_rgba[..., :3].copy()
    plt.close(fig_t)

    return control, target


def to_canny(control_rgb: np.ndarray, low_thresh: int = 100, high_thresh: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(control_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    # Convert to 3-channel for convenience; edges are white on black
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return edges_rgb


def discover_communities(data_dir: Path, cities_filter: List[str] | None = None) -> List[Tuple[str, str]]:
    """Return list of (city, community_id) pairs present in data_dir.
    If cities_filter is provided, only include those cities (case-insensitive).
    """
    cities_filter_norm = set([c.lower() for c in cities_filter]) if cities_filter else None
    results: List[Tuple[str, str]] = []
    for fp in sorted(data_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            city = str(data["source_info"]["city"]).strip()
            community_id = fp.stem
            if cities_filter_norm and city.lower() not in cities_filter_norm:
                continue
            results.append((city, community_id))
        except Exception:
            continue
    return results


def fast_load_roads(osm_dir: str | Path,
                    city: str,
                    center_utm: Tuple[float, float],
                    radius: float,
                    viz_like: UnifiedCommunityVisualizer) -> List[Dict]:
    """Optimized version of road loading:
    - Reads city shapefile
    - Converts to appropriate UTM CRS
    - Spatially filters features using an axis-aligned square query window
    - Returns list of {coords, highway} in center-relative coordinates
    """
    shp = Path(osm_dir) / f"{city.lower()}_roads.shp"
    if not shp.exists():
        return []

    # Use same CRS mapping as the visualizer to stay consistent
    city_crs_mapping = {
        'atlanta': "EPSG:32616",
        'dallas': "EPSG:32614",
        'chicago': "EPSG:32616",
        'los angeles': "EPSG:32611",
        'new york': "EPSG:32618",
        'philadelphia': "EPSG:32618",
        'phoenix': "EPSG:32612",
        'san antonio': "EPSG:32614",
        'san diego': "EPSG:32611",
        'denver': "EPSG:32613",
        'houston': "EPSG:32615",
        'seattle': "EPSG:32610",
        'miami': "EPSG:32617",
    }
    target_crs = city_crs_mapping.get(city.lower(), "EPSG:32616")

    try:
        gdf = gpd.read_file(shp)
        gdf = gdf.to_crs(target_crs)
    except Exception:
        return []

    cx, cy = center_utm
    query_poly = shapely_box(cx - radius, cy - radius, cx + radius, cy + radius)

    # Spatial index filter (fallback to bounding box filter if no sindex)
    try:
        sidx = gdf.sindex
        idx = list(sidx.intersection(query_poly.bounds))
        cand = gdf.iloc[idx]
    except Exception:
        cand = gdf.cx[cx - radius: cx + radius, cy - radius: cy + radius]

    segs: List[Dict] = []
    for _, row in cand.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        # Normalize to list of LineStrings
        if hasattr(geom, "coords"):
            lines = [geom]
        elif hasattr(geom, "geoms"):
            lines = [g for g in geom.geoms if hasattr(g, "coords")]
        else:
            continue
        highway = str(row.get('highway', 'unknown')).lower()
        for ln in lines:
            coords = list(ln.coords)
            if len(coords) < 2:
                continue
            # Keep if any point lies within the square radius window
            rel = [(x - cx, y - cy) for x, y in coords]
            if any((-radius <= x <= radius) and (-radius <= y <= radius) for x, y in rel):
                segs.append({"coords": rel, "highway": highway})

    return segs


def main():
    import argparse
    parser = argparse.ArgumentParser("Prepare SDXL+ControlNet (canny) dataset from community JSONs")
    parser.add_argument("--data_dir", default="evaluation_communities_v3", type=str)
    parser.add_argument("--osm_dir", default="Dataset/osm", type=str)
    parser.add_argument("--output_dir", default="baseline/sdxl_controlnet_canny/data", type=str)
    parser.add_argument("--cities", nargs="*", default=None, help="If set, only process these cities (match shapefile names)")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS_M)
    parser.add_argument("--max_per_city", type=int, default=0, help="0 for all")
    parser.add_argument("--save_canny", action="store_true", help="Also save canny edges computed from control image")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    viz = UnifiedCommunityVisualizer(road_data_dir=args.osm_dir, data_dir=args.data_dir)

    items = discover_communities(data_dir, args.cities)
    # Restrict to cities that have shapefiles in OSM dir (handle multi-word names)
    osm_shps = set()
    for p in Path(args.osm_dir).glob("*_roads.shp"):
        name = p.name
        if name.lower().endswith("_roads.shp"):
            city_name = name[:-len("_roads.shp")]  # keep spaces, full city string
            osm_shps.add(city_name.lower())
    filtered: List[Tuple[str, str]] = []
    for city, cid in items:
        if city.lower() in osm_shps:
            filtered.append((city, cid))

    per_city_count: Dict[str, int] = {}
    with tqdm(total=len(filtered), desc="Rendering communities") as pbar:
        for city, community_id in filtered:
            if args.max_per_city:
                c = per_city_count.get(city, 0)
                if c >= args.max_per_city:
                    pbar.update(1)
                    continue

            try:
                control, target = render_pair(viz, community_id, radius_m=args.radius)
            except Exception as e:
                pbar.set_postfix_str(f"skip {community_id}: {e}")
                pbar.update(1)
                continue

            # Save under output_dir/city/COMMUNITY/
            sample_dir = output_dir / city / community_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            control_path = sample_dir / "control.png"
            target_path = sample_dir / "target.png"
            cv2.imwrite(str(control_path), cv2.cvtColor(control, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(target_path), cv2.cvtColor(target, cv2.COLOR_RGB2BGR))

            meta = {
                "city": city,
                "community_id": community_id,
                "radius_m": args.radius,
                "control": str(control_path),
                "target": str(target_path),
            }

            if args.save_canny:
                edges = to_canny(control)
                edges_path = sample_dir / "control_canny.png"
                cv2.imwrite(str(edges_path), cv2.cvtColor(edges, cv2.COLOR_RGB2BGR))
                meta["control_canny"] = str(edges_path)

            (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            per_city_count[city] = per_city_count.get(city, 0) + 1
            pbar.update(1)

    print("Done. Dataset written to:", str(output_dir))


if __name__ == "__main__":
    main()
