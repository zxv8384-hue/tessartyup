# ====================== tess_art package (Jupyter version) ======================
from __future__ import annotations
from typing import Optional, List, Dict, Tuple, Callable, Any, Sequence
from dataclasses import dataclass
from pathlib import Path
import io
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage
from PIL import Image

# ----------------------------------------------------------------------
# config.py
# ----------------------------------------------------------------------
@dataclass
class RenderConfig:
    mesh_kind: str = "voronoi"
    target_cells: int = 4000
    relax_iters: int = 2
    hex_size: float = 20.0
    hexpent_split_quantile: float = 0.6
    star_threshold_sigma: float = 8.0
    star_min_separation: int = 9
    star_max_sources: int = 150
    jitter_strength: float = 0.06
    alpha_floor: float = 0.16
    seam_strength: float = 0.30
    out_size: int = 2400
    supersample: int = 2
    animate: bool = True
    anim_size: int = 520
    anim_frames: int = 28
    anim_fps: float = 14.0
    anim_twinkle: float = 0.16
    seed: Optional[int] = None

# ----------------------------------------------------------------------
# palette.py
# ----------------------------------------------------------------------
DEFAULT_STOPS = [
    (0.00, (108, 23, 201)),
    (0.33, (13, 22, 90)),
    (0.66, (255, 255, 255)),
    (1.00, (64, 224, 240)),
]

class Palette:
    def __init__(self, stops=None):
        stops = list(stops) if stops else DEFAULT_STOPS
        stops = sorted(stops, key=lambda s: s[0])
        self._positions = np.array([p for p, _ in stops], dtype=np.float64)
        self._colors = np.array([c for _, c in stops], dtype=np.float64) / 255.0

    def sample(self, t):
        t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
        r = np.interp(t, self._positions, self._colors[:, 0])
        g = np.interp(t, self._positions, self._colors[:, 1])
        b = np.interp(t, self._positions, self._colors[:, 2])
        return np.stack([r, g, b], axis=-1)

    def sample_cyclic(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        t = (theta / (2.0 * np.pi)) % 1.0
        n = len(self._colors)
        positions = np.concatenate([np.arange(n) / n, [1.0]])
        colors = np.concatenate([self._colors, self._colors[:1]], axis=0)
        r = np.interp(t, positions, colors[:, 0])
        g = np.interp(t, positions, colors[:, 1])
        b = np.interp(t, positions, colors[:, 2])
        return np.stack([r, g, b], axis=-1)

# ----------------------------------------------------------------------
# imgstats.py
# ----------------------------------------------------------------------
def sigma_clipped_stats(data, sigma=3.0, max_iters=5):
    values = data[np.isfinite(data)].ravel()
    if values.size == 0:
        return 0.0, 1.0
    for _ in range(max_iters):
        med = np.median(values)
        std = np.std(values)
        if std == 0:
            break
        keep = np.abs(values - med) <= sigma * std
        if keep.all():
            break
        values = values[keep]
        if values.size < 10:
            break
    return float(np.median(values)), float(max(np.std(values), 1e-6))

def normalize_stretch(values, background, noise, scale=4.0, percentiles=(1.0, 99.5)):
    z = (values - background) / max(noise, 1e-6)
    stretched = np.arcsinh(z / scale)
    finite = stretched[np.isfinite(stretched)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo, hi = np.percentile(finite, percentiles)
    if hi <= lo:
        hi = lo + 1.0
    out = (stretched - lo) / (hi - lo)
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, 1.0)

def local_background_map(data, block=48, smooth_sigma=1.0):
    ny, nx = data.shape
    safe = np.nan_to_num(data, nan=(np.nanmedian(data) if np.isfinite(data).any() else 0.0))
    pad_y, pad_x = (-ny) % block, (-nx) % block
    padded = np.pad(safe, ((0, pad_y), (0, pad_x)), mode="edge")
    by, bx = padded.shape[0] // block, padded.shape[1] // block
    coarse = np.median(padded.reshape(by, block, bx, block), axis=(1, 3))
    coarse = ndimage.gaussian_filter(coarse, smooth_sigma, mode="nearest")
    zoom_y, zoom_x = ny / coarse.shape[0], nx / coarse.shape[1]
    bg_map = ndimage.zoom(coarse, (zoom_y, zoom_x), order=1)[:ny, :nx]
    return bg_map

def detail_field(data, smooth_sigma=2.0):
    safe = np.nan_to_num(data, nan=np.nanmedian(data) if np.isfinite(data).any() else 0.0)
    smoothed = ndimage.gaussian_filter(safe, smooth_sigma)
    gy, gx = np.gradient(smoothed)
    grad_mag = np.hypot(gx, gy)
    med = np.median(grad_mag) + 1e-9
    d = np.log1p(grad_mag / med)
    d -= d.min()
    peak = d.max()
    if peak > 0:
        d = d / peak
    return d, gx, gy

def find_bright_sources(data, background, noise, threshold_sigma=8.0,
                        min_separation=9, max_sources=150):
    safe = np.nan_to_num(data, nan=background)
    footprint = np.ones((min_separation, min_separation), dtype=bool)
    local_max = ndimage.maximum_filter(safe, footprint=footprint, mode="nearest") == safe
    bright_enough = safe > (background + threshold_sigma * noise)
    ys, xs = np.nonzero(local_max & bright_enough)
    if len(xs) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([])
    peaks = safe[ys, xs]
    order = np.argsort(peaks)[::-1][:max_sources]
    return ys[order], xs[order], peaks[order]

# ----------------------------------------------------------------------
# mesh.py (simplified but complete)
# ----------------------------------------------------------------------
SQRT3 = np.sqrt(3.0)

class Mesh:
    @property
    def n_cells(self): raise NotImplementedError
    def assign(self, xs, ys): raise NotImplementedError

class VoronoiAdaptiveMesh(Mesh):
    def __init__(self, seeds_xy, image_shape):
        self._tree = cKDTree(seeds_xy)
        self.seeds = seeds_xy
        self.image_shape = image_shape
    @property
    def n_cells(self): return len(self.seeds)
    def assign(self, xs, ys):
        pts = np.stack([np.asarray(xs, dtype=np.float64),
                        np.asarray(ys, dtype=np.float64)], axis=-1)
        _, idx = self._tree.query(pts, k=1)
        return np.asarray(idx, dtype=np.int64)

def generate_voronoi_mesh(detail, target_cells=4000, relax_iters=2, rng=None):
    rng = rng or np.random.default_rng()
    ny, nx = detail.shape
    weights = detail + 0.08
    flat_w = weights.ravel()
    u = np.clip(rng.random(flat_w.size), 1e-12, 1.0)
    keys = -np.log(u) / flat_w
    k = min(target_cells, flat_w.size)
    idx = np.argpartition(keys, k-1)[:k]
    ys0, xs0 = np.unravel_index(idx, detail.shape)
    seeds = np.stack([xs0.astype(np.float64), ys0.astype(np.float64)], axis=-1)
    seeds += rng.uniform(-0.5, 0.5, seeds.shape)

    step = max(1, int(round(max(nx, ny) / 700)))
    gyy, gxx = np.mgrid[0:ny:step, 0:nx:step].astype(np.float64)
    gw = weights[0:ny:step, 0:nx:step].ravel()
    gpts = np.stack([gxx.ravel(), gyy.ravel()], axis=-1)

    for _ in range(relax_iters):
        tree = cKDTree(seeds)
        _, labels = tree.query(gpts, k=1)
        wsum = np.bincount(labels, weights=gw, minlength=len(seeds))
        xsum = np.bincount(labels, weights=gw * gpts[:, 0], minlength=len(seeds))
        ysum = np.bincount(labels, weights=gw * gpts[:, 1], minlength=len(seeds))
        has = wsum > 1e-12
        new_seeds = seeds.copy()
        new_seeds[has, 0] = xsum[has] / wsum[has]
        new_seeds[has, 1] = ysum[has] / wsum[has]
        seeds = new_seeds

    seeds[:, 0] = np.clip(seeds[:, 0], 0, nx-1)
    seeds[:, 1] = np.clip(seeds[:, 1], 0, ny-1)
    return VoronoiAdaptiveMesh(seeds, detail.shape)

def compute_cell_stats(image, mesh, n_cells):
    ny, nx = image.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float64)
    labels = mesh.assign(xx.ravel(), yy.ravel())
    flux = image.ravel()
    valid = np.isfinite(flux)
    labels = labels[valid]
    flux = flux[valid]
    counts = np.bincount(labels, minlength=n_cells).astype(np.float64)
    counts_safe = np.where(counts == 0, 1.0, counts)
    sums = np.bincount(labels, weights=flux, minlength=n_cells)
    mean = sums / counts_safe
    sums2 = np.bincount(labels, weights=flux**2, minlength=n_cells)
    std = np.sqrt(np.maximum(sums2 / counts_safe - mean**2, 0.0))
    return mean, std

# ----------------------------------------------------------------------
# shading.py
# ----------------------------------------------------------------------
def cell_colors(mean_flux, std_flux, background, noise, palette, rng,
                jitter_strength=0.06, alpha_floor=0.16):
    t = normalize_stretch(mean_flux, background, noise)
    shot = np.sqrt(np.clip(mean_flux - background, 0.0, None) + 1.0)
    shot_norm = shot / (np.percentile(shot, 95) + 1e-9)
    jitter = rng.normal(0.0, jitter_strength, size=t.shape) * np.clip(shot_norm, 0, 1.5)
    t_j = np.clip(t + jitter, 0.0, 1.0)
    colors = palette.sample(t_j)
    contrast = std_flux / (np.abs(mean_flux) + noise + 1e-6)
    finite = contrast[np.isfinite(contrast)]
    c_lo, c_hi = (np.percentile(finite, [5, 95]) if finite.size else (0., 1.))
    if c_hi <= c_lo: c_hi = c_lo + 1
    c_t = np.clip((np.nan_to_num(contrast) - c_lo) / (c_hi - c_lo), 0, 1)
    combo = 0.55 * c_t + 0.45 * t_j
    alpha = np.clip(alpha_floor + (1 - alpha_floor) * combo, 0, 1)
    rgba = np.concatenate([colors, alpha[:, None]], axis=1)
    return rgba, t_j

# ----------------------------------------------------------------------
# glow.py
# ----------------------------------------------------------------------
def build_glow_sources(ys, xs, peaks, background, noise, rng,
                       base_core=1.6, base_glow=14.0, flux_power=0.35):
    sources = []
    if len(xs) == 0:
        return sources
    rel = np.clip((peaks - background) / max(noise, 1e-6), 1.0, None)
    scale = rel ** flux_power
    scale = scale / (np.median(scale) + 1e-9)
    for x, y, s, peak in zip(xs, ys, scale, peaks):
        sources.append(dict(
            x=float(x), y=float(y), peak=float(peak),
            r_core=base_core * float(s),
            r_glow=base_glow * float(s),
            phase=float(rng.uniform(0, 2*np.pi)),
            rate=float(rng.uniform(0.7, 1.3)),
        ))
    return sources

def render_glow_layer(canvas_shape, sources, to_canvas, palette,
                      time_phase=0.0, twinkle=0.0):
    h, w = canvas_shape
    layer = np.zeros((h, w, 4), dtype=np.float32)
    for src in sources:
        cx, cy, cs = to_canvas(src["x"], src["y"])
        r_core = max(src["r_core"] * cs, 1e-3)
        r_glow = max(src["r_glow"] * cs, 1e-3)
        twinkle_mod = 1.0 + twinkle * np.sin(time_phase * src["rate"] + src["phase"])
        half = int(max(6, r_glow * 4))
        x0, x1 = int(max(0, cx-half)), int(min(w, cx+half))
        y0, y1 = int(max(0, cy-half)), int(min(h, cy+half))
        if x1 <= x0 or y1 <= y0: continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        r = np.hypot(xx - cx, yy - cy)
        theta = np.arctan2(yy - cy, xx - cx)
        core = np.exp(-(r / r_core)**2)
        env = np.clip(np.exp(-(r / r_glow)**2) * twinkle_mod, 0, 1)
        hue = palette.sample_cyclic(theta)
        color = hue * (1 - core[..., None]) + np.array([1.,1.,1.]) * core[..., None]
        region = layer[y0:y1, x0:x1]
        light = (color * env[..., None]).astype(np.float32)
        region[..., :3] = 1.0 - (1.0 - region[..., :3]) * (1.0 - light)
        region[..., 3] = np.clip(region[..., 3] + env, 0, 1)
        layer[y0:y1, x0:x1] = region
    return layer

# ----------------------------------------------------------------------
# render.py
# ----------------------------------------------------------------------
def fit_output_size(image_shape, out_size):
    ny, nx = image_shape
    aspect = nx / ny
    if aspect >= 1:
        return out_size, max(1, int(round(out_size / aspect)))
    else:
        return max(1, int(round(out_size * aspect))), out_size

def render(image, mesh, cell_rgba, glow_sources, palette,
           out_size=2400, supersample=2, background_rgb=None,
           seam_strength=0.30, time_phase=0.0, twinkle=0.0):
    out_w, out_h = fit_output_size(image.shape, out_size)
    if background_rgb is None:
        background_rgb = tuple(np.asarray(palette.sample(0.0)) * 0.12)
    
    ss_w, ss_h = out_w * supersample, out_h * supersample
    src_ny, src_nx = image.shape
    yy, xx = np.mgrid[0:ss_h, 0:ss_w].astype(np.float64)
    src_x = (xx + 0.5) * (src_nx / ss_w)
    src_y = (yy + 0.5) * (src_ny / ss_h)
    labels = mesh.assign(src_x.ravel(), src_y.ravel()).reshape(ss_h, ss_w)
    
    rgb = cell_rgba[labels, :3].astype(np.float32)
    alpha = cell_rgba[labels, 3].astype(np.float32)
    bg = np.array(background_rgb, dtype=np.float32)
    rgb = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
    
    # simple seam darken
    seam = np.zeros_like(labels, dtype=bool)
    seam[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    seam[:-1, :] |= labels[:-1, :] != labels[1:, :]
    rgb[seam] *= (1 - seam_strength)
    
    def to_canvas(x, y):
        return x * (ss_w / src_nx), y * (ss_h / src_ny), (ss_w / src_nx)
    
    glow_layer = render_glow_layer((ss_h, ss_w), glow_sources, to_canvas, palette,
                                   time_phase=time_phase, twinkle=twinkle)
    final = 1.0 - (1.0 - rgb) * (1.0 - glow_layer[..., :3] * glow_layer[..., 3:4])
    final = np.clip(final, 0, 1)
    
    img = Image.fromarray((final * 255 + 0.5).astype(np.uint8), mode="RGB")
    if supersample > 1:
        img = img.resize((out_w, out_h), Image.LANCZOS)
    return img

print("tess_art modules loaded successfully!")
