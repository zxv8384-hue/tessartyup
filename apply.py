# ========== Apply tess_art to the loaded FITS ==========

# Make sure 'image' is your loaded flux array
print(f"Working on image shape: {image.shape}")

# Configuration
cfg = RenderConfig(
    target_cells = 3500,      # increase for more detail (slower)
    out_size     = 1800,      # final image size
    supersample  = 2,
    star_threshold_sigma = 7.0,
    star_max_sources = 180,
)

rng = np.random.default_rng(42)
palette = Palette()

# 1. Background & noise
bg_level, noise = sigma_clipped_stats(image)
print(f"Background: {bg_level:.2f}  |  Noise: {noise:.2f}")

# 2. Detail field + adaptive mesh
detail, gx, gy = detail_field(image)
mesh = generate_voronoi_mesh(detail, target_cells=cfg.target_cells, relax_iters=2, rng=rng)
print(f"Mesh cells: {mesh.n_cells}")

# 3. Per-cell statistics & colors
mean, std = compute_cell_stats(image, mesh, mesh.n_cells)
cell_rgba, _ = cell_colors(mean, std, bg_level, noise, palette, rng,
                           jitter_strength=0.05, alpha_floor=0.14)

# 4. Bright sources + glows
ys, xs, peaks = find_bright_sources(
    image, bg_level, noise,
    threshold_sigma=cfg.star_threshold_sigma,
    max_sources=cfg.star_max_sources
)
glow_sources = build_glow_sources(ys, xs, peaks, bg_level, noise, rng)
print(f"Glow sources: {len(glow_sources)}")

# 5. Render final image
result = render(
    image, mesh, cell_rgba, glow_sources, palette,
    out_size=cfg.out_size,
    supersample=cfg.supersample,
    seam_strength=0.28
)

display(result)
result.save("tess_art_from_fits.png")
print("✅ Saved: tess_art_from_fits.png")
