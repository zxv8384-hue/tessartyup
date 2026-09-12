# Generate a synthetic star field and render it
rng = np.random.default_rng(42)
ny, nx = 800, 1000
yy, xx = np.mgrid[0:ny, 0:nx]
background = 100 + 30*np.sin(xx/200)*np.cos(yy/300)
image = background.copy()

# Add some stars
for _ in range(120):
    x, y = rng.uniform(20, nx-20), rng.uniform(20, ny-20)
    flux = rng.pareto(1.5)*300 + 50
    sigma = rng.uniform(1.2, 2.5)
    image += flux * np.exp(-((xx-x)**2 + (yy-y)**2)/(2*sigma**2))

image += rng.normal(0, np.sqrt(np.clip(image, 1, None)))
image = image.astype(np.float64)

# Pipeline
cfg = RenderConfig(target_cells=2500, out_size=1200, supersample=2)
palette = Palette()
bg_level, noise = sigma_clipped_stats(image)
detail, gx, gy = detail_field(image)
mesh = generate_voronoi_mesh(detail, target_cells=cfg.target_cells, rng=rng)
mean, std = compute_cell_stats(image, mesh, mesh.n_cells)
cell_rgba, _ = cell_colors(mean, std, bg_level, noise, palette, rng)

ys, xs, peaks = find_bright_sources(image, bg_level, noise)
glow_sources = build_glow_sources(ys, xs, peaks, bg_level, noise, rng)

# Render
result = render(image, mesh, cell_rgba, glow_sources, palette,
                out_size=cfg.out_size, supersample=cfg.supersample)

display(result)
result.save("tess_art_preview.png")
print("Saved tess_art_preview.png")
