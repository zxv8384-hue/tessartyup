# ---------- Animated GIF with gentle star twinkling ----------
from IPython.display import display, Image as IPImage
import io

def build_preview_gif(image, mesh, cell_rgba, glow_sources, palette,
                      size=520, n_frames=28, fps=14.0, twinkle=0.16):
    out_w, out_h = fit_output_size(image.shape, size)
    background_rgb = tuple(np.asarray(palette.sample(0.0)) * 0.12)
    
    # Pre-render the static mesh layer once
    ss = 2
    base = render(image, mesh, cell_rgba, glow_sources, palette,
                  out_size=size, supersample=ss, time_phase=0.0, twinkle=0.0)
    
    frames = []
    for i in range(n_frames):
        phase = 2.0 * np.pi * i / n_frames
        frame = render(image, mesh, cell_rgba, glow_sources, palette,
                       out_size=size, supersample=ss,
                       time_phase=phase, twinkle=twinkle)
        frames.append(frame.convert("P", palette=Image.ADAPTIVE, colors=220))
    
    # Save to bytes for display + optional file
    buf = io.BytesIO()
    duration_ms = int(round(1000.0 / fps))
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0)
    buf.seek(0)
    
    # Also write to disk
    with open("tess_art_preview.gif", "wb") as f:
        f.write(buf.getvalue())
    
    print("Saved tess_art_preview.gif")
    return IPImage(data=buf.getvalue())

# Generate and display the animation
gif = build_preview_gif(image, mesh, cell_rgba, glow_sources, palette,
                        size=520, n_frames=24, fps=12, twinkle=0.18)
display(gif)
