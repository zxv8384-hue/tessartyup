# ---------- Load a real TESS Full-Frame Image ----------
# Put the path to your .fits file here
fits_path = "path/to/your/tess_ffi.fits"   # ← change this

try:
    from astropy.io import fits
    with fits.open(fits_path) as hdul:
        # Prefer the first 2D image that is not an uncertainty/quality map
        data = None
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                name = (hdu.name or "").upper()
                if not any(tag in name for tag in ("UNCERT", "ERROR", "ERR", "QUALITY", "DQ")):
                    data = np.asarray(hdu.data, dtype=np.float64)
                    break
        if data is None:
            data = np.asarray(hdul[1].data, dtype=np.float64)
    print(f"Loaded real FFI: shape {data.shape}")
    image = data
except Exception as e:
    print("Could not load FITS (using synthetic instead):", e)
