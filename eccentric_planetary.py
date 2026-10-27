"""
Cycloidal drive with a one-lobe sun.

The sun is an eccentric circle: its centre is offset from the shaft by the
eccentricity, so the input has a single lobe. That cam turns inside a
cycloidal disc. The disc has one fewer lobe than the ring has pins, and
the reduction with the ring fixed is lobes:1.

Outer diameter is the housing size across the outside of the pin ring.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cycloidal_disc import cycloidal_profile, seat_disc
from planetary_cycloid import _simplify

LOBES = 16
OUTER_DIAMETER = 200.0
OUT_DIR = Path(__file__).resolve().parent


def build(lobes: int, outer_diameter: float, resolution: int = 8):
    if lobes < 3:
        raise ValueError("The disc needs at least 3 lobes.")
    if outer_diameter <= 0:
        raise ValueError("Outer diameter must be positive.")
    pins = lobes + 1
    # Pin radius is 5% of the pin-circle radius, eccentricity 0.56 of that,
    # matching the cycloidal disc generator. The housing ends at the requested diameter.
    pin_radius_ratio = 0.05
    radius = outer_diameter / (2.0 * (1.0 + 2.0 * pin_radius_ratio))
    pin_radius = pin_radius_ratio * radius
    eccentricity = 0.56 * pin_radius
    profile = cycloidal_profile(pins, eccentricity, pin_radius, radius, lobes, resolution)
    disc = seat_disc(profile, lobes, eccentricity)
    disc_radius = float(np.hypot(disc[:, 0] - eccentricity, disc[:, 1]).min())
    sun_radius = 0.28 * disc_radius
    if sun_radius <= eccentricity:
        raise ValueError("The eccentric cam does not fit inside the disc. Use more lobes or a larger diameter.")
    return {
        "disc": disc,
        "pins": pins,
        "radius": radius,
        "pin_radius": pin_radius,
        "eccentricity": eccentricity,
        "sun_radius": sun_radius,
        "outer": outer_diameter / 2.0,
        "lobes": lobes,
    }


def _dxf_pair(code, value) -> str:
    return f"{code}\n{value}\n"


def write_fusion_dxf(path: Path, parts: dict) -> None:
    """AutoCAD R12 DXF in millimetres, with the outlines thinned for Fusion."""
    lines: list[str] = []

    def add(code, value) -> None:
        lines.append(_dxf_pair(code, value))

    def circle(x, y, radius) -> None:
        add(0, "CIRCLE")
        add(8, "0")
        add(10, f"{x:.6f}")
        add(20, f"{y:.6f}")
        add(30, "0.0")
        add(40, f"{radius:.6f}")

    def point(x, y) -> None:
        add(0, "POINT")
        add(8, "0")
        add(10, f"{x:.6f}")
        add(20, f"{y:.6f}")
        add(30, "0.0")

    def polyline(points) -> None:
        pts = np.asarray(points, dtype=float)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        tolerance = max(0.05, parts["outer"] * 0.002)
        pts = _simplify(np.vstack((pts, pts[0])), tolerance)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        add(0, "POLYLINE")
        add(8, "0")
        add(66, 1)
        add(70, 1)
        for x, y in pts:
            add(0, "VERTEX")
            add(8, "0")
            add(10, f"{x:.6f}")
            add(20, f"{y:.6f}")
            add(30, "0.0")
        add(0, "SEQEND")

    add(0, "SECTION")
    add(2, "HEADER")
    add(9, "$ACADVER")
    add(1, "AC1009")
    add(9, "$INSUNITS")
    add(70, 4)
    add(0, "ENDSEC")
    add(0, "SECTION")
    add(2, "TABLES")
    add(0, "TABLE")
    add(2, "LAYER")
    add(70, 1)
    add(0, "LAYER")
    add(2, "0")
    add(70, 0)
    add(62, 7)
    add(6, "CONTINUOUS")
    add(0, "ENDTAB")
    add(0, "ENDSEC")
    add(0, "SECTION")
    add(2, "BLOCKS")
    add(0, "ENDSEC")
    add(0, "SECTION")
    add(2, "ENTITIES")

    circle(0.0, 0.0, parts["outer"])
    angles = np.linspace(0.0, 2.0 * np.pi, parts["pins"], endpoint=False)
    for angle in angles:
        circle(parts["radius"] * np.sin(angle), parts["radius"] * np.cos(angle), parts["pin_radius"])
    polyline(parts["disc"])
    circle(parts["eccentricity"], 0.0, parts["sun_radius"])
    point(0.0, 0.0)
    point(parts["eccentricity"], 0.0)

    add(0, "ENDSEC")
    add(0, "EOF")
    path.write_bytes("".join(lines).replace("\n", "\r\n").encode("ascii"))


def draw_set(ax, lobes, outer_diameter) -> str:
    ax.clear()
    parts = build(int(lobes), float(outer_diameter))
    ring = plt.Circle((0.0, 0.0), parts["outer"], color="#8d8d92", zorder=0)
    ax.add_patch(ring)
    hole = plt.Circle((0.0, 0.0), parts["radius"] + parts["pin_radius"], color="#f4f4f6", zorder=1)
    ax.add_patch(hole)
    angles = np.linspace(0.0, 2.0 * np.pi, parts["pins"], endpoint=False)
    for angle in angles:
        pin = plt.Circle(
            (parts["radius"] * np.sin(angle), parts["radius"] * np.cos(angle)),
            parts["pin_radius"],
            color="#c47b55",
            zorder=2,
        )
        ax.add_patch(pin)
    disc = parts["disc"]
    ax.fill(disc[:, 0], disc[:, 1], color="#c23b3b", zorder=3)
    ax.plot(disc[:, 0], disc[:, 1], color="#6e1d1d", linewidth=0.8, zorder=4)
    sun = plt.Circle((parts["eccentricity"], 0.0), parts["sun_radius"], color="#e6d7b8", zorder=5)
    ax.add_patch(sun)
    ax.plot(0.0, 0.0, marker="o", color="#1d4e89", markersize=4, zorder=6)
    ax.plot(parts["eccentricity"], 0.0, marker="o", color="#1d4e89", markersize=4, zorder=6)
    limit = parts["outer"] * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Eccentric cycloidal drive, {parts['lobes']}:1\n1-lobe sun, {parts['lobes']}-lobe disc, {parts['pins']} pins")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return (
        f"{parts['lobes']}:1, outer diameter {2.0 * parts['outer']:.1f} mm\n"
        f"eccentricity {parts['eccentricity']:.2f} mm"
    )


def launch_ui() -> None:
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Eccentric cycloidal drive")
    root.minsize(900, 680)
    fields = tk.Frame(root, padx=12, pady=10)
    fields.pack(side=tk.LEFT, fill=tk.Y)
    specs = (
        ("Disc lobes", "lobes", str(LOBES)),
        ("Outer diameter (mm)", "diameter", f"{OUTER_DIAMETER:g}"),
    )
    entries = {}
    for row, (label, key, default) in enumerate(specs):
        tk.Label(fields, text=label, anchor="w").grid(row=row * 2, column=0, sticky="w")
        entry = ttk.Entry(fields, width=16)
        entry.insert(0, default)
        entry.grid(row=row * 2 + 1, column=0, sticky="we", pady=(0, 10))
        entries[key] = entry

    status = tk.StringVar(value="")
    tk.Label(fields, textvariable=status, wraplength=220, justify="left").grid(row=8, column=0, sticky="w", pady=(8, 0))
    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    figure = Figure(figsize=(7, 7), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def values():
        return int(float(entries["lobes"].get())), float(entries["diameter"].get())

    def redraw(_event=None):
        try:
            message = draw_set(ax, *values())
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
        else:
            status.set(message)
            figure.tight_layout()
            canvas.draw_idle()

    def export_dxf():
        try:
            parts = build(*values())
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
            return
        path = OUT_DIR / "eccentric_planetary.dxf"
        write_fusion_dxf(path, parts)
        status.set(f"Saved {path.name} for Fusion 360 (Insert DXF, millimetres)")

    buttons = tk.Frame(fields)
    buttons.grid(row=6, column=0, sticky="we")
    ttk.Button(buttons, text="Update", command=redraw).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Export DXF", command=export_dxf).pack(side=tk.LEFT, padx=(8, 0))
    for entry in entries.values():
        entry.bind("<Return>", redraw)
    redraw()
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
