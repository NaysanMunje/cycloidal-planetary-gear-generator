"""
Planetary set of cycloidal gears.

Sun and planets are cycloidal discs from the same Chen profile as
cycloidal_disc.py. The ring bore is a larger cycloidal profile, so the
planet lobes sit in its pockets. Pitch radius scales with the lobe count,
and the ring has sun + 2 * planet lobes, which is the count that lets a
planet mesh with the sun and the ring at the same centre distance.

Defaults follow the reference layout: six planets around a sun of the
same size, inside a lobed ring, with a bore in every gear.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ring fixed, sun driving, carrier output. The ratio is (sun lobes + ring lobes) / sun lobes.
# With the current tooth counts that is (9 + 21) / 9.
GEAR_RATIO = (9 + 21) / 9
SUN_LOBES = 9
PLANET_COUNT = 6
# Diameter across the outermost ring lobes. That is the overall size of the set.
OUTER_DIAMETER = 200.0
BORE_FRACTION = 0.42         # bore radius / root radius
POINTS = 6

OUT_DIR = Path(__file__).resolve().parent


def ring_lobes(sun_lobes: int, planet_lobes: int) -> int:
    return sun_lobes + 2 * planet_lobes


def overall_ratio(sun_lobes: int, planet_lobes: int) -> float:
    """Sun turns per carrier turn with the ring held fixed."""
    return (sun_lobes + ring_lobes(sun_lobes, planet_lobes)) / sun_lobes


def choose_planet_lobes(sun_lobes: int, gear_ratio: float, planet_count: int) -> int:
    """Planet lobes for this sun that come closest to the ratio without the planets crossing."""
    if sun_lobes < 4:
        raise ValueError("Sun lobes must be at least 4.")
    if gear_ratio <= 2.0:
        raise ValueError("Gear ratio must be greater than 2. Below that the planets would have no teeth.")
    if planet_count < 3:
        raise ValueError("Planet count must be at least 3.")
    best = None
    gap = np.sin(np.pi / planet_count)
    for planet in range(3, sun_lobes * 4):
        ring = ring_lobes(sun_lobes, planet)
        if (sun_lobes + ring) % planet_count != 0:
            continue
        # Tooth tips stay inside half the gap between planet centres.
        if (planet + 1) >= (sun_lobes + planet) * gap * 0.98:
            continue
        actual = overall_ratio(sun_lobes, planet)
        error = abs(actual - gear_ratio)
        if best is None or error < best[0]:
            best = (error, planet)
    if best is None:
        raise ValueError(
            f"A {sun_lobes}-lobe sun cannot reach a ratio near {gear_ratio:g} "
            f"with {planet_count} planets without the outer gears crossing."
        )
    return best[1]


def module_for_outer_diameter(sun_lobes: int, planet_lobes: int, diameter: float) -> float:
    """Module that puts the outermost ring lobe at this diameter."""
    if diameter <= 0:
        raise ValueError("Outer diameter must be positive.")
    ring_count = ring_lobes(sun_lobes, planet_lobes)
    # At module 1 the ring pitch radius is the ring lobe count.
    unit = epi_hypo_gear(float(ring_count), ring_count, 0.5, samples=24)
    return diameter / (2.0 * float(radii(unit).max()))


def _rotation(angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, sine], [-sine, cosine]])


def _trochoid_part(radius: float, rolling: float, sign: float, samples: int) -> np.ndarray:
    """One epi (sign +1) or hypo (sign -1) flank. Same construction as pygeartrain."""
    teeth = radius / rolling
    angle = np.linspace(0.0, 2.0 * np.pi / teeth, samples, endpoint=False)
    rolled = (radius + rolling * sign) / rolling * angle * sign
    basis = np.array([[np.cos(angle), np.cos(rolled)], [np.sin(angle), np.sin(rolled)]])
    return np.dot([radius + rolling * sign, -rolling * sign], basis).T


def epi_hypo_gear(pitch_radius: float, teeth: int, epi_fraction: float = 0.5, samples: int = 48) -> np.ndarray:
    """Closed cycloidal gear: epicycloid addendum, hypocycloid dedendum."""
    rolling = pitch_radius / teeth
    tooth = 2.0 * np.pi / teeth
    epi = _trochoid_part(pitch_radius, rolling * epi_fraction, +1.0, samples) @ _rotation(-tooth * epi_fraction / 2.0)
    hypo = _trochoid_part(pitch_radius, rolling * (1.0 - epi_fraction), -1.0, samples) @ _rotation(tooth * epi_fraction / 2.0)
    one = np.vstack((epi, hypo))
    return np.vstack([one @ _rotation(tooth * i) for i in range(teeth)])


def transform(profile: np.ndarray, rotation: float, center: tuple[float, float]) -> np.ndarray:
    cosine, sine = np.cos(rotation), np.sin(rotation)
    x = cosine * profile[:, 0] - sine * profile[:, 1] + center[0]
    y = sine * profile[:, 0] + cosine * profile[:, 1] + center[1]
    return np.column_stack((x, y))


def radii(profile: np.ndarray) -> np.ndarray:
    return np.hypot(profile[:, 0], profile[:, 1])


def radius_at(profile: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Radius of a star-shaped outline at each angle."""
    angle = np.arctan2(profile[:, 1], profile[:, 0])
    order = np.argsort(angle)
    angle = angle[order]
    radius = np.hypot(profile[order, 0], profile[order, 1])
    query = (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi
    return np.interp(query, angle, radius, period=2.0 * np.pi)


def flank_gap(moved: np.ndarray, boundary: np.ndarray, internal: bool) -> float:
    """Smallest radial gap. Negative means the profiles cross."""
    theta = np.arctan2(moved[:, 1], moved[:, 0])
    wall = radius_at(boundary, theta)
    radial = radii(moved)
    if internal:
        return float((wall - radial).min())
    return float((radial - wall).min())


def assembly(sun_lobes: int, planet_lobes: int, planet_count: int, module: float, resolution: int = POINTS, carrier: float = 0.0):
    """
    Epi/hypo cycloidal planetary, scaled so the planet orbit equals
    the sum of the sun and planet pitch radii. That is the distance at
    which the flanks stay tangent for every carrier angle.
    """
    if min(sun_lobes, planet_lobes, planet_count) < 3:
        raise ValueError("Sun lobes, planet lobes, and planet count must each be at least 3.")
    ring_count = ring_lobes(sun_lobes, planet_lobes)
    if (sun_lobes + ring_count) % planet_count != 0:
        raise ValueError(
            f"Planet count {planet_count} must divide sun + ring "
            f"({sun_lobes} + {ring_count} = {sun_lobes + ring_count}) "
            "or the planets cannot all stay in mesh."
        )
    samples = max(24, int(resolution) * 8)
    # Pitch radius = module * teeth. Orbit radius = sun pitch + planet pitch.
    orbit = module * (sun_lobes + planet_lobes)
    tooth_scale = orbit  # their generator uses an orbit radius of 1
    # The sun's addendum is the planet's dedendum, and the other way around,
    # so a tooth of one sits in the space of the other instead of crossing it.
    fraction = 0.5
    sun = epi_hypo_gear(tooth_scale * sun_lobes / (sun_lobes + planet_lobes), sun_lobes, 1.0 - fraction, samples)
    planet = epi_hypo_gear(tooth_scale * planet_lobes / (sun_lobes + planet_lobes), planet_lobes, fraction, samples)
    ring = epi_hypo_gear(tooth_scale * ring_count / (sun_lobes + planet_lobes), ring_count, fraction, samples)
    # Even planet counts need the sun turned half a tooth so a lobe meets a valley.
    sun_turn = (sun_lobes + ring_count) / sun_lobes * carrier
    if planet_lobes % 2 == 0:
        sun_turn += np.pi / sun_lobes
    sun = transform(sun, sun_turn, (0.0, 0.0))
    planets = []
    for index in range(planet_count):
        angle = carrier + 2.0 * np.pi * index / planet_count
        # Ring fixed. This spin, together with the sun turn above, keeps every
        # planet tangent to the sun and to the ring at every carrier angle.
        spin = -(ring_count - planet_lobes) / planet_lobes * angle
        planets.append(transform(planet, spin, (orbit * np.cos(angle), orbit * np.sin(angle))))
    sun_gaps = [flank_gap(profile, sun, internal=False) for profile in planets]
    ring_gaps = [flank_gap(profile, ring, internal=True) for profile in planets]
    return {
        "sun": sun,
        "planets": planets,
        "ring": ring,
        "distance": orbit,
        "ring_lobes": ring_count,
        "bore": BORE_FRACTION * float(radii(planet).min()),
        "sun_bore": BORE_FRACTION * float(radii(sun).min()),
        "sun_gaps": sun_gaps,
        "ring_gaps": ring_gaps,
    }


def build_parts(gear_ratio, outer_diameter, planet_count, sun_lobes):
    sun_lobes = int(sun_lobes)
    planet_count = int(planet_count)
    planet_lobes = choose_planet_lobes(sun_lobes, float(gear_ratio), planet_count)
    module = module_for_outer_diameter(sun_lobes, planet_lobes, float(outer_diameter))
    parts = assembly(sun_lobes, planet_lobes, planet_count, module)
    return parts, sun_lobes, planet_lobes


def _dxf_pair(code, value) -> str:
    return f"{code}\n{value}\n"


def write_fusion_dxf(path: Path, parts: dict) -> None:
    """AutoCAD R12 DXF, millimetres. Fusion: Insert > Insert DXF."""
    lines: list[str] = []

    def add(code, value) -> None:
        lines.append(_dxf_pair(code, value))

    def polyline(points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        # Drop duplicates so Fusion does not reject a zero-length segment.
        keep = [pts[0]]
        for point in pts[1:]:
            if np.hypot(point[0] - keep[-1][0], point[1] - keep[-1][1]) > 1e-6:
                keep.append(point)
        add(0, "POLYLINE")
        add(8, "0")
        add(66, 1)
        add(70, 1)
        for x, y in keep:
            add(0, "VERTEX")
            add(8, "0")
            add(10, f"{x:.6f}")
            add(20, f"{y:.6f}")
            add(30, "0.0")
        add(0, "SEQEND")

    def centre_point(x: float, y: float) -> None:
        add(0, "POINT")
        add(8, "0")
        add(10, f"{float(x):.6f}")
        add(20, f"{float(y):.6f}")
        add(30, "0.0")

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

    outer = float(radii(parts["ring"]).max())
    add(0, "CIRCLE")
    add(8, "0")
    add(10, "0.0")
    add(20, "0.0")
    add(30, "0.0")
    add(40, f"{outer:.6f}")
    polyline(parts["ring"])
    centre_point(0.0, 0.0)
    polyline(parts["sun"])
    centre_point(*parts["sun"].mean(axis=0))
    for profile in parts["planets"]:
        polyline(profile)
        centre_point(*profile.mean(axis=0))

    add(0, "ENDSEC")
    add(0, "EOF")
    path.write_bytes("".join(lines).replace("\n", "\r\n").encode("ascii"))


def draw_set(ax, gear_ratio, outer_diameter, planet_count, sun_lobes) -> str:
    ax.clear()
    parts, sun_lobes, planet_lobes = build_parts(gear_ratio, outer_diameter, planet_count, sun_lobes)
    ring = parts["ring"]
    outer = float(radii(ring).max())
    wall = plt.Circle((0.0, 0.0), outer, color="#8d8d92", zorder=0)
    ax.add_patch(wall)
    ax.fill(ring[:, 0], ring[:, 1], color="#e6e6ea", zorder=1)
    ax.plot(ring[:, 0], ring[:, 1], color="#4a4a4e", linewidth=1.0, zorder=2)

    def gear_body(profile, face="#6e6e73"):
        ax.fill(profile[:, 0], profile[:, 1], color=face, zorder=3)
        ax.plot(profile[:, 0], profile[:, 1], color="#2c2c2e", linewidth=0.8, zorder=4)
        centre = profile.mean(axis=0)
        ax.plot(*centre, marker="o", color="#1d4e89", markersize=4, zorder=6)
        return centre

    gear_body(parts["sun"], face="#7a7a80")
    for profile in parts["planets"]:
        gear_body(profile)

    limit = outer * 1.05
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ratio = overall_ratio(sun_lobes, planet_lobes)
    ax.set_title(
        f"Cycloidal planetary set, ratio {ratio:.3f}:1\n"
        f"sun {sun_lobes},  {int(planet_count)} planets × {planet_lobes},  ring {parts['ring_lobes']}"
    )
    ax.grid(True, linewidth=0.3, alpha=0.4)
    return (
        f"ratio {ratio:.3f}:1, outer diameter {2.0 * outer:.1f} mm\n"
        f"sun {sun_lobes}, planets {planet_lobes}, ring {parts['ring_lobes']}"
    )


def launch_ui() -> None:
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Cycloidal planetary set")
    root.minsize(980, 700)

    fields = tk.Frame(root, padx=12, pady=10)
    fields.pack(side=tk.LEFT, fill=tk.Y)
    specs = (
        ("Gear ratio", "ratio", f"{GEAR_RATIO:.4g}"),
        ("Sun lobes", "sun", str(SUN_LOBES)),
        ("Outer diameter (mm)", "diameter", f"{OUTER_DIAMETER:g}"),
        ("Planet count", "count", str(PLANET_COUNT)),
    )
    entries = {}
    for row, (label, key, default) in enumerate(specs):
        tk.Label(fields, text=label, anchor="w").grid(row=row * 2, column=0, sticky="w")
        entry = ttk.Entry(fields, width=16)
        entry.insert(0, default)
        entry.grid(row=row * 2 + 1, column=0, sticky="we", pady=(0, 10))
        entries[key] = entry

    status = tk.StringVar(value="")
    tk.Label(fields, textvariable=status, wraplength=220, justify="left").grid(row=10, column=0, sticky="w", pady=(8, 0))

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    figure = Figure(figsize=(7, 7), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def redraw(_event=None):
        try:
            message = draw_set(
                ax,
                float(entries["ratio"].get()),
                float(entries["diameter"].get()),
                int(float(entries["count"].get())),
                int(float(entries["sun"].get())),
            )
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
        else:
            status.set(message)
            figure.tight_layout()
            canvas.draw_idle()

    def save_png():
        redraw()
        path = OUT_DIR / "planetary_cycloid.png"
        figure.savefig(path, dpi=160)
        status.set(f"Saved {path.name}")

    def export_dxf():
        try:
            parts, _, _ = build_parts(
                float(entries["ratio"].get()),
                float(entries["diameter"].get()),
                int(float(entries["count"].get())),
                int(float(entries["sun"].get())),
            )
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
            return
        path = OUT_DIR / "planetary_cycloid.dxf"
        write_fusion_dxf(path, parts)
        status.set(f"Saved {path.name} for Fusion 360 (Insert DXF, millimetres)")

    buttons = tk.Frame(fields)
    buttons.grid(row=9, column=0, sticky="we")
    ttk.Button(buttons, text="Update", command=redraw).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Save PNG", command=save_png).pack(side=tk.LEFT, padx=(8, 0))
    ttk.Button(buttons, text="Export DXF", command=export_dxf).pack(side=tk.LEFT, padx=(8, 0))
    for entry in entries.values():
        entry.bind("<Return>", redraw)

    redraw()
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
