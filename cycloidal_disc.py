"""
Cycloidal disc, 20:1.

Profile method copied from mikedh/cycloidal (BSD), which implements
equations 10 and 11 of Chen, BingKui et al., "Gear geometry of cycloid
drives". The four inputs are the same ones that generator uses:

    Zb  number of pins
    Zg  number of lobes on the disc (Zb - 1 for a single-stage drive)
    Rz  radius of the pin-centre circle
    rz  pin radius
    e   eccentricity

    K1 = e * Zb / (Rz * (Zb - Zg))
    Ze = Zb / (Zb - Zg)
    Zd = Zg / (Zb - Zg)

    B  = sqrt(1 + K1^2 - 2 K1 cos(Zd * psi))
    cos B = sign(Zb - Zg) * (K1 sin(Ze psi) - sin psi) / B
    sin B = sign(Zb - Zg) * (-K1 cos(Ze psi) + cos psi) / B

    x = Rz sin psi - e sin(Ze psi) + rz cos B
    y = Rz cos psi - e cos(Ze psi) - rz sin B

K1 must be below 1 or the flanks cross. Pin radius and eccentricity are
scaled from that project's example (rz / Rz = 0.05, e = 0.56 rz).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 21 pins, 20 lobes -> 20:1. Sizes follow mikedh/cycloidal's example ratio.
PIN_COUNT = 21
PIN_PATTERN_RADIUS = 50.0          # Rz, mm
PIN_RADIUS = 50.0 * (0.125 / 2.5)  # rz
ECCENTRICITY = PIN_RADIUS * 0.56   # e
POINTS_PER_DEGREE = 16

OUT_DIR = Path(__file__).resolve().parent


def cycloidal_profile(count_pin, eccentricity, radius_pin, radius_pattern, count_cam=None, resolution=16):
    """Return an (n, 2) disc outline. Same construction as mikedh/cycloidal."""
    rz = float(radius_pattern)
    pin_r = float(radius_pin)
    eccentricity = float(eccentricity)
    pin_count = int(count_pin)
    lobe_count = pin_count - 1 if count_cam is None else int(count_cam)

    tooth_ratio = pin_count / (pin_count - lobe_count)
    lobe_ratio = lobe_count / (pin_count - lobe_count)
    shortening = (eccentricity * pin_count) / (rz * (pin_count - lobe_count))
    if shortening >= 1.0:
        raise ValueError(f"K1 = {shortening:.3f} must be below 1 or the flanks cross.")

    psi = np.linspace(0.0, 2.0 * np.pi, int(resolution * 360), endpoint=False)
    flank = np.sqrt(1.0 + shortening**2 - 2.0 * shortening * np.cos(lobe_ratio * psi))
    direction = np.sign(pin_count - lobe_count)
    cos_b = direction * (shortening * np.sin(tooth_ratio * psi) - np.sin(psi)) / flank
    sin_b = direction * (-shortening * np.cos(tooth_ratio * psi) + np.cos(psi)) / flank
    x = rz * np.sin(psi) - eccentricity * np.sin(tooth_ratio * psi) + pin_r * cos_b
    y = rz * np.cos(psi) - eccentricity * np.cos(tooth_ratio * psi) - pin_r * sin_b
    return np.column_stack((np.append(x, x[0]), np.append(y, y[0])))


def seat_disc(profile: np.ndarray, lobes: int, eccentricity: float) -> np.ndarray:
    """Rotate half a lobe and shift by +e so every pin is tangent to a flank."""
    half_lobe = 0.5 * np.pi / lobes
    cosine, sine = np.cos(half_lobe), np.sin(half_lobe)
    return np.column_stack((
        cosine * profile[:, 0] - sine * profile[:, 1] + eccentricity,
        sine * profile[:, 0] + cosine * profile[:, 1],
    ))


def draw_disc(ax, pin_count, pin_pattern_radius, pin_radius, eccentricity) -> str:
    """Draw the seated disc and return a one-line status, or an error message."""
    ax.clear()
    lobes = int(pin_count) - 1
    if lobes < 2:
        raise ValueError("Need at least 3 pins (2 lobes).")
    profile = cycloidal_profile(
        count_pin=pin_count,
        eccentricity=eccentricity,
        radius_pin=pin_radius,
        radius_pattern=pin_pattern_radius,
        count_cam=lobes,
        resolution=POINTS_PER_DEGREE,
    )
    seated = seat_disc(profile, lobes, eccentricity)
    ax.fill(seated[:, 0], seated[:, 1], color="#d9e4f0", zorder=1)
    ax.plot(seated[:, 0], seated[:, 1], color="#143d59", linewidth=1.6, label=f"{lobes}-lobe disc", zorder=2)
    angles = np.linspace(0.0, 2.0 * np.pi, int(pin_count), endpoint=False)
    for i, angle in enumerate(angles):
        pin = plt.Circle(
            (pin_pattern_radius * np.sin(angle), pin_pattern_radius * np.cos(angle)),
            pin_radius,
            fill=False,
            color="#b85c38",
            linewidth=1.1,
            zorder=3,
        )
        ax.add_patch(pin)
        if i == 0:
            pin.set_label(f"{int(pin_count)} pins")
    ax.plot(0.0, 0.0, "o", color="#b85c38", markersize=5, label="ring centre")
    ax.plot(eccentricity, 0.0, "+", color="black", markersize=10, markeredgewidth=1.4, label="disc centre")
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    shortening = eccentricity * pin_count / pin_pattern_radius
    radius = np.hypot(profile[:, 0], profile[:, 1])
    ax.set_title(f"Cycloidal disc, {lobes}:1     K1 = {shortening:.3f}")
    ax.legend(loc="upper right")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    limit = pin_pattern_radius + 3.0 * pin_radius
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    return f"{lobes}:1 reduction, lobe height {radius.max() - radius.min():.2f} mm, K1 = {shortening:.3f}"


def launch_ui() -> None:
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    root = tk.Tk()
    root.title("Cycloidal disc")
    root.minsize(860, 640)

    fields = tk.Frame(root, padx=12, pady=10)
    fields.pack(side=tk.LEFT, fill=tk.Y)

    specs = (
        ("Pins", "pins", f"{PIN_COUNT:g}", "Lobes = pins − 1, so the reduction is that many to 1."),
        ("Pin circle radius (mm)", "radius", f"{PIN_PATTERN_RADIUS:g}", "Distance from the ring centre to a pin centre."),
        ("Pin radius (mm)", "pin", f"{PIN_RADIUS:.4g}", "Radius of each ring roller."),
        ("Eccentricity (mm)", "eccentricity", f"{ECCENTRICITY:.4g}", "Input-shaft offset. K1 = e × pins / radius must stay below 1."),
    )
    entries = {}
    for row, (label, key, default, hint) in enumerate(specs):
        tk.Label(fields, text=label, anchor="w").grid(row=row * 2, column=0, sticky="w")
        entry = ttk.Entry(fields, width=18)
        entry.insert(0, default)
        entry.grid(row=row * 2 + 1, column=0, sticky="we", pady=(0, 10))
        entries[key] = entry

    status = tk.StringVar(value="")
    tk.Label(fields, textvariable=status, wraplength=220, justify="left").grid(row=9, column=0, sticky="w", pady=(8, 0))

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    figure = Figure(figsize=(6.5, 6.5), dpi=100)
    ax = figure.add_subplot(111)
    canvas = FigureCanvasTkAgg(figure, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def read_values():
        pins = int(float(entries["pins"].get()))
        return (
            pins,
            float(entries["radius"].get()),
            float(entries["pin"].get()),
            float(entries["eccentricity"].get()),
        )

    def redraw(_event=None):
        try:
            message = draw_disc(ax, *read_values())
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
        else:
            status.set(message)
            figure.tight_layout()
            canvas.draw_idle()

    def save_files():
        try:
            pins, radius, pin_radius, eccentricity = read_values()
            lobes = pins - 1
            profile = cycloidal_profile(
                count_pin=pins,
                eccentricity=eccentricity,
                radius_pin=pin_radius,
                radius_pattern=radius,
                count_cam=lobes,
                resolution=POINTS_PER_DEGREE,
            )
        except (ValueError, ZeroDivisionError) as error:
            status.set(str(error))
            return
        csv_path = OUT_DIR / "cycloidal_disc.csv"
        png_path = OUT_DIR / "cycloidal_disc.png"
        np.savetxt(csv_path, profile, delimiter=",", header="x_mm,y_mm", comments="")
        figure.savefig(png_path, dpi=160)
        status.set(f"Saved {csv_path.name} and {png_path.name}")

    buttons = tk.Frame(fields)
    buttons.grid(row=8, column=0, sticky="we")
    ttk.Button(buttons, text="Update", command=redraw).pack(side=tk.LEFT)
    ttk.Button(buttons, text="Save CSV and PNG", command=save_files).pack(side=tk.LEFT, padx=(8, 0))
    for entry in entries.values():
        entry.bind("<Return>", redraw)

    redraw()
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
