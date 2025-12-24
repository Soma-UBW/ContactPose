import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# OpenPose 21 order:
# 0 wrist
# 1-4 thumb: CMC, MCP, IP(DIP), TIP
# 5-8 index: MCP, PIP, DIP, TIP
# 9-12 middle
# 13-16 ring
# 17-20 pinky
OPENPOSE21 = [
    ("Hand",       "Position"),  # wrist-like (your CSV has Hand_Position_X/Y/Z)
    ("Thumb_CMC",  "Position"),
    ("Thumb_MCP",  "Position"),
    ("Thumb_DIP",  "Position"),  # Manus: DIP label; treat as IP
    ("Thumb_TIP",  "Position"),
    ("Index_MCP",  "Position"),
    ("Index_PIP",  "Position"),
    ("Index_DIP",  "Position"),
    ("Index_TIP",  "Position"),
    ("Middle_MCP", "Position"),
    ("Middle_PIP", "Position"),
    ("Middle_DIP", "Position"),
    ("Middle_TIP", "Position"),
    ("Ring_MCP",   "Position"),
    ("Ring_PIP",   "Position"),
    ("Ring_DIP",   "Position"),
    ("Ring_TIP",   "Position"),
    ("Pinky_MCP",  "Position"),
    ("Pinky_PIP",  "Position"),
    ("Pinky_DIP",  "Position"),
    ("Pinky_TIP",  "Position"),
]

def _candidate_cols(name: str, kind: str, axis: str):
    # Your header style: "Index_MCP_Position_X"
    a = f"{name}_{kind}_{axis}"
    # fallback: some exports may be "Index_MCP_X"
    b = f"{name}_{axis}"
    return (a, b)

def _pick_col(df: pd.DataFrame, name: str, kind: str, axis: str) -> str:
    for c in _candidate_cols(name, kind, axis):
        if c in df.columns:
            return c
    a, b = _candidate_cols(name, kind, axis)
    raise KeyError(f"column not found: {a} (or {b})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--skiprows", type=int, default=0, help="CSV has extra header lines")
    ap.add_argument("--unit", choices=["m", "cm", "mm", "auto"], default="auto",
                    help="CSV position unit. auto: infer from median wrist-index_tip distance.")
    # ---- mirror / flip options ----
    ap.add_argument("--flip_x", action="store_true", help="mirror: X *= -1")
    ap.add_argument("--flip_y", action="store_true", help="mirror: Y *= -1")
    ap.add_argument("--flip_z", action="store_true", help="mirror: Z *= -1")
    ap.add_argument("--make_right", action="store_true",
                    help="convenience: mirror to behave like right-hand (currently equals --flip_x)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, skiprows=args.skiprows)
    T = len(df)
    J = np.zeros((T, 21, 3), dtype=np.float64)

    # Build joints
    for ji, (name, kind) in enumerate(OPENPOSE21):
        for ai, axis in enumerate(["X", "Y", "Z"]):
            c = _pick_col(df, name, kind, axis)
            J[:, ji, ai] = df[c].to_numpy(dtype=np.float64)

    # Unit → meters
    d = np.linalg.norm(J[:, 0, :] - J[:, 8, :], axis=1)  # wrist to index_tip
    med = float(np.median(d))
    scale = 1.0
    if args.unit == "m":
        scale = 1.0
    elif args.unit == "cm":
        scale = 0.01
    elif args.unit == "mm":
        scale = 0.001
    else:
        # auto: choose scale so median wrist-index_tip is plausible in meters
        # typical ~0.10..0.25 m
        if med > 5.0:      # likely mm
            scale = 0.001
        elif med > 0.5:    # likely cm
            scale = 0.01
        else:
            scale = 1.0
    J *= scale

    # Mirror / flip
    flip_x = args.flip_x or args.make_right
    flip_y = args.flip_y
    flip_z = args.flip_z
    if flip_x:
        J[:, :, 0] *= -1.0
    if flip_y:
        J[:, :, 1] *= -1.0
    if flip_z:
        J[:, :, 2] *= -1.0

    # Report
    d2 = np.linalg.norm(J[:, 0, :] - J[:, 8, :], axis=1)
    print("frames:", T)
    print("median wrist-index_tip distance (m):", float(np.median(d2)))
    print("min/max (m):", float(d2.min()), float(d2.max()))
    print("scale applied:", scale)
    print("flip:", {"x": bool(flip_x), "y": bool(flip_y), "z": bool(flip_z)})

    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, J)
    print("saved:", args.out_npy)

if __name__ == "__main__":
    main()
