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
    ("Hand",  "Position"),  # wrist (Manus docs: "Hand" joint is the wrist joint; CSV is world-space joints)  # see note below
    ("Thumb_CMC",  "Position"),
    ("Thumb_MCP",  "Position"),
    ("Thumb_DIP",  "Position"),  # Manus uses DIP label; treat as IP
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

def col(name, kind, axis):
    # Manus export you showed: e.g. "Index_MCP_Position_X"
    # Some docs show "Index_MCP_X" (older header), so support both.
    a = f"{name}_{kind}_{axis}"
    b = f"{name}_{axis}"
    return a, b

def pick_col(df, name, kind, axis):
    a, b = col(name, kind, axis)
    if a in df.columns:
        return a
    if b in df.columns:
        return b
    raise KeyError(f"column not found: {a} (or {b})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--unit", choices=["m","cm","mm","auto"], default="auto",
                    help="CSV position unit. auto: infer from median wrist-index_tip distance.")
    ap.add_argument("--skiprows", type=int, default=0, help="if CSV has extra header lines")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, skiprows=args.skiprows)
    # build (T,21,3)
    T = len(df)
    J = np.zeros((T, 21, 3), dtype=np.float64)

    for ji, (name, kind) in enumerate(OPENPOSE21):
        for ai, axis in enumerate(["X","Y","Z"]):
            c = pick_col(df, name, kind, axis)
            J[:, ji, ai] = df[c].to_numpy(dtype=np.float64)

    # unit handling (Manus CSV unit is configurable; enforce meters for ContactPose)
    # infer using wrist-index_tip distance
    d = np.linalg.norm(J[:,0,:] - J[:,8,:], axis=1)  # wrist to index_tip
    med = float(np.median(d))
    scale = 1.0

    if args.unit == "m":
        scale = 1.0
    elif args.unit == "cm":
        scale = 0.01
    elif args.unit == "mm":
        scale = 0.001
    else:
        # auto: pick scale so med is in a plausible range [0.10, 0.25] m
        # if it's ~10..25 -> cm, if ~100..250 -> mm
        if med > 5.0:         # likely mm
            scale = 0.001
        elif med > 0.5:       # likely cm
            scale = 0.01
        else:
            scale = 1.0

    J *= scale

    # sanity report
    d2 = np.linalg.norm(J[:,0,:] - J[:,8,:], axis=1)
    print("frames:", T)
    print("median wrist-index_tip distance (m):", float(np.median(d2)))
    print("min/max (m):", float(d2.min()), float(d2.max()))
    print("scale applied:", scale)

    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, J)
    print("saved:", args.out_npy)

if __name__ == "__main__":
    main()
