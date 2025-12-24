import argparse, json
import numpy as np
import pandas as pd

def load_mapping(path: str):
    """
    mapping.json 例:
    {
      "format": "wide",
      "unit": "mm",
      "scale": 0.001,
      "joints": [
        {"x":"wrist_x","y":"wrist_y","z":"wrist_z"},
        ...
      ]
    }
    joints は length=21 必須
    """
    with open(path, "r", encoding="utf-8") as f:
        mp = json.load(f)
    if "joints" not in mp or len(mp["joints"]) != 21:
        raise ValueError("mapping.json の joints は長さ21にしてください")
    mp.setdefault("format", "wide")
    mp.setdefault("scale", 1.0)
    return mp

def wide_csv_to_joints(df: pd.DataFrame, joints_map, scale: float):
    T = len(df)
    out = np.zeros((T, 21, 3), dtype=np.float32)
    for j in range(21):
        xm, ym, zm = joints_map[j]["x"], joints_map[j]["y"], joints_map[j]["z"]
        missing = [c for c in (xm,ym,zm) if c not in df.columns]
        if missing:
            raise KeyError(f"CSVに列がありません joint{j}: {missing}")
        out[:, j, 0] = df[xm].to_numpy(dtype=np.float32) * scale
        out[:, j, 1] = df[ym].to_numpy(dtype=np.float32) * scale
        out[:, j, 2] = df[zm].to_numpy(dtype=np.float32) * scale
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--out_npy", required=True)
    args = ap.parse_args()

    mp = load_mapping(args.mapping)
    df = pd.read_csv(args.csv)

    if mp["format"] != "wide":
        raise ValueError("今はformat=wideのみ対応（1行=1フレームで joint0_x... のような列構造）")

    joints = wide_csv_to_joints(df, mp["joints"], float(mp["scale"]))
    np.save(args.out_npy, joints)
    print("saved:", args.out_npy, "shape:", joints.shape)

if __name__ == "__main__":
    main()
