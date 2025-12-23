import argparse, json, sys, math
from pathlib import Path
import numpy as np
import open3d as o3d

# make repo importable (utilities/...)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilities import misc as mutils
from utilities import mano_fitting as mf

# Your joints order in npy (same as earlier viewer)
EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

def quat_wxyz_to_R(q):
    w, x, y, z = q
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def mTc_to_mat(mTc):
    t = np.array(mTc["translation"], dtype=np.float64).reshape(3)
    q = np.array(mTc["rotation"], dtype=np.float64).reshape(4)
    R = quat_wxyz_to_R(q)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def apply_T(T4, X):
    X = np.asarray(X, dtype=np.float64)
    Xh = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
    Y = (T4 @ Xh.T).T[:, :3]
    return Y

def load_final(json_path, frame=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # single-frame: {"out":[init, final]}
    if isinstance(data, dict) and "out" in data and isinstance(data["out"], list):
        last = data["out"][-1]
        if not isinstance(last, dict):
            raise ValueError("single-frame json: out[-1] is not dict")
        return last

    # all-frames: {"results":[{"frame":i,"ok":...,"out":...}, ...]}
    if isinstance(data, dict) and "results" in data:
        if frame is None:
            raise ValueError("all-frames json requires --frame")
        for r in data["results"]:
            if int(r.get("frame",-999999)) == int(frame):
                if not r.get("ok", True):
                    raise ValueError(f"frame={frame} ok=false: {r.get('error')}")
                out = r.get("out")
                if isinstance(out, dict):
                    return out
                if isinstance(out, list) and len(out) and isinstance(out[-1], dict):
                    return out[-1]
                raise ValueError("results[].out is neither dict nor list-of-dict")
        raise ValueError(f"frame={frame} not found in results")

    raise ValueError("Unknown json format (needs 'out' or 'results')")

def mano_joints_from_params(side, pose, betas, n_pose_params=15):
    # initialize fitter (sets class vars)
    _ = mf.MANOFitter()
    hand_idx = 1 if side == "right" else 0
    mano_dicts = getattr(mf.MANOFitter, "_mano_dicts", None)
    if mano_dicts is None:
        raise RuntimeError("MANOFitter._mano_dicts is None")

    m = mutils.load_mano_model(mano_dicts[hand_idx], ncomps=n_pose_params, flat_hand_mean=False)

    # set betas safely
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    bb = np.zeros(m.betas.size, dtype=np.float64)
    bb[:min(bb.size, b.size)] = b[:min(bb.size, b.size)]
    m.betas[:] = bb

    # set pose safely
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    pp = np.zeros(m.pose.size, dtype=np.float64)
    pp[:min(pp.size, p.size)] = p[:min(pp.size, p.size)]
    m.pose[:] = pp

    mano_joints = mutils.mano_joints_with_fingertips(m)  # list of 21 vec3
    J = np.array([[float(v[0]), float(v[1]), float(v[2])] for v in mano_joints], dtype=np.float64)
    return J

def make_lineset(J, color_rgb):
    lines = np.array(EDGES, dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(J, dtype=np.float64, copy=True)),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.paint_uniform_color(color_rgb)
    return ls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--n_pose_params", type=int, default=15)
    ap.add_argument("--joints_npy", help="元の関節も重ねて表示（任意）")
    ap.add_argument("--mode", choices=["mano_frame","cp_frame"], default="mano_frame",
                    help="mano_frame: fit内部の座標系で重ねる / cp_frame: 元座標系で重ねる")
    args = ap.parse_args()

    final = load_final(args.json, frame=args.frame)
    pose = final.get("pose")
    betas = final.get("betas")
    mTc = final.get("mTc")

    if pose is None or betas is None or mTc is None:
        raise ValueError(f"jsonに pose/betas/mTc がありません。keys={list(final.keys())}")

    # MANO joints (in MANO model space)
    J_mano = mano_joints_from_params(args.side, pose, betas, n_pose_params=args.n_pose_params)

    # CP joints (optionally) and transform to match mode
    geoms = []
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    geoms.append(cf)

    # always show MANO joints in green
    if args.mode == "mano_frame":
        Jm_show = J_mano
    else:
        # map MANO -> CP by inv(mTc) (assuming mTc is CP->MANO used inside fit)
        T = mTc_to_mat(mTc)
        Jm_show = apply_T(np.linalg.inv(T), J_mano)

    geoms.append(make_lineset(Jm_show, [0.0, 0.8, 0.0]))

    if args.joints_npy:
        A = np.load(args.joints_npy)
        assert A.ndim == 3 and A.shape[1:] == (21,3), A.shape
        J_raw = np.array(A[args.frame], dtype=np.float64, copy=True)

        T = mTc_to_mat(mTc)

        if args.mode == "mano_frame":
            # パターンA: そのまま mTc をかける（= すでにmano順の可能性）
            J_A = apply_T(T, J_raw)

            # パターンB: openpose2mano してから mTc をかける（= openpose順の可能性）
            J_B = apply_T(T, mutils.openpose2mano(J_raw))

            # 3色で同時表示
            geoms.append(make_lineset(J_A, [0.9, 0.0, 0.0]))  # 赤: A
            geoms.append(make_lineset(J_B, [0.0, 0.2, 1.0]))  # 青: B

            # どっちが近いかRMSEも表示
            def rmse(X, Y):
                return float(np.sqrt(np.mean(np.sum((X - Y)**2, axis=1))))
            print("RMSE vs MANO (A raw->mTc):", rmse(Jm_show, J_A))
            print("RMSE vs MANO (B openpose2mano->mTc):", rmse(Jm_show, J_B))

        else:
            # cp_frame（元座標系）ではそのまま表示
            geoms.append(make_lineset(J_raw, [0.9, 0.0, 0.0]))


if __name__ == "__main__":
    main()
