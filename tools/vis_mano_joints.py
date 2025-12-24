import argparse, json, sys, math
from pathlib import Path
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utilities import misc as mutils
from utilities import mano_fitting as mf

EDGES = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

def mano2openpose(Jm):
    # this repo's "mano order":
    # 0 wrist, 1-4 index, 5-8 middle, 9-12 pinky, 13-16 ring, 17-20 thumb
    Jo = np.zeros_like(Jm)
    Jo[0] = Jm[0]
    Jo[1:5]   = Jm[17:21]  # thumb
    Jo[5:9]   = Jm[1:5]    # index
    Jo[9:13]  = Jm[5:9]    # middle
    Jo[13:17] = Jm[13:17]  # ring
    Jo[17:21] = Jm[9:13]   # pinky
    return Jo

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

    if isinstance(data, dict) and "out" in data and isinstance(data["out"], list):
        return data["out"][-1]

    if isinstance(data, dict) and "results" in data:
        if frame is None:
            raise ValueError("all-frames json requires --frame")
        for r in data["results"]:
            if int(r.get("frame",-999999)) == int(frame):
                if not r.get("ok", True):
                    raise ValueError(f"frame={frame} ok=false: {r.get('error')}")
                out = r.get("out")
                return out if isinstance(out, dict) else out[-1]
        raise ValueError(f"frame={frame} not found")

    raise ValueError("Unknown json format (needs 'out' or 'results')")

def mano_joints_from_params(side, pose, betas, n_pose_params=15):
    _ = mf.MANOFitter()
    hand_idx = 1 if side == "right" else 0
    mano_dicts = mf.MANOFitter._mano_dicts
    m = mutils.load_mano_model(mano_dicts[hand_idx], ncomps=n_pose_params, flat_hand_mean=False)

    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    bb = np.zeros(m.betas.size, dtype=np.float64)
    bb[:min(bb.size, b.size)] = b[:min(bb.size, b.size)]
    m.betas[:] = bb

    # --- set pose robustly ---
    p = np.asarray(pose, dtype=np.float64).reshape(-1)

    # このMANOモデルが期待する pose 次元は (3 + n_pose_params) のはず
    # 例: n_pose_params=15 -> 18次元, n_pose_params=10 -> 13次元
    expected = m.pose.size

    # もし JSON が PCA成分だけ (N個) を保存しているなら、先頭に global rot 3個を足す
    if p.size == n_pose_params:
        p = np.concatenate([np.zeros(3, dtype=np.float64), p], axis=0)

    # 足りなければ0埋め、長ければ切り捨て（安全策）
    pp = np.zeros(expected, dtype=np.float64)
    pp[:min(expected, p.size)] = p[:min(expected, p.size)]
    m.pose[:] = pp


    mano_joints = mutils.mano_joints_with_fingertips(m)
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

def make_pcd(J, color_rgb):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(J, dtype=np.float64, copy=True))
    pcd.paint_uniform_color(color_rgb)
    return pcd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--n_pose_params", type=int, default=15)
    ap.add_argument("--joints_npy")
    ap.add_argument("--mode", choices=["mano_frame","cp_frame"], default="mano_frame")
    args = ap.parse_args()

    final = load_final(args.json, frame=args.frame)
    pose, betas, mTc = final["pose"], final["betas"], final["mTc"]

    Jm = mano_joints_from_params(args.side, pose, betas, n_pose_params=args.n_pose_params)

    if args.mode == "mano_frame":
        Jm_show = Jm
    else:
        T = mTc_to_mat(mTc)
        Jm_show = apply_T(np.linalg.inv(T), Jm)
    
    # 表示はOpenPose順に戻す（直感通りにする）
    Jm_show = mano2openpose(Jm_show)

    # 手首を原点にする（原点が変問題を消す）
    Jm_show = Jm_show - Jm_show[0]

    geoms = []
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    geoms.append(make_pcd(Jm_show, [0.0, 0.8, 0.0]))
    geoms.append(make_lineset(Jm_show, [0.0, 0.8, 0.0]))

    if args.joints_npy:
        A = np.load(args.joints_npy)
        Jraw = np.array(A[args.frame], dtype=np.float64)

        if args.mode == "mano_frame":
            T = mTc_to_mat(mTc)
            Jobs = apply_T(T, mutils.openpose2mano(Jraw))
        else:
            Jobs = Jraw

        geoms.append(make_pcd(Jobs, [0.0, 0.2, 1.0]))
        geoms.append(make_lineset(Jobs, [0.0, 0.2, 1.0]))

        rmse = float(np.sqrt(np.mean(np.sum((Jm_show - Jobs)**2, axis=1))))
        print("RMSE (shown frame) =", rmse)

    # デバッグ出力（ここが出てるなら draw_geometries までは来てる）
    Jmin = Jm_show.min(axis=0); Jmax = Jm_show.max(axis=0)
    print("J bounds:", Jmin, Jmax, "geoms:", len(geoms))

    o3d.visualization.draw_geometries(geoms, window_name="MANO joints viewer")

if __name__ == "__main__":
    main()
