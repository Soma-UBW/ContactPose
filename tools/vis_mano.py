import argparse, json, os, sys
from pathlib import Path
import numpy as np

# ---- make repo importable (utilities/...) even after reboot ----
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import open3d as o3d

def quat_wxyz_to_R(q):
    # q = [w, x, y, z]
    w, x, y, z = q
    # normalize
    n = (w*w + x*x + y*y + z*z) ** 0.5
    if n == 0:
        return np.eye(3)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

def mTc_dict_to_matrix(mTc):
    # {"translation":[tx,ty,tz], "rotation":[w,x,y,z]}
    t = np.array(mTc["translation"], dtype=np.float64)
    q = np.array(mTc["rotation"], dtype=np.float64)
    R = quat_wxyz_to_R(q)
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def load_final_param_from_json(json_path, frame=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # case1: single-frame json from fit_one_frame.py
    if "out" in data and isinstance(data["out"], list):
        # usually [init, final]
        final = data["out"][-1] if isinstance(data["out"][-1], dict) else None
        if final is None:
            raise ValueError("single-frame json の out から final dict を取れませんでした")
        return final

    # case2: all-frames json from fit_all_frames_mp.py etc.
    if "results" in data:
        if frame is None:
            raise ValueError("all-frames json には --frame が必要です")
        cand = [r for r in data["results"] if int(r.get("frame",-1)) == int(frame)]
        if not cand:
            raise ValueError(f"frame={frame} が json.results に見つかりません")
        r = cand[0]
        if not r.get("ok", True):
            raise ValueError(f"frame={frame} は ok=false: {r.get('error')}")
        out = r.get("out")
        # out は dict (only_final) のはず。listの場合は最後を取る
        if isinstance(out, dict):
            return out
        if isinstance(out, list) and len(out) > 0 and isinstance(out[-1], dict):
            return out[-1]
        raise ValueError("results[].out から final dict を取れませんでした")

    raise ValueError("未知のJSON形式です（out または results が必要）")

def build_mano_mesh(side, pose, betas, n_pose_params=15, trans=None):
    from utilities import misc as mutils
    from utilities import mano_fitting as mf  # ← クラス初期化の挙動差に備えてこちらから取る

    # 重要：MANOFitterを一度生成して、クラス変数(_mano_dicts等)の初期化を走らせる
    _ = mf.MANOFitter()

    hand_idx = 1 if side == "right" else 0
    mano_dicts = getattr(mf.MANOFitter, "_mano_dicts", None)
    if mano_dicts is None:
        raise RuntimeError("MANOFitter._mano_dicts が初期化されていません。MANOFitterの初期化ロジックを確認してください。")

    m = mutils.load_mano_model(mano_dicts[hand_idx],
                               ncomps=n_pose_params, flat_hand_mean=False)

    m.betas[:] = np.array(betas, dtype=np.float64)

    # pose 長さが n_pose_params と一致しないケースに備えて切り詰め/パディング
    pose = np.array(pose, dtype=np.float64).reshape(-1)
    if pose.size != m.pose.size:
        pp = np.zeros(m.pose.size, dtype=np.float64)
        n = min(pp.size, pose.size)
        pp[:n] = pose[:n]
        pose = pp
    m.pose[:] = pose

    if trans is not None and hasattr(m, "trans"):
        tr = np.array(trans, dtype=np.float64).reshape(-1)
        if tr.size == 3:
            m.trans[:] = tr

    V = np.array(m.r, dtype=np.float64)
    F = np.array(m.f, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V)
    mesh.triangles = o3d.utility.Vector3iVector(F)
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, help="fit結果json（single frame / all framesどちらも可）")
    ap.add_argument("--joints_npy", help="元関節 (T,21,3) を重ねて表示したい場合")
    ap.add_argument("--frame", type=int, default=0, help="all-frames json のとき必要")
    ap.add_argument("--side", choices=["right","left"], default="right")
    ap.add_argument("--n_pose_params", type=int, default=15)
    ap.add_argument("--show_mano_frame", action="store_true",
                    help="cp座標ではなく、fit内部で使ったMANO座標系のまま表示（mTcの逆変換をしない）")
    args = ap.parse_args()

    final = load_final_param_from_json(args.json, frame=args.frame)

    pose = final.get("pose")
    betas = final.get("betas")
    mTc = final.get("mTc")
    trans = final.get("trans", None)

    if pose is None or betas is None or mTc is None:
        raise ValueError(f"jsonに pose/betas/mTc がありません: keys={list(final.keys())}")

    mesh = build_mano_mesh(args.side, pose, betas, n_pose_params=args.n_pose_params, trans=trans)

    # mTc は「cp joints を mano座標へ」変換したものとして使われているので、
    # 元のcp座標へ戻すには inv(mTc) をメッシュにかける。
    if not args.show_mano_frame:
        T = mTc_dict_to_matrix(mTc)
        Tinv = np.linalg.inv(T)
        mesh.transform(Tinv)

    geoms = [mesh]

    # overlay joints
    if args.joints_npy:
        Jall = np.load(args.joints_npy)
        J = np.array(Jall[args.frame], dtype=np.float64)

        # jointsもmanoフレーム表示に合わせるなら、同じ変換をかける
        if not args.show_mano_frame:
            # jointsは元cp座標のままなので変換なし
            pass
        else:
            # manoフレームで見たい場合は mTc を joints に適用
            T = mTc_dict_to_matrix(mTc)
            Jh = np.concatenate([J, np.ones((J.shape[0],1))], axis=1)
            Jm = (T @ Jh.T).T[:, :3]
            J = Jm

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(J)
        pcd.paint_uniform_color([1.0, 0.0, 0.0])  # 赤
        geoms.append(pcd)

    o3d.visualization.draw_geometries(geoms)

if __name__ == "__main__":
    main()
