import argparse
import numpy as np
import open3d as o3d

# joints order (your mapping):
# 0 wrist
# 1-4 thumb: CMC, MCP, DIP, TIP
# 5-8 index: MCP, PIP, DIP, TIP
# 9-12 middle
# 13-16 ring
# 17-20 pinky

EDGES = [
    # wrist to each MCP
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),

    # thumb chain
    (1, 2), (2, 3), (3, 4),

    # index chain
    (5, 6), (6, 7), (7, 8),

    # middle chain
    (9, 10), (10, 11), (11, 12),

    # ring chain
    (13, 14), (14, 15), (15, 16),

    # pinky chain
    (17, 18), (18, 19), (19, 20),
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--every", type=int, default=1, help="間引き表示(デバッグ用)")
    ap.add_argument("--flip_x", action="store_true")
    args = ap.parse_args()

    A = np.load(args.npy)
    assert A.ndim == 3 and A.shape[1:] == (21, 3), A.shape

    f = max(0, min(args.frame, A.shape[0]-1))
    J = np.array(A[f], dtype=np.float64, copy=True)
    
    if args.flip_x:
        J[:, 0] *= -1

    # points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(J)
    pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # lines
    lines = np.array(EDGES, dtype=np.int32)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(J),
        lines=o3d.utility.Vector2iVector(lines)
    )
    ls.paint_uniform_color([0.0, 0.0, 0.0])

    # coordinate frame for orientation
    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    o3d.visualization.draw_geometries([pcd, ls, cf])

if __name__ == "__main__":
    main()
