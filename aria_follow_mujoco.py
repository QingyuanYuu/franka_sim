#!/usr/bin/env python3
import argparse
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import mujoco
import mujoco.viewer

import aria.sdk_gen2 as sdk_gen2
import aria.stream_receiver as receiver
from projectaria_tools.core.mps import hand_tracking


# -----------------------------
# Hand skeleton edges (MediaPipe-style 21 landmarks)
# -----------------------------
HAND_EDGES_21: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
]


def _try_extract_landmarks_device(hand_obj) -> Optional[np.ndarray]:
    """
    Best-effort landmark extraction (device frame), because SDK versions differ.
    Returns array (N,3) in meters, or None.
    """
    candidates = [
        "get_landmark_positions_device",
        "get_landmarks_device",
        "landmark_positions_device",
        "landmarks_device",
        "keypoints_device",
        "joints_device",
    ]
    for name in candidates:
        if hasattr(hand_obj, name):
            v = getattr(hand_obj, name)
            try:
                pts = v() if callable(v) else v
                arr = np.array(pts, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 5:
                    return arr
            except Exception:
                pass
    return None


# -----------------------------
# Shared state
# -----------------------------
@dataclass
class HandState:
    t_ns: int = 0
    valid: bool = False
    conf: float = 0.0
    wrist_dev: Optional[np.ndarray] = None    # (3,)
    palm_dev: Optional[np.ndarray] = None     # (3,)
    landmarks_dev: Optional[np.ndarray] = None # (N,3)


_lock = threading.Lock()
_hand = HandState()


def hand_cb(ht: hand_tracking.HandTrackingResult):
    """Update right-hand info in DEVICE frame."""
    global _hand
    now_ns = time.time_ns()

    r = ht.right_hand
    if r is None:
        with _lock:
            _hand.t_ns = now_ns
            _hand.valid = False
        return

    wrist = np.array(r.get_wrist_position_device(), dtype=np.float64)
    conf = float(r.confidence)

    # palm (best-effort)
    palm = None
    try:
        palm = np.array(r.get_palm_position_device(), dtype=np.float64)
    except Exception:
        palm = None

    lm = _try_extract_landmarks_device(r)

    with _lock:
        _hand.t_ns = now_ns
        _hand.valid = True
        _hand.conf = conf
        _hand.wrist_dev = wrist
        _hand.palm_dev = palm
        _hand.landmarks_dev = lm


# -----------------------------
# Receiver
# -----------------------------
def start_receiver(host: str, port: int):
    cfg = sdk_gen2.HttpServerConfig()
    cfg.address = host
    cfg.port = port

    sr = receiver.StreamReceiver(enable_image_decoding=False, enable_raw_stream=False)
    sr.set_server_config(cfg)
    sr.register_hand_pose_callback(hand_cb)

    print(f"[Receiver] Listening on {host}:{port} ... (start your device streaming)")
    sr.start_server()
    return sr


# -----------------------------
# MuJoCo visualization helpers
# -----------------------------
def _safe_make_connector(geom, radius: float, p1: np.ndarray, p2: np.ndarray) -> bool:
    """
    Try to create a capsule connector between p1 and p2.
    Some MuJoCo builds expose mjv_makeConnector; if not available, return False.
    """
    if not hasattr(mujoco, "mjv_makeConnector"):
        return False
    try:
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            float(p1[0]), float(p1[1]), float(p1[2]),
            float(p2[0]), float(p2[1]), float(p2[2]),
        )
        return True
    except Exception:
        return False


def _draw_hand_markers(viewer, points_mj: np.ndarray, edges: Optional[List[Tuple[int, int]]],
                       sphere_size: float = 0.008, edge_radius: float = 0.004):
    """
    Draw hand points (and edges if possible) into viewer.user_scn.
    NOTE: We clear and redraw each frame (simple + robust).
    """
    scn = viewer.user_scn
    scn.ngeom = 0

    # points as spheres
    for p in points_mj:
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([sphere_size, 0, 0], dtype=np.float64),
            p.astype(np.float64),
            np.eye(3).flatten(),
            np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),  # green
        )
        scn.ngeom += 1

    # edges as capsules (optional)
    if edges is None:
        return

    for i, j in edges:
        if i >= len(points_mj) or j >= len(points_mj):
            continue
        if scn.ngeom >= scn.maxgeom:
            break
        p1, p2 = points_mj[i], points_mj[j]
        g = scn.geoms[scn.ngeom]

        ok = _safe_make_connector(g, edge_radius, p1, p2)
        if not ok:
            # If connector isn't available, just skip edges (points still shown)
            return

        g.rgba[:] = np.array([0.1, 0.8, 1.0, 1.0], dtype=np.float32)  # cyan
        scn.ngeom += 1


# -----------------------------
# IK (position-only) with Damped Least Squares
# -----------------------------
def ik_step_pos(model, data, site_id, target_pos, damping=1e-2, step_scale=0.4, max_dq=0.12):
    ee = data.site_xpos[site_id].copy()
    err = (target_pos - ee).astype(np.float64)  # (3,)

    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

    JJt = jacp @ jacp.T
    A = JJt + damping * np.eye(3)
    y = np.linalg.solve(A, err)
    dq = jacp.T @ y

    dq = np.clip(dq, -max_dq, max_dq)
    mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1)
    mujoco.mj_forward(model, data)

    return float(np.linalg.norm(err))


# -----------------------------
# Axis mapping utility
# -----------------------------
def apply_axis_ops(v: np.ndarray, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    out = v.copy()
    if flip_x:
        out[0] *= -1
    if flip_y:
        out[1] *= -1
    if flip_z:
        out[2] *= -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="MuJoCo XML path (e.g. franka_panda.xml)")
    ap.add_argument("--ee-site", type=str, default="end_effector", help="EE site name (default: end_effector)")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=6768)

    # mapping / smoothing
    ap.add_argument("--scale", type=float, default=1.0, help="Scale device delta -> MJ meters")
    ap.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing (0..1); smaller=more smooth")
    ap.add_argument("--conf", type=float, default=0.5, help="Min hand confidence to accept updates")
    ap.add_argument("--timeout", type=float, default=0.25, help="Seconds without update => hold target")
    ap.add_argument("--flip-x", action="store_true", help="Flip mapped X")
    ap.add_argument("--flip-y", action="store_true", help="Flip mapped Y")
    ap.add_argument("--flip-z", action="store_true", help="Flip mapped Z (common for up/down)")

    # IK params
    ap.add_argument("--ik-damping", type=float, default=1e-2)
    ap.add_argument("--ik-step-scale", type=float, default=0.4)
    ap.add_argument("--ik-max-dq", type=float, default=0.12)

    # hand visualization
    ap.add_argument("--show-hand", action="store_true", help="Visualize hand skeleton (3D) in MuJoCo viewer")
    ap.add_argument("--hand-point-size", type=float, default=0.008, help="Hand point sphere radius")
    ap.add_argument("--hand-edge-radius", type=float, default=0.004, help="Hand edge capsule radius")

    args = ap.parse_args()

    # 1) start receiver (bind 6768 once)
    start_receiver(args.host, args.port)

    # 2) load MuJoCo
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.ee_site)
    if site_id < 0:
        raise ValueError(f"Site '{args.ee_site}' not found. Use --ee-site end_effector or ee_site.")

    # 3) calibration: first confident hand sample defines origin mapping
    print("[Teleop] Waiting for first confident right-hand sample...")
    origin_dev = None
    origin_mj = data.site_xpos[site_id].copy()

    while origin_dev is None:
        with _lock:
            h = HandState(**_hand.__dict__)
        if h.valid and h.wrist_dev is not None and h.conf >= args.conf:
            origin_dev = h.wrist_dev.copy()
            origin_mj = data.site_xpos[site_id].copy()
            print(f"[Teleop] Calibrated origin.")
            print(f"         origin_dev={origin_dev}")
            print(f"         origin_mj ={origin_mj}")
        else:
            time.sleep(0.01)

    target = origin_mj.copy()
    target_smooth = target.copy()

    # 4) viewer loop
    print("[Teleop] Running. Move your right wrist; EE follows. Close viewer or Ctrl-C to exit.")
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                now_ns = time.time_ns()
                with _lock:
                    h = HandState(**_hand.__dict__)

                age_s = (now_ns - h.t_ns) / 1e9 if h.t_ns else 999.0

                # update EE target from wrist
                if h.valid and h.wrist_dev is not None and h.conf >= args.conf and age_s <= args.timeout:
                    delta = (h.wrist_dev - origin_dev) * args.scale
                    delta = apply_axis_ops(delta, args.flip_x, args.flip_y, args.flip_z)
                    target = origin_mj + delta
                    target_smooth = (1 - args.alpha) * target_smooth + args.alpha * target

                # IK step
                err = ik_step_pos(
                    model, data, site_id, target_smooth,
                    damping=args.ik_damping,
                    step_scale=args.ik_step_scale,
                    max_dq=args.ik_max_dq,
                )

                # hand skeleton visualization in same mapped space (optional)
                if args.show_hand:
                    pts_dev = None
                    edges = None

                    if h.landmarks_dev is not None and h.landmarks_dev.ndim == 2 and h.landmarks_dev.shape[1] == 3:
                        pts_dev = h.landmarks_dev
                        edges = HAND_EDGES_21 if pts_dev.shape[0] == 21 else None
                    else:
                        fallback = []
                        if h.wrist_dev is not None:
                            fallback.append(h.wrist_dev)
                        if h.palm_dev is not None:
                            fallback.append(h.palm_dev)
                        if len(fallback) >= 1:
                            pts_dev = np.stack(fallback, axis=0)
                            edges = [(0, 1)] if len(fallback) == 2 else None

                    if pts_dev is not None and origin_dev is not None:
                        pts_delta = (pts_dev - origin_dev[None, :]) * args.scale
                        # apply same axis flips as EE target
                        if args.flip_x:
                            pts_delta[:, 0] *= -1
                        if args.flip_y:
                            pts_delta[:, 1] *= -1
                        if args.flip_z:
                            pts_delta[:, 2] *= -1

                        pts_mj = origin_mj[None, :] + pts_delta
                        _draw_hand_markers(
                            viewer,
                            pts_mj,
                            edges=edges,
                            sphere_size=args.hand_point_size,
                            edge_radius=args.hand_edge_radius,
                        )
                    else:
                        # clear markers if no data
                        viewer.user_scn.ngeom = 0

                # Some mujoco versions don't expose viewer.overlay on the passive handle.
                if hasattr(viewer, "overlay"):
                    try:
                        viewer.overlay(
                            mujoco.viewer.OverlayGrid.TOPLEFT,
                            "Aria streaming follow",
                            f"site={args.ee_site} conf={h.conf:.2f} age={age_s*1000:.0f}ms err={err:.3f}m",
                        )
                    except Exception:
                        pass

                viewer.sync()
                time.sleep(0.002)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()