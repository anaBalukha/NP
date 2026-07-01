#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
task3_run.py — Navigation in a pedestrian flow (Task 3)

Automatically uses 'gettyimages-1944915243-640_adpp.mp4'.
Click Start (green) and Target (red) on the first frame.
Generates:
- task3_AtoB.gif
- task3_BtoA.gif
- task3_metrics.png
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter

VIDEO_PATH = "gettyimages-1944915243-640_adpp.mp4"

@dataclass
class Params:
    m: float = 1.0
    kp: float = 4.0
    kd: float = 1.8
    vmax: float = 220.0
    krep: float = 60000.0
    rsafe: float = 55.0
    dt: float = 0.05

def detect_pedestrians(fg_mask: np.ndarray, min_area: int = 220) -> List[Tuple[np.ndarray, float]]:
    mask = cv2.medianBlur(fg_mask, 5)
    mask = cv2.threshold(mask, 220, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones: List[Tuple[np.ndarray, float]] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area: continue
        (x, y), r = cv2.minEnclosingCircle(c)
        zones.append((np.array([x, y], dtype=np.float64), float(r)))
    return zones

def repulsive_force(robot_xy: np.ndarray, zones: List[Tuple[np.ndarray, float]], p: Params) -> Tuple[np.ndarray, float]:
    f = np.zeros(2, dtype=np.float64)
    min_eff = float("inf")
    for center, r in zones:
        diff = robot_xy - center
        dist = float(np.hypot(diff[0], diff[1]))
        eff = dist - r
        min_eff = min(min_eff, eff)
        if dist < 1e-9: continue
        if eff < p.rsafe:
            denom = max(eff, 1.0)
            f += p.krep * diff / (denom ** 3)
    return f, min_eff

def simulate(frames: List[np.ndarray], zones_per_frame: List[List[Tuple[np.ndarray,float]]],
             start_xy: np.ndarray, goal_xy: np.ndarray, p: Params) -> Tuple[np.ndarray, List[float], List[float]]:
    pos = start_xy.copy().astype(np.float64)
    vel = np.zeros(2, dtype=np.float64)
    T = len(frames)
    traj = np.zeros((T, 2), dtype=np.float64)
    min_d = []
    speeds = []

    for k in range(T):
        traj[k] = pos
        zones = zones_per_frame[k]
        f_attr = p.kp * (goal_xy - pos)
        f_rep, md = repulsive_force(pos, zones, p)
        f_damp = -p.kd * vel
        acc = (f_attr + f_rep + f_damp) / p.m
        vel = vel + acc * p.dt
        sp = float(np.hypot(vel[0], vel[1]))
        if sp > p.vmax:
            vel *= p.vmax / sp
            sp = p.vmax
        pos = pos + vel * p.dt
        h, w = frames[0].shape[:2]
        pos[0] = float(np.clip(pos[0], 0, w-1))
        pos[1] = float(np.clip(pos[1], 0, h-1))
        min_d.append(md if math.isfinite(md) else 0.0)
        speeds.append(sp)
        if float(np.hypot(pos[0]-goal_xy[0], pos[1]-goal_xy[1])) < 15.0:
            traj[k:] = pos
            break
    return traj, min_d, speeds

def pick_points(frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    clicks: List[Tuple[float,float]] = []

    def onclick(event):
        if event.xdata is None or event.ydata is None: return
        if len(clicks) < 2:
            clicks.append((float(event.xdata), float(event.ydata)))
            ax.plot([clicks[-1][0]], [clicks[-1][1]], 'ro' if len(clicks)==2 else 'go', markersize=10)
            fig.canvas.draw()
        if len(clicks) == 2: plt.close(fig)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.imshow(rgb)
    ax.set_title("Click Start (green) then Target (red).")
    ax.axis("off")
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    if len(clicks) != 2: raise RuntimeError("You must click two points.")
    s = np.array(clicks[0], dtype=np.float64)
    g = np.array(clicks[1], dtype=np.float64)
    return s, g

def save_gif(frames: List[np.ndarray], zones_per_frame, traj: np.ndarray, start, goal, out_name: str):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.axis("off")
    robot = Circle((traj[0,0], traj[0,1]), radius=10, color="cyan", alpha=0.8)
    ax.add_patch(robot)
    def update(k):
        ax.clear()
        ax.axis("off")
        frame = cv2.cvtColor(frames[k], cv2.COLOR_BGR2RGB)
        ax.imshow(frame)
        for center, r in zones_per_frame[k]:
            ax.add_patch(Circle((center[0], center[1]), r, color="blue", alpha=0.25))
            ax.add_patch(Circle((center[0], center[1]), r+55, color="orange", alpha=0.1, fill=False, linestyle="--", linewidth=1))
        ax.plot([start[0]], [start[1]], "go", markersize=10)
        ax.plot([goal[0]], [goal[1]], "r*", markersize=14)
        ax.plot(traj[:k+1,0], traj[:k+1,1], "y-", linewidth=2, alpha=0.9)
        ax.add_patch(Circle((traj[k,0], traj[k,1]), radius=10, color="cyan", alpha=0.8))
        ax.set_title(out_name, fontsize=12)
        return ()
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100, blit=False)
    anim.save(out_name, writer=PillowWriter(fps=10))
    plt.close(fig)

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): raise RuntimeError("Cannot open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps>1e-6 else 25.0

    frames: List[np.ndarray] = []
    zones_per_frame: List[List[Tuple[np.ndarray,float]]] = []

    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=22, detectShadows=False)

    ret, first = cap.read()
    if not ret: raise RuntimeError("Cannot read first frame.")
    scale = 1.0
    h0, w0 = first.shape[:2]
    max_size = 720
    max_dim = max(w0, h0)
    if max_dim > max_size:
        scale = max_dim / max_size
        new_w = int(round(w0 / scale))
        new_h = int(round(h0 / scale))
        first = cv2.resize(first, (new_w,new_h), interpolation=cv2.INTER_AREA)

    frames.append(first)
    _ = bg.apply(first)

    idx = 0
    max_frames = 260
    frame_skip = 2
    while True:
        ret, frame = cap.read()
        if not ret or len(frames)>=max_frames: break
        idx += 1
        if idx % frame_skip != 0: continue
        if scale != 1.0:
            frame = cv2.resize(frame, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
        fg = bg.apply(frame)
        zones = detect_pedestrians(fg)
        frames.append(frame)
        zones_per_frame.append(zones)

    if len(zones_per_frame) < len(frames):
        zones_per_frame = [zones_per_frame[0] if zones_per_frame else []] + zones_per_frame
    else:
        zones_per_frame = zones_per_frame[:len(frames)]

    cap.release()

    p = Params()
    p.dt = frame_skip / fps
    p.vmax = 0.55 * min(frames[0].shape[0], frames[0].shape[1])
    p.rsafe = 0.07 * min(frames[0].shape[0], frames[0].shape[1])
    p.krep = 90000.0

    start, goal = pick_points(frames[0])

    traj1, md1, sp1 = simulate(frames, zones_per_frame, start, goal, p)
    traj2, md2, sp2 = simulate(frames, zones_per_frame, goal, start, p)

    save_gif(frames, zones_per_frame, traj1, start, goal, "task3_AtoB.gif")
    save_gif(frames, zones_per_frame, traj2, goal, start, "task3_BtoA.gif")

    fig, ax = plt.subplots(2,1,figsize=(10,8))
    ax[0].plot(md1, label="A->B min distance")
    ax[0].plot(md2, label="B->A min distance")
    ax[0].axhline(y=0, linestyle="--")
    ax[0].set_title("Minimum effective distance to pedestrians")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    ax[1].plot(sp1, label="A->B speed")
    ax[1].plot(sp2, label="B->A speed")
    ax[1].set_title("Robot speed (px/s)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("task3_metrics.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()
