"""
Top-down 2D preview renderer for battery module assembly layouts.

Renders a bird's-eye view of the module tray showing:
- Cell positions with color-coded types
- Robot arm base with reach envelope circle
- Inspection camera position
- Safety gap indicators
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

from schema import ModuleTask
from config import PREVIEW_DPI, PREVIEW_FIGSIZE, CELL_CATALOG, ROBOT_PROFILES


# ── Color palette for cell types ──
CELL_COLORS = {
    "LG_E63": "#4CAF50",       # Green — pouch cell
    "HY_50Ah": "#2196F3",      # Blue — prismatic
    "CATL_LFP": "#FF9800",     # Orange — LFP prismatic
}
DEFAULT_CELL_COLOR = "#9E9E9E"


def render_preview(spec: ModuleTask, output_path: str = "preview.png") -> str:
    """
    Render a top-down (XY-plane) view of the battery module assembly.
    Returns the path to the saved image.
    """
    fig, ax = plt.subplots(1, 1, figsize=PREVIEW_FIGSIZE)

    bounds_x = spec.module_tray_bounds[0]
    bounds_y = spec.module_tray_bounds[1]

    # ── Draw module tray ──
    tray_rect = patches.FancyBboxPatch(
        (0, 0), bounds_x, bounds_y,
        boxstyle="round,pad=0.01",
        linewidth=2.5, edgecolor='#37474F', facecolor='#ECEFF1',
        alpha=0.4, label='Module Tray', zorder=1,
    )
    ax.add_patch(tray_rect)

    # ── Draw robot reach envelope ──
    robot = spec.robot
    profile = ROBOT_PROFILES.get(robot.model.value, ROBOT_PROFILES["UR10e"])
    max_reach = profile["max_reach"]
    dead_zone = profile["dead_zone"]

    # Max reach circle
    reach_circle = patches.Circle(
        (robot.base_position[0], robot.base_position[1]),
        max_reach,
        linewidth=1.5, edgecolor='#F44336', facecolor='#FFCDD2',
        alpha=0.15, linestyle='--', label=f'{robot.model.value} reach ({max_reach}m)',
        zorder=2,
    )
    ax.add_patch(reach_circle)

    # Dead zone circle
    dead_circle = patches.Circle(
        (robot.base_position[0], robot.base_position[1]),
        dead_zone,
        linewidth=1, edgecolor='#B71C1C', facecolor='#EF9A9A',
        alpha=0.3, linestyle=':', label=f'Dead zone ({dead_zone}m)',
        zorder=2,
    )
    ax.add_patch(dead_circle)

    # Robot base marker
    ax.plot(
        robot.base_position[0], robot.base_position[1],
        marker='s', markersize=14, color='#D32F2F',
        markeredgecolor='white', markeredgewidth=2,
        zorder=10, label='Robot Base',
    )

    # ── Draw each cell ──
    for cell in spec.cells:
        cat = CELL_CATALOG.get(cell.cell_type.value, {})
        cell_w = cat.get("width", 0.05)
        cell_d = cat.get("depth", 0.12)
        color = CELL_COLORS.get(cell.cell_type.value, DEFAULT_CELL_COLOR)

        # Rectangle centered on position
        x = cell.position[0] - cell_w / 2
        y = cell.position[1] - cell_d / 2

        cell_rect = patches.FancyBboxPatch(
            (x, y), cell_w, cell_d,
            boxstyle="round,pad=0.001",
            linewidth=1.2,
            edgecolor='#263238',
            facecolor=color,
            alpha=0.85,
            zorder=5,
        )
        ax.add_patch(cell_rect)

        # Label
        fontsize = max(5, min(8, int(30 / max(len(spec.cells), 1))))
        label = cell.id.replace("Cell_", "C")
        ax.annotate(
            label,
            xy=(cell.position[0], cell.position[1]),
            ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white',
            zorder=6,
        )

        # Polarity indicator (small arrow showing rotation direction)
        arrow_len = cell_d * 0.3
        angle_rad = np.radians(cell.rotation_y)
        dx = arrow_len * np.sin(angle_rad)
        dy = arrow_len * np.cos(angle_rad)
        ax.annotate(
            '', xy=(cell.position[0] + dx, cell.position[1] + dy),
            xytext=(cell.position[0], cell.position[1]),
            arrowprops=dict(arrowstyle='->', color='white', lw=1),
            zorder=7,
        )

    # ── Draw camera ──
    cam = spec.camera
    ax.plot(
        cam.position[0], cam.position[1],
        marker='^', markersize=12, color='#7B1FA2',
        markeredgecolor='white', markeredgewidth=2,
        zorder=10, label=f'Camera (z={cam.position[2]:.2f}m)',
    )

    # ── Title and labels ──
    ax.set_title(
        f"🔋 {spec.task_id}  |  {len(spec.cells)} cells  |  "
        f"{robot.model.value} + {robot.gripper.value}",
        fontsize=12, fontweight='bold', pad=12,
    )
    ax.set_xlabel("X (meters)", fontsize=10)
    ax.set_ylabel("Y (meters)", fontsize=10)

    # Adjust view to show full reach envelope
    margin = 0.1
    view_range = max(max_reach * 1.1, bounds_x + margin, bounds_y + margin)
    ax.set_xlim(-view_range * 0.3, view_range)
    ax.set_ylim(-view_range * 0.3, view_range)

    # Legend
    cell_types_used = set(c.cell_type.value for c in spec.cells)
    legend_patches = [
        patches.Patch(
            facecolor=CELL_COLORS.get(ct, DEFAULT_CELL_COLOR),
            edgecolor='#263238', label=ct,
        )
        for ct in sorted(cell_types_used)
    ]
    ax.legend(
        handles=legend_patches + ax.get_legend_handles_labels()[0][-4:],
        loc='upper right', fontsize=7,
        framealpha=0.9, fancybox=True,
    )

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='-', color='#B0BEC5')

    # Save
    fig.savefig(output_path, dpi=PREVIEW_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return output_path
