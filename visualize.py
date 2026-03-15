"""3D pallet packing visualizer: generates interactive HTML with Three.js.

Usage:
    python visualize.py benchmark_viz.json -o viz/
    python visualize.py benchmark_viz.json --single combined.html

Reads JSON with pallet + placement data per scenario and generates
standalone HTML files with interactive 3D visualization.
"""

import argparse
import json
import logging
import os
import hashlib
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# X5 Tech brand-aligned palette for 3D visualization on dark background
SKU_COLORS = [
    "#67DB3A", "#CBB7F6", "#EEDCA8", "#BFE8C6",  # Brand green, Lavender, Sand, Mint
    "#A8F36A", "#E8A0C8", "#8BD4E0", "#D7F4A8",  # Lime, Rose, Teal, Pale lime
    "#59C436", "#B0A0E8", "#F0C878", "#7ECBA1",  # Stable green, Soft violet, Gold, Sage
    "#6FEA3A", "#D4C4F0", "#E0B890", "#90D8B8",  # Bright green, Light lavender, Peach, Aqua
    "#4AAE2A", "#C8B0E0", "#E8D0A0", "#A8E0C8",  # Deep green, Mauve, Cream, Pale mint
]


def _color_for_sku(sku_id: str, index: int) -> str:
    """Deterministic color per SKU."""
    return SKU_COLORS[index % len(SKU_COLORS)]


def _calc_overlap_2d(b1: Dict[str, Any], b2: Dict[str, Any]) -> float:
    """Площадь пересечения проекций на XY (x_min/x_max, y_min/y_max)."""
    dx = max(0, min(b1["x_max"], b2["x_max"]) - max(b1["x_min"], b2["x_min"]))
    dy = max(0, min(b1["y_max"], b2["y_max"]) - max(b1["y_min"], b2["y_min"]))
    return dx * dy


def _layout_unplaced_boxes(
    unplaced: List[Dict[str, Any]],
    boxes_meta: Dict[str, Any],
    pallet: Dict[str, Any],
    margin_mm: int = 450,
) -> List[Dict[str, Any]]:
    """Раскладывает неразмещённые коробки сбоку от паллеты (вдоль X). Возвращает список как placements."""
    out = []
    p_w = pallet["width_mm"]
    p_l = pallet["length_mm"]
    x_base = p_l + margin_mm
    current_z = 0
    current_y = 0
    row_max_dy = 0

    for u in unplaced:
        sku_id = u.get("sku_id", "")
        qty = u.get("quantity_unplaced", 0)
        if qty <= 0:
            continue
        meta = boxes_meta.get(sku_id, {})
        length_mm = meta.get("length_mm", 100)
        width_mm = meta.get("width_mm", 100)
        height_mm = meta.get("height_mm", 100)
        for _ in range(qty):
            out.append({
                "sku_id": sku_id,
                "x_mm": x_base,
                "y_mm": current_y,
                "z_mm": current_z,
                "length_mm": length_mm,
                "width_mm": width_mm,
                "height_mm": height_mm,
                "fragile": meta.get("fragile", False),
                "weight_kg": meta.get("weight_kg", 0),
            })
            current_z += width_mm
            row_max_dy = max(row_max_dy, height_mm)
            if current_z > p_w * 1.2:
                current_z = 0
                current_y += row_max_dy
                row_max_dy = 0

    return out


def _fragility_violation_bottom_indices(placements: List[Dict[str, Any]]) -> set:
    """Индексы placement'ов, которые являются нижним (хрупким) объектом в паре с нарушением fragile.
    Правило: тяжёлая (>2 kg) коробка сверху на хрупкой при пересечении по XY.
    """
    n = len(placements)
    boxes = []
    for p in placements:
        boxes.append({
            "x_min": p["x_mm"],
            "x_max": p["x_mm"] + p["length_mm"],
            "y_min": p["y_mm"],
            "y_max": p["y_mm"] + p["width_mm"],
            "z_min": p["z_mm"],
            "z_max": p["z_mm"] + p["height_mm"],
            "weight_kg": p.get("weight_kg", 0),
            "fragile": p.get("fragile", False),
        })
    bottom_indices = set()
    for i in range(n):
        if not boxes[i]["fragile"]:
            continue
        for j in range(n):
            if i == j:
                continue
            if boxes[j]["weight_kg"] <= 2.0:
                continue
            if abs(boxes[j]["z_min"] - boxes[i]["z_max"]) >= 1e-6:
                continue
            if _calc_overlap_2d(boxes[i], boxes[j]) <= 0:
                continue
            bottom_indices.add(i)
            break
    return bottom_indices


def build_scenario_viz_data(
    request_dict: Dict[str, Any],
    response_dict: Dict[str, Any],
    scenario_name: str,
    score: float = 0.0,
) -> Dict[str, Any]:
    """Build visualization payload for a single request/response pair."""
    pallet = dict(request_dict.get("pallet", {}))
    boxes_meta = {b["sku_id"]: b for b in request_dict.get("boxes", [])}

    placements = []
    for p in response_dict.get("placements", []):
        sku = boxes_meta.get(p["sku_id"], {})
        dim = p["dimensions_placed"]
        pos = p["position"]
        placements.append(
            {
                "sku_id": p["sku_id"],
                "x_mm": pos["x_mm"],
                "y_mm": pos["y_mm"],
                "z_mm": pos["z_mm"],
                "length_mm": dim["length_mm"],
                "width_mm": dim["width_mm"],
                "height_mm": dim["height_mm"],
                "fragile": sku.get("fragile", False),
                "weight_kg": sku.get("weight_kg", 0),
            }
        )

    total_items = sum(int(box.get("quantity", 0)) for box in request_dict.get("boxes", []))
    placed = len(response_dict.get("placements", []))
    unplaced = response_dict.get("unplaced", [])
    unplaced_boxes = _layout_unplaced_boxes(unplaced, boxes_meta, pallet)

    return {
        "pallet": pallet,
        "placements": placements,
        "unplaced_boxes": unplaced_boxes,
        "meta": {
            "scenario": scenario_name,
            "score": score,
            "placed": placed,
            "total_items": total_items,
        },
    }


def _generate_html(scenario: Dict[str, Any]) -> str:
    """Generate a standalone HTML file with Three.js 3D visualization."""
    pallet = scenario["pallet"]
    placements = scenario.get("placements", [])
    meta = scenario.get("meta", {})
    scenario_name = meta.get("scenario", "unknown")
    score = meta.get("score", 0)
    placed = meta.get("placed", 0)
    total = meta.get("total_items", 0)

    # Assign colors to SKUs (placed + unplaced)
    unplaced_boxes = scenario.get("unplaced_boxes", [])
    sku_ids = sorted(set(p["sku_id"] for p in placements) | set(p["sku_id"] for p in unplaced_boxes))
    sku_color_map = {sid: _color_for_sku(sid, i) for i, sid in enumerate(sku_ids)}

    # Fragility violations: нижний (хрупкий) объект в паре — выделяем ему верхнюю грань
    violation_bottom = _fragility_violation_bottom_indices(placements)

    # Build JS data for placed boxes
    boxes_js = json.dumps([
        {
            "sku_id": p["sku_id"],
            "x": p["x_mm"], "y": p["z_mm"], "z": p["y_mm"],  # Three.js: Y is up
            "dx": p["length_mm"], "dy": p["height_mm"], "dz": p["width_mm"],
            "color": sku_color_map[p["sku_id"]],
            "fragile": p.get("fragile", False),
            "weight_kg": p.get("weight_kg", 0),
            "fragility_violation_bottom": i in violation_bottom,
        }
        for i, p in enumerate(placements)
    ])

    # Build JS data for unplaced boxes (same coords: x_mm, z_mm -> y, y_mm -> z in Three)
    unplaced_boxes_js = json.dumps([
        {
            "sku_id": p["sku_id"],
            "x": p["x_mm"], "y": p["z_mm"], "z": p["y_mm"],
            "dx": p["length_mm"], "dy": p["height_mm"], "dz": p["width_mm"],
            "color": sku_color_map[p["sku_id"]],
            "fragile": p.get("fragile", False),
            "weight_kg": p.get("weight_kg", 0),
        }
        for p in unplaced_boxes
    ])

    pallet_js = json.dumps({
        "dx": pallet["length_mm"],
        "dy": pallet["max_height_mm"],
        "dz": pallet["width_mm"],
    })

    unplaced_count = len(unplaced_boxes)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Pallet — {scenario_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: "Space Grotesk", -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: #171717; color: #EDEDED; overflow: hidden;
    width: 100vw; height: 100vh;
  }}
  #canvas-container {{
    position: absolute; inset: 0;
  }}
  #hud {{
    position: absolute; inset: 0; pointer-events: none; z-index: 10;
    display: grid;
    grid-template-columns: auto 1fr auto;
    grid-template-rows: auto 1fr auto;
    padding: 12px;
    gap: 10px;
  }}
  #info, #legend, #slider-wrap {{ pointer-events: auto; }}
  #info {{
    grid-column: 1; grid-row: 1;
    background: rgba(31,31,31,0.88); padding: 12px 16px; border-radius: 16px;
    font-size: 12px; line-height: 1.6; backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    max-width: 280px;
  }}
  #info h2 {{ font-size: 14px; margin-bottom: 4px; color: #67DB3A; font-weight: 600; }}
  #info .metric {{ color: #8E8E8E; }}
  #info .value {{ color: #EDEDED; font-weight: 600; }}
  #legend {{
    grid-column: 3; grid-row: 1 / 3;
    align-self: start;
    background: rgba(31,31,31,0.88); padding: 10px 14px; border-radius: 16px;
    font-size: 11px; max-height: calc(100vh - 80px); overflow-y: auto; backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    max-width: 260px;
  }}
  #legend .title {{ color: #8E8E8E; font-size: 10px; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 6px; }}
  #legend .item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; line-height: 1.5; color: #BDBDBD; }}
  #legend .swatch {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
  #tooltip {{
    position: absolute; display: none; z-index: 20;
    background: rgba(31,31,31,0.94); padding: 10px 14px; border-radius: 12px;
    font-size: 12px; pointer-events: none; white-space: nowrap;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(12px);
    color: #EDEDED;
  }}
  #slider-wrap {{
    grid-column: 1 / 4; grid-row: 3;
    justify-self: center;
    background: rgba(31,31,31,0.88); padding: 10px 20px; border-radius: 16px;
    display: flex; align-items: center; gap: 12px; backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
  }}
  #slider-wrap label {{ color: #8E8E8E; font-size: 12px; white-space: nowrap; }}
  #slider-wrap input[type="range"] {{ width: 240px; accent-color: #67DB3A; }}
  #slider-wrap .step-value {{ color: #EDEDED; font-weight: 600; min-width: 6ch; font-size: 12px; }}
  canvas {{ display: block; }}

  /* ── Mobile Responsive ─────────────────────────── */
  @media (max-width: 768px) {{
    #hud {{
      grid-template-columns: 1fr;
      grid-template-rows: auto 1fr auto;
      padding: 8px;
      gap: 6px;
    }}
    #info {{
      grid-column: 1; grid-row: 1;
      max-width: 100%;
      font-size: 11px;
      padding: 8px 12px;
      border-radius: 12px;
    }}
    #info h2 {{ font-size: 12px; }}
    #legend {{
      grid-column: 1; grid-row: auto;
      position: fixed;
      bottom: 60px;
      right: 8px;
      max-width: 180px;
      max-height: 40vh;
      font-size: 10px;
      padding: 8px 10px;
      border-radius: 12px;
      opacity: 0.85;
    }}
    #slider-wrap {{
      grid-column: 1;
      padding: 8px 12px;
      border-radius: 12px;
      gap: 8px;
    }}
    #slider-wrap input[type="range"] {{ width: 160px; }}
    #slider-wrap label {{ font-size: 11px; }}
    #tooltip {{ font-size: 11px; }}
  }}

  @media (max-width: 480px) {{
    #info {{ padding: 6px 10px; }}
    #info h2 {{ font-size: 11px; }}
    #legend {{ max-width: 150px; font-size: 9px; bottom: 52px; }}
    #slider-wrap input[type="range"] {{ width: 120px; }}
  }}
</style>
</head>
<body>

<div id="canvas-container"></div>

<div id="hud">
  <div id="info">
    <h2>{scenario_name}</h2>
    <span class="metric">Score:</span> <span class="value">{score:.4f}</span><br>
    <span class="metric">Placed:</span> <span class="value">{placed}/{total}</span><br>
    <span class="metric">Unplaced:</span> <span class="value">{unplaced_count}</span> <span class="metric">(справа)</span><br>
  </div>

  <div></div><!-- grid spacer -->

  <div id="legend"></div>

  <div></div><!-- grid spacer row 2 col 1 -->
  <div></div><!-- grid spacer row 2 col 2 -->

  <div id="slider-wrap">
    <label for="step-slider">Шаг укладки:</label>
    <input type="range" id="step-slider" min="0" max="1" value="1" step="1">
    <span class="step-value" id="step-value">0 / 0</span>
  </div>
</div>

<div id="tooltip"></div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const PALLET = {pallet_js};
const BOXES = {boxes_js};
const UNPLACED_BOXES = {unplaced_boxes_js};

// Scale to meters for better camera behavior
const S = 1 / 1000;

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x171717);

// Camera: if unplaced exist, frame both pallet and unplaced zone
const cx = PALLET.dx * S / 2, cy = PALLET.dy * S / 2, cz = PALLET.dz * S / 2;
const maxDim = Math.max(PALLET.dx, PALLET.dy, PALLET.dz) * S;
const unplacedWidth = UNPLACED_BOXES.length > 0 ? 700 * S : 0;
const viewCenterX = cx + (unplacedWidth / 2);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(viewCenterX + maxDim * 1.4, cy + maxDim * 0.8, cz + maxDim * 1.2);
camera.lookAt(viewCenterX, cy * 0.4, cz);

// Renderer
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.getElementById('canvas-container').appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy * 0.4, cz);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

// Lighting
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(maxDim * 2, maxDim * 3, maxDim * 2);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
fillLight.position.set(-maxDim, maxDim * 0.5, -maxDim);
scene.add(fillLight);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(maxDim * 4, maxDim * 4);
const groundMat = new THREE.MeshStandardMaterial({{ color: 0x1F1F1F, roughness: 0.9 }});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(cx, -0.001, cz);
ground.receiveShadow = true;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(maxDim * 3, 30, 0x2A2A2A, 0x1F1F1F);
grid.position.set(cx, 0, cz);
scene.add(grid);

// 3D Pallet — realistic wooden pallet with deck boards, stringers, bottom boards
const palletGroup = new THREE.Group();
const pL = PALLET.dx * S;  // pallet length (X)
const pW = PALLET.dz * S;  // pallet width (Z)
const boardH = 0.018 * S * 1000;   // board thickness ~18mm
const stringerH = 0.078 * S * 1000; // stringer height ~78mm
const stringerW = 0.090 * S * 1000; // stringer width ~90mm
const palletTotalH = boardH * 2 + stringerH; // total pallet height
const woodMat = new THREE.MeshStandardMaterial({{
  color: 0x8B7355, roughness: 0.85, metalness: 0.02
}});
const woodDarkMat = new THREE.MeshStandardMaterial({{
  color: 0x6B5740, roughness: 0.9, metalness: 0.02
}});
const woodEdgeMat = new THREE.LineBasicMaterial({{ color: 0x5A4A35, transparent: true, opacity: 0.5 }});

function addBoard(w, h, d, x, y, z, mat) {{
  const geo = new THREE.BoxGeometry(w, h, d);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(x, y, z);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  palletGroup.add(mesh);
  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(edges, woodEdgeMat);
  line.position.copy(mesh.position);
  palletGroup.add(line);
}}

// Top deck boards (7 boards across width)
const deckCount = 7;
const deckBoardW = pW / deckCount * 0.85;
const deckGap = (pW - deckBoardW * deckCount) / (deckCount - 1);
const topY = -boardH / 2;
for (let i = 0; i < deckCount; i++) {{
  const zPos = deckBoardW / 2 + i * (deckBoardW + deckGap);
  addBoard(pL, boardH, deckBoardW, pL / 2, topY, zPos, woodMat);
}}

// 3 stringers (along length)
const stringerPositions = [stringerW / 2, pW / 2, pW - stringerW / 2];
const stringerY = -(boardH + stringerH / 2);
for (const zPos of stringerPositions) {{
  addBoard(pL, stringerH, stringerW, pL / 2, stringerY, zPos, woodDarkMat);
}}

// Bottom boards (3 boards across length)
const bottomY = -(boardH + stringerH + boardH / 2);
const bottomCount = 3;
const bottomBoardL = pL / bottomCount * 0.85;
const bottomGap = (pL - bottomBoardL * bottomCount) / (bottomCount - 1);
for (let i = 0; i < bottomCount; i++) {{
  const xPos = bottomBoardL / 2 + i * (bottomBoardL + bottomGap);
  addBoard(bottomBoardL, boardH, pW, xPos, bottomY, pW / 2, woodMat);
}}

scene.add(palletGroup);

// Pallet height limit (wireframe) — offset up by pallet height so boxes sit on top
const limitGeo = new THREE.BoxGeometry(PALLET.dx * S, PALLET.dy * S, PALLET.dz * S);
const limitEdges = new THREE.EdgesGeometry(limitGeo);
const limitLine = new THREE.LineSegments(limitEdges, new THREE.LineBasicMaterial({{ color: 0x2A2A2A, transparent: true, opacity: 0.25 }}));
limitLine.position.set(cx, cy, cz);
scene.add(limitLine);

// Box meshes
const boxMeshes = [];
const boxData = [];
const placedBoxNodes = [];

BOXES.forEach((box, i) => {{
  const geo = new THREE.BoxGeometry(box.dx * S, box.dy * S, box.dz * S);
  const color = new THREE.Color(box.color);
  const mat = new THREE.MeshStandardMaterial({{
    color: color,
    roughness: 0.5,
    metalness: 0.1,
    transparent: true,
    opacity: 0.88,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(
    (box.x + box.dx / 2) * S,
    (box.y + box.dy / 2) * S,
    (box.z + box.dz / 2) * S
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);

  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x111111, transparent: true, opacity: 0.4 }}));
  line.position.copy(mesh.position);
  scene.add(line);

  let marker = null;
  const cx = (box.x + box.dx / 2) * S;
  const cyTop = (box.y + box.dy) * S + 0.005;
  const cz = (box.z + box.dz / 2) * S;
  const markerMat = new THREE.MeshStandardMaterial({{ color: 0xef4444, emissive: 0xdc2626, emissiveIntensity: 0.7 }});
  if (box.fragility_violation_bottom) {{
    const r = Math.min(box.dx, box.dz) * S * 0.2;
    const t = r * 0.2;
    marker = new THREE.Group();
    marker.add(new THREE.Mesh(new THREE.BoxGeometry(r * 2, t, t), markerMat));
    marker.add(new THREE.Mesh(new THREE.BoxGeometry(t, r * 2, t), markerMat));
    marker.add(new THREE.Mesh(new THREE.BoxGeometry(t, t, r * 2), markerMat));
    marker.position.set(cx, cyTop, cz);
    scene.add(marker);
  }} else if (box.fragile) {{
    marker = new THREE.Mesh(new THREE.SphereGeometry(Math.min(box.dx, box.dz) * S * 0.15, 8, 8), markerMat);
    marker.position.set(cx, cyTop, cz);
    scene.add(marker);
  }}

  placedBoxNodes.push({{ mesh, line, marker }});
  boxMeshes.push(mesh);
  boxData.push({{ ...box, unplaced: false }});
}});

UNPLACED_BOXES.forEach((box, i) => {{
  const geo = new THREE.BoxGeometry(box.dx * S, box.dy * S, box.dz * S);
  const mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(box.color),
    roughness: 0.5,
    metalness: 0.1,
    transparent: true,
    opacity: 0.6,
  }});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(
    (box.x + box.dx / 2) * S,
    (box.y + box.dy / 2) * S,
    (box.z + box.dz / 2) * S
  );
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  scene.add(mesh);
  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x333333, transparent: true, opacity: 0.35 }}));
  line.position.copy(mesh.position);
  scene.add(line);
  boxMeshes.push(mesh);
  boxData.push({{ ...box, unplaced: true }});
}});

// Legend
const legend = document.getElementById('legend');
const titleEl = document.createElement('div');
titleEl.className = 'title';
titleEl.textContent = 'SKU Legend';
legend.appendChild(titleEl);
const skuSet = new Map();
[...BOXES, ...UNPLACED_BOXES].forEach(b => {{
  if (!skuSet.has(b.sku_id)) {{
    const placed = BOXES.filter(x => x.sku_id === b.sku_id).length;
    const unplaced = UNPLACED_BOXES.filter(x => x.sku_id === b.sku_id).length;
    skuSet.set(b.sku_id, {{ color: b.color, placed, unplaced }});
  }}
}});
skuSet.forEach((v, sku) => {{
  const item = document.createElement('div');
  item.className = 'item';
  const label = v.unplaced > 0 ? `${{sku}} (${{v.placed}}+${{v.unplaced}})` : `${{sku}} (${{v.placed}})`;
  item.innerHTML = `<div class="swatch" style="background:${{v.color}}"></div>${{label}}`;
  legend.appendChild(item);
}});

// Step slider: show only first N placed boxes
const stepSlider = document.getElementById('step-slider');
const stepValueEl = document.getElementById('step-value');
const totalPlaced = BOXES.length;
stepSlider.min = 0;
stepSlider.max = Math.max(1, totalPlaced);
stepSlider.value = totalPlaced;
stepSlider.step = 1;
stepValueEl.textContent = totalPlaced + ' / ' + totalPlaced;
function updateStepVisibility(step) {{
  const n = Math.max(0, Math.min(step, totalPlaced));
  placedBoxNodes.forEach((node, i) => {{
    const visible = i < n;
    node.mesh.visible = visible;
    node.line.visible = visible;
    if (node.marker) node.marker.visible = visible;
  }});
  stepValueEl.textContent = n + ' / ' + totalPlaced;
}}
stepSlider.addEventListener('input', () => updateStepVisibility(parseInt(stepSlider.value, 10)));
if (totalPlaced === 0) {{
  document.getElementById('slider-wrap').style.display = 'none';
}} else {{
  updateStepVisibility(0);
  const targetFrames = 42;
  const stepAdvance = Math.max(1, Math.ceil(totalPlaced / targetFrames));
  let currentStep = 0;
  const autoplay = window.setInterval(() => {{
    currentStep = Math.min(totalPlaced, currentStep + stepAdvance);
    stepSlider.value = currentStep;
    updateStepVisibility(currentStep);
    if (currentStep >= totalPlaced) {{
      window.clearInterval(autoplay);
    }}
  }}, 26);
}}

// Tooltip on hover
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');

renderer.domElement.addEventListener('mousemove', (e) => {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(boxMeshes);
  if (intersects.length > 0) {{
    const idx = boxMeshes.indexOf(intersects[0].object);
    if (idx >= 0) {{
      const b = boxData[idx];
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX + 12) + 'px';
      tooltip.style.top = (e.clientY + 12) + 'px';
      tooltip.innerHTML = `<b>${{b.sku_id}}</b>${{b.unplaced ? ' <span style="color:#888">(Unplaced)</span>' : ''}}<br>${{b.dx}}×${{b.dz}}×${{b.dy}} mm<br>Weight: ${{b.weight_kg}} kg${{b.fragile ? '<br><span style="color:#ff6666">FRAGILE</span>' : ''}}`;
    }}
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

// Resize
window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// Animate
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""


def generate_scenario_html(scenario: Dict[str, Any]) -> str:
    """Public wrapper for generating standalone visualization HTML."""
    return _generate_html(scenario)


def generate_viz_data(benchmark_results: List[Dict], requests: List[Dict]) -> List[Dict]:
    """Build visualization data from benchmark results and original requests.

    Args:
        benchmark_results: List of result dicts from benchmark.py
        requests: List of original request dicts (pallet + boxes + placements)
    """
    viz_data = []
    for result, request in zip(benchmark_results, requests):
        viz_data.append(
            build_scenario_viz_data(
                request_dict=request,
                response_dict=result.get("response", {}),
                scenario_name=result["scenario"],
                score=result.get("final_score", 0),
            )
        )

    return viz_data


def generate_html_files(viz_json_path: str, output_dir: str) -> List[str]:
    """Read viz JSON and generate HTML files for each scenario.

    Args:
        viz_json_path: Path to JSON file with viz data
        output_dir: Directory to write HTML files

    Returns:
        List of generated HTML file paths
    """
    with open(viz_json_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    generated = []
    for scenario in scenarios:
        name = scenario.get("meta", {}).get("scenario", "unknown")
        html = _generate_html(scenario)
        path = os.path.join(output_dir, f"viz_{name}.html")
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        generated.append(path)
        logger.info("[visualize] generated %s (%d placements)", path, len(scenario.get("placements", [])))

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate 3D pallet packing visualizations")
    parser.add_argument("input", help="Path to benchmark_viz.json")
    parser.add_argument("-o", "--output-dir", default="viz", help="Output directory (default: viz/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    files = generate_html_files(args.input, args.output_dir)
    print(f"Generated {len(files)} visualization(s):")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()
