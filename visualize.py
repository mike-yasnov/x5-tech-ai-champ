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

# Distinct colors for different SKUs
SKU_COLORS = [
    "#4285F4", "#EA4335", "#FBBC04", "#34A853",  # Google palette
    "#FF6D01", "#46BDC6", "#7BAAF7", "#F07B72",
    "#FCD04F", "#57BB8A", "#FF8A65", "#4DD0E1",
    "#9575CD", "#F06292", "#AED581", "#FFD54F",
    "#A1887F", "#90A4AE", "#CE93D8", "#80CBC4",
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
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }}
  #info {{
    position: absolute; top: 12px; left: 12px; z-index: 10;
    background: rgba(0,0,0,0.7); padding: 12px 16px; border-radius: 8px;
    font-size: 13px; line-height: 1.6; backdrop-filter: blur(8px);
  }}
  #info h2 {{ font-size: 16px; margin-bottom: 4px; color: #7BAAF7; }}
  #info .metric {{ color: #aaa; }}
  #info .value {{ color: #fff; font-weight: 600; }}
  #legend {{
    position: absolute; bottom: 12px; left: 12px; z-index: 10;
    background: rgba(0,0,0,0.7); padding: 10px 14px; border-radius: 8px;
    font-size: 12px; max-height: 200px; overflow-y: auto; backdrop-filter: blur(8px);
  }}
  #legend .item {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
  #legend .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; }}
  #tooltip {{
    position: absolute; display: none; z-index: 20;
    background: rgba(0,0,0,0.85); padding: 8px 12px; border-radius: 6px;
    font-size: 12px; pointer-events: none; white-space: nowrap;
  }}
  canvas {{ display: block; }}
</style>
</head>
<body>

<div id="info">
  <h2>{scenario_name}</h2>
  <span class="metric">Score:</span> <span class="value">{score:.4f}</span><br>
  <span class="metric">Placed:</span> <span class="value">{placed}/{total}</span><br>
  <span class="metric">Unplaced:</span> <span class="value">{unplaced_count}</span> <span class="metric">(shown to the right)</span><br>
  <span class="metric">Controls:</span> <span class="metric">drag to rotate, scroll to zoom</span>
</div>

<div id="legend"></div>
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
scene.background = new THREE.Color(0x1a1a2e);

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
document.body.appendChild(renderer.domElement);

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
const groundMat = new THREE.MeshStandardMaterial({{ color: 0x16213e, roughness: 0.9 }});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(cx, -0.001, cz);
ground.receiveShadow = true;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(maxDim * 3, 30, 0x334466, 0x1a2a44);
grid.position.set(cx, 0, cz);
scene.add(grid);

// Pallet base (wireframe)
const palletGeo = new THREE.BoxGeometry(PALLET.dx * S, 0.005, PALLET.dz * S);
const palletMat = new THREE.MeshStandardMaterial({{
  color: 0x8B7355, roughness: 0.8, metalness: 0.1
}});
const palletMesh = new THREE.Mesh(palletGeo, palletMat);
palletMesh.position.set(cx, -0.0025, cz);
palletMesh.receiveShadow = true;
scene.add(palletMesh);

// Pallet height limit (wireframe)
const limitGeo = new THREE.BoxGeometry(PALLET.dx * S, PALLET.dy * S, PALLET.dz * S);
const limitEdges = new THREE.EdgesGeometry(limitGeo);
const limitLine = new THREE.LineSegments(limitEdges, new THREE.LineBasicMaterial({{ color: 0x445566, transparent: true, opacity: 0.4 }}));
limitLine.position.set(cx, cy, cz);
scene.add(limitLine);

// Box meshes
const boxMeshes = [];
const boxData = [];

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

  // Wireframe edges
  const edges = new THREE.EdgesGeometry(geo);
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x000000, transparent: true, opacity: 0.3 }}));
  line.position.copy(mesh.position);
  scene.add(line);

  // Marker: 3D cross for fragility violation (heavy on fragile), sphere for fragile only
  const cx = (box.x + box.dx / 2) * S;
  const cyTop = (box.y + box.dy) * S + 0.005;
  const cz = (box.z + box.dz / 2) * S;
  const markerMat = new THREE.MeshStandardMaterial({{ color: 0xff4444, emissive: 0xff2222, emissiveIntensity: 0.6 }});
  if (box.fragility_violation_bottom) {{
    const r = Math.min(box.dx, box.dz) * S * 0.2;
    const t = r * 0.2;
    const cross = new THREE.Group();
    cross.add(new THREE.Mesh(new THREE.BoxGeometry(r * 2, t, t), markerMat));
    cross.add(new THREE.Mesh(new THREE.BoxGeometry(t, r * 2, t), markerMat));
    cross.add(new THREE.Mesh(new THREE.BoxGeometry(t, t, r * 2), markerMat));
    cross.position.set(cx, cyTop, cz);
    scene.add(cross);
  }} else if (box.fragile) {{
    const markerGeo = new THREE.SphereGeometry(Math.min(box.dx, box.dz) * S * 0.15, 8, 8);
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.set(cx, cyTop, cz);
    scene.add(marker);
  }}

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
  const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: 0x666666, transparent: true, opacity: 0.4 }}));
  line.position.copy(mesh.position);
  scene.add(line);
  boxMeshes.push(mesh);
  boxData.push({{ ...box, unplaced: true }});
}});

// Legend
const legend = document.getElementById('legend');
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
  const label = v.unplaced > 0 ? `${{sku}} (placed: ${{v.placed}}, unplaced: ${{v.unplaced}})` : `${{sku}} (${{v.placed}})`;
  item.innerHTML = `<div class="swatch" style="background:${{v.color}}"></div>${{label}}`;
  legend.appendChild(item);
}});

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


def generate_viz_data(benchmark_results: List[Dict], requests: List[Dict]) -> List[Dict]:
    """Build visualization data from benchmark results and original requests.

    Args:
        benchmark_results: List of result dicts from benchmark.py
        requests: List of original request dicts (pallet + boxes + placements)
    """
    viz_data = []
    for result, request in zip(benchmark_results, requests):
        pallet = request["pallet"]
        boxes_meta = {b["sku_id"]: b for b in request["boxes"]}

        placements = []
        for p in result.get("response", {}).get("placements", []):
            sku = boxes_meta.get(p["sku_id"], {})
            dim = p["dimensions_placed"]
            pos = p["position"]
            placements.append({
                "sku_id": p["sku_id"],
                "x_mm": pos["x_mm"],
                "y_mm": pos["y_mm"],
                "z_mm": pos["z_mm"],
                "length_mm": dim["length_mm"],
                "width_mm": dim["width_mm"],
                "height_mm": dim["height_mm"],
                "fragile": sku.get("fragile", False),
                "weight_kg": sku.get("weight_kg", 0),
            })

        unplaced = result.get("response", {}).get("unplaced", [])
        unplaced_boxes = _layout_unplaced_boxes(unplaced, boxes_meta, pallet)
        viz_data.append({
            "pallet": pallet,
            "placements": placements,
            "unplaced_boxes": unplaced_boxes,
            "meta": {
                "scenario": result["scenario"],
                "score": result.get("final_score", 0),
                "placed": result.get("placed", len(placements)),
                "total_items": result.get("total_items", 0),
            },
        })

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
