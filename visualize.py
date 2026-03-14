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


def _generate_html(scenario: Dict[str, Any]) -> str:
    """Generate a standalone HTML file with Three.js 3D visualization."""
    pallet = scenario["pallet"]
    placements = scenario.get("placements", [])
    meta = scenario.get("meta", {})
    scenario_name = meta.get("scenario", "unknown")
    score = meta.get("score", 0)
    placed = meta.get("placed", 0)
    total = meta.get("total_items", 0)

    # Assign colors to SKUs
    sku_ids = sorted(set(p["sku_id"] for p in placements))
    sku_color_map = {sid: _color_for_sku(sid, i) for i, sid in enumerate(sku_ids)}

    # Build JS data
    boxes_js = json.dumps([
        {
            "sku_id": p["sku_id"],
            "x": p["x_mm"], "y": p["z_mm"], "z": p["y_mm"],  # Three.js: Y is up
            "dx": p["length_mm"], "dy": p["height_mm"], "dz": p["width_mm"],
            "color": sku_color_map[p["sku_id"]],
            "fragile": p.get("fragile", False),
        }
        for p in placements
    ])

    pallet_js = json.dumps({
        "dx": pallet["length_mm"],
        "dy": pallet["max_height_mm"],
        "dz": pallet["width_mm"],
    })

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

// Scale to meters for better camera behavior
const S = 1 / 1000;

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

// Camera
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 100);
const cx = PALLET.dx * S / 2, cy = PALLET.dy * S / 2, cz = PALLET.dz * S / 2;
const maxDim = Math.max(PALLET.dx, PALLET.dy, PALLET.dz) * S;
camera.position.set(cx + maxDim * 1.2, cy + maxDim * 0.8, cz + maxDim * 1.2);
camera.lookAt(cx, cy * 0.4, cz);

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

  // Fragile marker
  if (box.fragile) {{
    const markerGeo = new THREE.SphereGeometry(Math.min(box.dx, box.dz) * S * 0.15, 8, 8);
    const markerMat = new THREE.MeshStandardMaterial({{ color: 0xff4444, emissive: 0xff2222, emissiveIntensity: 0.5 }});
    const marker = new THREE.Mesh(markerGeo, markerMat);
    marker.position.set(
      (box.x + box.dx / 2) * S,
      (box.y + box.dy) * S + 0.005,
      (box.z + box.dz / 2) * S
    );
    scene.add(marker);
  }}

  boxMeshes.push(mesh);
  boxData.push(box);
}});

// Legend
const legend = document.getElementById('legend');
const skuSet = new Map();
BOXES.forEach(b => {{
  if (!skuSet.has(b.sku_id)) {{
    const count = BOXES.filter(x => x.sku_id === b.sku_id).length;
    skuSet.set(b.sku_id, {{ color: b.color, count }});
  }}
}});
skuSet.forEach((v, sku) => {{
  const item = document.createElement('div');
  item.className = 'item';
  item.innerHTML = `<div class="swatch" style="background:${{v.color}}"></div>${{sku}} (${{v.count}})`;
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
      tooltip.innerHTML = `<b>${{b.sku_id}}</b><br>${{b.dx}}x${{b.dz}}x${{b.dy}}mm${{b.fragile ? '<br><span style="color:#ff6666">FRAGILE</span>' : ''}}`;
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
            })

        viz_data.append({
            "pallet": pallet,
            "placements": placements,
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
