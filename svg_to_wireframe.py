import xml.etree.ElementTree as ET
import re

def parse_polyline_points(points_str):
    points = []
    for chunk in points_str.strip().replace("\n", " ").split():
        if "," in chunk:
            x_str, y_str = chunk.split(",")
            points.append((float(x_str), float(y_str)))
    return points

def parse_path_d(d_str):
    tokens = re.findall(r'[MLml]|-?\d*\.?\d+(?:[eE][+-]?\d+)?', d_str)
    polylines = []
    idx = 0
    cmd = None
    current_poly = []
    x = y = 0.0

    def flush_poly():
        nonlocal current_poly
        if len(current_poly) >= 2:
            polylines.append(current_poly)
        current_poly = []

    while idx < len(tokens):
        t = tokens[idx]
        if t in ("M", "m", "L", "l"):
            cmd = t
            idx += 1
            continue
        if cmd is None:
            idx += 1
            continue
        if idx + 1 >= len(tokens):
            break
        x_val = float(tokens[idx])
        y_val = float(tokens[idx + 1])
        idx += 2

        if cmd == "M":
            flush_poly()
            x, y = x_val, y_val
            current_poly.append((x, y))
        elif cmd == "m":
            flush_poly()
            x += x_val
            y += y_val
            current_poly.append((x, y))
        elif cmd == "L":
            x, y = x_val, y_val
            current_poly.append((x, y))
        elif cmd == "l":
            x += x_val
            y += y_val
            current_poly.append((x, y))
    flush_poly()
    return polylines

def extract_polylines_from_svg(svg_path):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    polylines = []

    for elem in root.iter():
        tag = elem.tag
        if tag.endswith("polyline") or tag.endswith("polygon"):
            pts_str = elem.attrib.get("points", "")
            pts = parse_polyline_points(pts_str)
            if len(pts) >= 2:
                polylines.append(pts)
        elif tag.endswith("line"):
            x1 = elem.attrib.get("x1")
            y1 = elem.attrib.get("y1")
            x2 = elem.attrib.get("x2")
            y2 = elem.attrib.get("y2")
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                p1 = (float(x1), float(y1))
                p2 = (float(x2), float(y2))
                polylines.append([p1, p2])
        elif tag.endswith("path"):
            d_str = elem.attrib.get("d", "")
            if not d_str:
                continue
            pls = parse_path_d(d_str)
            for p in pls:
                if len(p) >= 2:
                    polylines.append(p)
    return polylines

def build_vertices_edges(polylines, eps=1e-3):
    key_to_index = {}
    vertices = []
    edges_set = set()

    def get_index(pt):
        x, y = pt
        key = (round(x / eps), round(y / eps))
        if key in key_to_index:
            return key_to_index[key]
        idx = len(vertices)
        vertices.append((x, y))
        key_to_index[key] = idx
        return idx

    for poly in polylines:
        idxs = [get_index(pt) for pt in poly]
        for i in range(len(idxs) - 1):
            a = idxs[i]
            b = idxs[i + 1]
            if a == b:
                continue
            if a < b:
                edge = (a, b)
            else:
                edge = (b, a)
            edges_set.add(edge)

    edges = sorted(edges_set)
    return vertices, edges

def normalize_vertices(vertices):
    xs = [x for (x, y) in vertices]
    ys = [y for (x, y) in vertices]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x if max_x != min_x else 1.0
    range_y = max_y - min_y if max_y != min_y else 1.0
    norm = []
    for (x, y) in vertices:
        nx = (x - min_x) / range_x
        ny = (y - min_y) / range_y
        norm.append((nx, ny))
    return norm

def svg_to_wireframe(svg_path, eps=1e-3):
    polylines = extract_polylines_from_svg(svg_path)
    vertices_2d, edges0 = build_vertices_edges(polylines, eps=eps)
    vertices_norm = normalize_vertices(vertices_2d)
    vertices_3d = [(x, y, 0.0) for (x, y) in vertices_norm]
    edges = [(a + 1, b + 1) for (a, b) in edges0]
    return vertices_3d, edges

def main():
    svg_path = "svg/hex2.svg"
    v3d, edges = svg_to_wireframe(svg_path)

    print("V2D = Float32.([")
    for i, (x, y, z) in enumerate(v3d):
        sep = ";" if i < len(v3d) - 1 else ""
        print(f"    {x:.4f} {y:.4f} {z:.4f}{sep}")
    print("])\n")

    print("E2D = [", end="")
    for i, (a, b) in enumerate(edges):
        sep = "," if i < len(edges) - 1 else ""
        print(f"({a},{b}){sep}", end="")
    print("]")

if __name__ == "__main__":
    main()
