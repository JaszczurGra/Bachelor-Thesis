import random
from tkinter import Image
import PIL
from PIL import Image,ImageDraw
import requests
import os


def load_token():
    if not os.path.exists("token.txt"):
        raise FileNotFoundError("token.txt file not found")
    with open("token.txt", "r") as f:
        token = f.read().strip()
        if not token:
            raise ValueError("API token is empty")
    return token

def get_vector_tile(zoom, x, y, format, token):
    url = f"https://api.mapbox.com/v4/mapbox.mapbox-streets-v8/{zoom}/{x}/{y}.{format}?access_token={token}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()  # raise for 4xx/5xx
        data = resp.content  # binary content of the vector tile
    except requests.RequestException as e:
        print("HTTP error:", e)
        return None
    return data

def get_raster_tile(zoom, x, y, format, token): 
    # @1x for 512x512 tiles @2x for 1024x1024 tiles

    # zoom 19 for city 
    # zoom 15 for other 

    tileset_id = "mapbox.satellite"
    # tileset_id = "mapbox.terrain-rgb"
    # tileset_id = "mapbox.mapbox-terrain-v2"
    url = f"https://api.mapbox.com/v4/{tileset_id}/{zoom}/{x}/{y}@2x.{format}?access_token={token}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()  # raise for 4xx/5xx
        data = resp.content  # binary content of the vector tile
    except requests.RequestException as e:
        print("HTTP error:", e)
        return None
    return data





def generate_tile(zoom, x, y, token,output_dir="tiles"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, f"tile_{zoom}_{x}_{y}.mvt"), "wb") as f:
        vector_data = get_vector_tile(zoom, x, y,"mvt", token)
        if vector_data:
            f.write(vector_data)
            print(f"Saved vector tile to {f.name}")
    with open(os.path.join(output_dir, f"tile_{zoom}_{x}_{y}.jpg"), "wb") as f:
        raster_data = get_raster_tile(zoom, x, y,"jpg", token)
        if raster_data:
            f.write(raster_data)
            print(f"Saved raster tile to {f.name}")


def log_lat_to_tile_coords(lat, lon, zoom):
    """
    Convert latitude and longitude to tile x, y coordinates at a given zoom level.
    """
    import math
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(math.radians(lat)) + (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def merge_images(dir_path, tile_size=(512, 512)):
    #TODO better parsing of files x,y
    from PIL import Image
    
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    
    minimum = (float('inf'), float('inf'))
    maximum = (float('-inf'), float('-inf'))
    n = 0    

    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            x = int(filename.split('_')[2])
            y = int(filename.split('_')[3].split('.')[0])
            minimum = (min(minimum[0], x), min(minimum[1], y))
            maximum = (max(maximum[0], x), max(maximum[1], y))
            n += 1 

    w, h = maximum[0] - minimum[0] + 1, maximum[1] - minimum[1] + 1

    if w * h != n:
        raise ValueError("Tiles do not form a complete rectangle")

    merged_image = Image.new('RGB', (w * tile_size[0], h * tile_size[1]))


    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            tile_image = Image.open(os.path.join(dir_path, filename))
            x = int(filename.split('_')[2]) - minimum[0]
            y = int(filename.split('_')[3].split('.')[0]) - minimum[1]
            merged_image.paste(tile_image, (x * tile_size[0], y * tile_size[1]))

    return merged_image

def merge_and_save_image(dir_path, tile_size=(512, 512)):
    merged_image = merge_images(dir_path, tile_size)
    out_file = os.path.join(dir_path, "merged_image.jpg")
    merged_image.save(out_file)
    print(f"Merged image saved to {out_file}")

def generate_and_save_tiles(lat, lon, zoom, x_size, y_size, token, output_dir="a"):
    x, y = log_lat_to_tile_coords(lat, lon, zoom)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for xi in range (x_size):
        for yi in range (y_size):
            generate_tile(zoom, x + xi - x_size//2, y + yi - y_size//2, token, output_dir=output_dir)
            print(f"Generating {xi * y_size + yi + 1} / {x_size * y_size}")


def write_obj_with_texture(obj_path: str, image_path: str, out_obj_path: str | None = None, mtl_name: str | None = None):
    """
    Attach a planar texture (image) to an existing OBJ by adding UVs and an MTL that references the image.
    - obj_path: input OBJ (vertices present as "v x y z")
    - image_path: raster image to use as texture (will be referenced relatively in MTL)
    - out_obj_path: path to write new OBJ (defaults to obj_path with _tex.obj)
    - mtl_name: filename for the material (defaults to <outname>.mtl)
    Result: writes new OBJ and MTL next to out_obj_path and returns (out_obj_path, out_mtl_path).
    Notes:
    - UVs are calculated from X,Y vertex coordinates (normalized to bounding box).
    - This produces vt entries (one per vertex) and rewrites faces to use v/vt.
    - Works for faces that currently list only vertex indices (f v1 v2 v3 ...).
    """
    import os
    from shutil import copy2

    if out_obj_path is None:
        base, ext = os.path.splitext(obj_path)
        out_obj_path = base + "_tex.obj"
    if mtl_name is None:
        mtl_name = os.path.basename(os.path.splitext(out_obj_path)[0] + ".mtl")
    out_mtl_path = os.path.join(os.path.dirname(out_obj_path), mtl_name)

    # read input OBJ
    with open(obj_path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    v_positions = []
    other_lines = []
    face_lines = []
    header_lines = []

    for ln in lines:
        if ln.startswith("v "):
            parts = ln.strip().split()
            if len(parts) >= 4:
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                v_positions.append((x, y, z))
            else:
                v_positions.append((0.0, 0.0, 0.0))
        elif ln.startswith("f "):
            face_lines.append(ln.rstrip("\n"))
        elif ln.startswith("mtllib ") or ln.startswith("usemtl "):
            # skip existing material references
            continue
        else:
            header_lines.append(ln.rstrip("\n"))

    if not v_positions:
        raise RuntimeError("No vertex positions found in OBJ.")

    xs = [p[0] for p in v_positions]
    ys = [p[1] for p in v_positions]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    dx = maxx - minx if maxx > minx else 1.0
    dy = maxy - miny if maxy > miny else 1.0

    # build new OBJ contents
    out_lines = []
    out_lines.append(f"mtllib {mtl_name}\n")
    out_lines.append("usemtl texmat\n")
    out_lines.append("# header\n")
    for hl in header_lines:
        out_lines.append(hl + "\n")

    # write original vertex lines (keep as-is)
    # re-open input to preserve exact vertex formatting
    with open(obj_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("v "):
                out_lines.append(ln)
    # write vt for each vertex (u,v) mapped from x,y -> [0,1], flip v so image top aligns with maxy
    for (x, y, z) in v_positions:
        u = (x - minx) / dx
        v = 1.0 - ((y - miny) / dy)  # flip v
        out_lines.append(f"vt {u:.6f} {v:.6f}\n")

    # copy non-face, non-vertex lines (except we already wrote header), preserve groups/comments
    # then rewrite faces replacing "n" with "n/n"
    with open(obj_path, "r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.startswith("v ") or ln.startswith("f ") or ln.startswith("mtllib ") or ln.startswith("usemtl "):
                continue
            # include other lines (o, g, comments, vn etc.)
            out_lines.append(ln)

    # rewrite faces: for tokens without slashes, make v/vt
    for fl in face_lines:
        parts = fl.strip().split()
        tokens = parts[1:]
        new_tokens = []
        for t in tokens:
            if "/" in t:
                # leave as-is (or ensure vt present)
                # simple approach: if "v" only, not common here
                new_tokens.append(t)
            else:
                # assume vertex index
                new_tokens.append(f"{t}/{t}")
        out_lines.append("f " + " ".join(new_tokens) + "\n")

    # write OBJ and MTL + copy texture image
    os.makedirs(os.path.dirname(out_obj_path) or ".", exist_ok=True)
    with open(out_obj_path, "w", encoding="utf-8") as outfh:
        outfh.writelines(out_lines)

    # write simple MTL referencing the image basename
    image_basename = os.path.basename(image_path)
    mtl_contents = [
        "newmtl texmat\n",
        "Ka 1.000 1.000 1.000\n",
        "Kd 1.000 1.000 1.000\n",
        "Ks 0.000 0.000 0.000\n",
        f"map_Kd {image_basename}\n"
    ]
    with open(out_mtl_path, "w", encoding="utf-8") as mfh:
        mfh.writelines(mtl_contents)

    # copy image next to OBJ
    copy2(image_path, os.path.join(os.path.dirname(out_obj_path), image_basename))

    print(f"Wrote textured OBJ: {out_obj_path} and MTL: {out_mtl_path}")
    return out_obj_path, out_mtl_path
# ...existing code...

def merge_meshes_to_obj(input_path, out_obj=None, default_height=5.0, scale=1.0):
    """
    Convert one .mvt file or a directory of .mvt files into an OBJ mesh.
    - input_path: .mvt file or directory with .mvt files
    - out_obj: output .obj filepath (if None, derived from input_path)
    - default_height: extrusion height for features without height property
    - scale: multiplier for X/Y coords (tile local units -> scene units)
    Notes:
    - Uses mapbox-vector-tile (pip install mapbox-vector-tile)
    - This is a simple exporter: outer polygon rings become faces; holes are ignored.
    """
    #TODO uninstall mapbox-vector-tile and use maplibre vector tile decoder?
    #TODO 
    try:
        from mapbox_vector_tile import decode
    except Exception as e:
        raise ImportError("Install mapbox-vector-tile (pip install mapbox-vector-tile)") from e

    import glob

    def process_mvt_bytes(mvt_bytes, obj_lines, vertex_offset):
        tile = decode(mvt_bytes)  # dict of layers

        print(tile)

        all_types = {}
        



        vcount = 0
        for layer_name, layer in tile.items():
            features = layer.get("features", [])
            for feat in features:
                geom = feat.get("geometry") or feat.get("geometry", None)
                props = feat.get("properties", {}) or feat.get("tags", {}) or {}
                feat_type = feat.get("type", None)
                # decode output often has 'geometry' as GeoJSON-like dict
                coords = None
                if isinstance(geom, dict) and "coordinates" in geom:
                    coords = geom["coordinates"]
                    gtype = geom.get("type", "").lower()
                else:
                    # fallback: some decoders put geometry directly as coordinates
                    coords = geom
                    gtype = feat_type
                if not coords:
                    continue

                # Determine extrusion height
                h = None
                for key in ("height", "building:height", "ele", "z"):
                    if key in props:
                        try:
                            h = float(props[key])
                            break
                        except Exception:
                            pass
                if h is None:
                    h = float(default_height)

                # multiline string, polygon, point, linestring

                # print(f"Processing feature in layer '{layer_name}' with height {h} and geometry type '{gtype}'")

                all_types[layer_name] = all_types.get(layer_name, []) + [(gtype, coords)]
        

        #TODO set this extend from tile
        w,h = 4096,4096  # default extent
        img = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)


        draw.line([(0,0),(w,0),(w,h),(0,h),(0,0)], fill=(255,255,255), width=5)

        for layer, items in all_types.items():
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            
            for gtype, coords in items:
                # print(f"Layer: {layer}, Geometry Type: {gtype}, Coordinates: {coords}, Color: {color}")
                if gtype == 'point':
                    draw.ellipse((coords[0]-15 , coords[1]-15, coords[0]+15, coords[1]+15), fill=color)
                elif gtype == 'multipoint':
                    for pt in coords:
                        draw.ellipse((pt[0]-15, pt[1]-15, pt[0]+15, pt[1]+15), fill=color)
                elif gtype == 'linestring':
                    draw.line(coords, fill=color, width=15)
                elif gtype == 'multilinestring':
                    for line in coords:
                        draw.line(line, fill=color, width=15)
                elif gtype == 'polygon':
                    draw.polygon(coords[0], outline=color, fill=color)
                else:
                    print(f"Unknown geometry type: {gtype}")
        # img.save(input_path + "debug_output.png")
        # print ("Saved debug image to", input_path + "debug_output.png")
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        return img



    if not os.path.exists(input_path):
            raise FileNotFoundError(f"Directory {input_path} does not exist")
        
    minimum = (float('inf'), float('inf'))
    maximum = (float('-inf'), float('-inf'))
    n = 0    

    for filename in os.listdir(input_path):
        if filename.endswith(".mvt"):
            x = int(filename.split('_')[2])
            y = int(filename.split('_')[3].split('.')[0])
            minimum = (min(minimum[0], x), min(minimum[1], y))
            maximum = (max(maximum[0], x), max(maximum[1], y))
            n += 1 

    w, h = maximum[0] - minimum[0] + 1, maximum[1] - minimum[1] + 1

    if w * h != n:
        raise ValueError("Tiles do not form a complete rectangle")

    #changed to extend 
    merged_image = Image.new('RGBA', (w * 4096, h * 4096))


    for filename in os.listdir(input_path):
        if filename.endswith(".mvt"):
            print("Processing", filename)
            tile_image = process_mvt_bytes(open(os.path.join(input_path, filename), "rb").read(), [], 0)
            x = int(filename.split('_')[2]) - minimum[0]
            y = int(filename.split('_')[3].split('.')[0]) - minimum[1]
            merged_image.paste(tile_image, (x * 4096, y * 4096))


    merged_image.save(input_path + "/debug_output.png")

    return merged_image

    # # Build list of .mvt files
    # import os
    # files = []
    # if os.path.isdir(input_path):
    #     files = glob.glob(os.path.join(input_path, "*.mvt"))
    # elif os.path.isfile(input_path):
    #     files = [input_path]
    # else:
    #     raise FileNotFoundError(f"{input_path} not found")

    # if not files:
    #     raise FileNotFoundError("No .mvt files found in input")

    # if out_obj is None:
    #     if os.path.isdir(input_path):
    #         out_obj = os.path.join(input_path, "merged.obj")
    #     else:
    #         out_obj = os.path.splitext(input_path)[0] + ".obj"



    obj_lines = []
    vertex_offset = 0
    for fpath in files:
        print(f"Processing {fpath}...")
        with open(fpath, "rb") as f:
            mvt_bytes = f.read()
        added = process_mvt_bytes(mvt_bytes, obj_lines, vertex_offset)
        vertex_offset += added

    # write file
    with open(out_obj, "w", encoding="utf-8") as out:
        out.write("# OBJ generated from MVT\n")
        out.writelines(obj_lines)

    print(f"Wrote OBJ to {out_obj}")
    return out_obj

def main():
    #TODO change to enviromental variable the token 
    #TODO add _1, _2 for the subsequent folders 
    #TODO maybe swithc to static images for images 
    locations = {
        "San Francisco": (37.7749, -122.4194),
        "Zalasewo": (52.38211111,17.07336111),
        "Malta jezioro": (52.402, 16.977),
        "Kubus House": (52.1946,18.2990)
    }


    location = "Kubus House"
    x_size = 5
    y_size = 5
    zoom  = 19  # devides by this in x and y direction the whole map 
    if location not in locations:
        raise ValueError(f"Location {location} not recognized")
    
    lat, lon = locations[location]
    output_dir = f"data/{location}{zoom}_{x_size}x{y_size}_2"
    token = load_token()

    # TODO swithc with argparser 

    generate_and_save_tiles(lat, lon, zoom, x_size, y_size, token, output_dir=output_dir)
    merge_and_save_image(output_dir)
    # merge_meshes_to_obj(output_dir,output_dir+"/merged.obj")

    





if __name__ == "__main__":
    main()





# parser = argparse.ArgumentParser(description="Generate and/or merge tiles.")
# parser.add_argument("--lat", type=float, default=37.7749, help="Center latitude for generate")
# parser.add_argument("--lon", type=float, default=-122.4194, help="Center longitude for generate")
# parser.add_argument("--zoom", "-z", type=int, default=19, help="Zoom level")
# parser.add_argument("--x-size", type=int, default=5, help="Number of tiles in x direction")
# parser.add_argument("--y-size", type=int, default=5, help="Number of tiles in y direction")
# parser.add_argument("--output-dir", "-o", default=None, help="Output directory for tiles (default data/Center_Z)")
# parser.add_argument("--out", default="merged.jpg", help="Merged output filename (saved in output-dir)")
# parser.add_argument("--generate", action="store_true", help="Only generate tiles")
# parser.add_argument("--merge", action="store_true", help="Only merge existing tiles")
# parser.add_argument("--tile-w", type=int, default=None, help="Tile width override (pixels)")
# parser.add_argument("--tile-h", type=int, default=None, help="Tile height override (pixels)")
# args = parser.parse_args()

# @click.group()
# def cli():
#     """Tile utilities: generate tiles and merge tiles."""
#     pass


# @cli.command("generate")
# @click.option("--lat", type=float, required=True, help="Center latitude")
# @click.option("--lon", type=float, required=True, help="Center longitude")
# @click.option("--zoom", "-z", type=int, required=True, help="Tile zoom level")
# @click.option("--x-size", type=int, default=1, help="Number of tiles in x direction (cols)")
# @click.option("--y-size", type=int, default=1, help="Number of tiles in y direction (rows)")
# @click.option("--output-dir", "-o", default="tiles", help="Output directory")
# @click.option("--token", "-t", default=None, help="Mapbox token (env API_TOKEN or token.txt used if omitted)")
# def generate_cmd(lat, lon, zoom, x_size, y_size, output_dir, token):
#     """Generate a grid of tiles around given lat/lon."""
#     if token is None:
#         token = load_token()
#     cx, cy = log_lat_to_tile_coords(lat, lon, zoom)
#     os.makedirs(output_dir, exist_ok=True)
#     for xi in range(x_size):
#         for yi in range(y_size):
#             tx = cx + xi - x_size // 2
#             ty = cy + yi - y_size // 2
#             generate_tile(zoom, tx, ty, token, output_dir=output_dir)


# @cli.command("merge")
# @click.option("--dir", "dir_path", type=click.Path(exists=True), required=True, help="Directory with tiles")
# @click.option("--zoom", "-z", type=int, required=True, help="Zoom level of tiles")
# @click.option("--x-start", type=int, required=True, help="Start tile x")
# @click.option("--y-start", type=int, required=True, help="Start tile y")
# @click.option("--cols", type=int, required=True, help="Number of columns")
# @click.option("--rows", type=int, required=True, help="Number of rows")
# @click.option("--tile-w", type=int, default=512, help="Tile width in pixels")
# @click.option("--tile-h", type=int, default=512, help="Tile height in pixels")
# @click.option("--out", "out_file", default="merged.jpg", help="Output filename")
# def merge_cmd(dir_path, zoom, x_start, y_start, cols, rows, tile_w, tile_h, out_file):
#     """Merge a rectangular grid of raster tiles into one image."""
#     merge_tiles(dir_path, zoom, x_start, y_start, cols, rows, tile_size=(tile_w, tile_h), out_file=out_file)

