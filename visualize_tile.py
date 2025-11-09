import mapbox_vector_tile
import mercantile
from shapely.geometry import shape
from shapely.ops import transform
import geopandas as gpd

def tile_transform_fn(x_tile, y_tile, z_tile, extent=4096):
    west, south, east, north = mercantile.bounds(x_tile, y_tile, z_tile)
    dx = (east - west) / extent
    dy = (north - south) / extent
    def fn(x, y, z=None):
        lon = west + x * dx
        lat = north - y * dy
        return lon, lat
    return fn

def mvt_to_gdf(mvt_bytes, z, x, y):
    decoded = mapbox_vector_tile.decode(mvt_bytes)  # layer -> dict
    extent = next(iter(decoded.values())).get("extent", 4096) if decoded else 4096
    tf = tile_transform_fn(x, y, z, extent)
    rows = []
    for layer_name, layer in decoded.items():
        for feat in layer.get("features", []):
            geom = feat.get("geometry")
            if not geom:
                continue
            shp = shape(geom)                 # tile-local coords
            shp_ll = transform(tf, shp)      # lon/lat WGS84
            props = dict(feat.get("properties", {}))
            props["_layer"] = layer_name
            rows.append({"geometry": shp_ll, **props})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")

# usage
with open("tile.vector.pbf","rb") as f:
    mvt = f.read()
gdf = mvt_to_gdf(mvt, z=14, x=4823, y=6160)
print(gdf.head())























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



                # # Handle polygon-like geometries (outer ring first)
                # if isinstance(coords, list) and isinstance(coords[0], list) and isinstance(coords[0][0], list):
                #     # If gtype contains 'polygon' or coords is list-of-rings
                #     if isinstance(coords[0][0], (list, tuple)):
                #         rings = coords  # rings: [outer_ring, hole1, hole2...]
                #     else:
                #         # maybe single ring
                #         rings = [coords]
                #     print(rings)
          
                #     outer = rings[0]
                #     if len(outer) < 3:
                #         continue
                    
                #     print('passed outer')

                #     # create top and bottom vertices
                #     top_indices = []
                #     bottom_indices = []
                #     for x, y in outer:
                #         vx = x * scale
                #         vy = y * scale
                #         vz_top = h
                #         vz_bot = 0.0
                #         obj_lines.append(f"v {vx:.6f} {vy:.6f} {vz_top:.6f}\n")
                #         obj_lines.append(f"v {vx:.6f} {vy:.6f} {vz_bot:.6f}\n")
                #         top_idx = vertex_offset + vcount + 1
                #         bottom_idx = vertex_offset + vcount + 2
                #         top_indices.append(top_idx)
                #         bottom_indices.append(bottom_idx)
                #         vcount += 2

                #     # top face (outer ring) - note: winding preserved as given
                #     obj_lines.append("g poly_top\n")
                #     obj_lines.append("f " + " ".join(str(i) for i in top_indices) + "\n")
                #     # bottom face (reverse order to keep normals consistent)
                #     obj_lines.append("g poly_bottom\n")
                #     obj_lines.append("f " + " ".join(str(i) for i in reversed(bottom_indices)) + "\n")
                #     # side faces (quad between top and bottom pairs)
                #     obj_lines.append("g poly_sides\n")
                #     L = len(top_indices)
                #     for i in range(L):
                #         a_top = top_indices[i]
                #         b_top = top_indices[(i + 1) % L]
                #         b_bot = bottom_indices[(i + 1) % L]
                #         a_bot = bottom_indices[i]
                #         obj_lines.append(f"f {a_top} {b_top} {b_bot} {a_bot}\n")
        

        w,h = 4096,4096  # default extent
        img = PIL.Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(img)


        for layer, items in all_types.items():
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            
            for gtype, coords in items:
                # print(f"Layer: {layer}, Geometry Type: {gtype}, Coordinates: {coords}, Color: {color}")
                if gtype == 'point':
                    draw.ellipse((coords[0]-5, coords[1]-5, coords[0]+5, coords[1]+5), fill=color)
                elif gtype == 'multipoint':
                    for pt in coords:
                        draw.ellipse((pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5), fill=color)
                elif gtype == 'linestring':
                    draw.line(coords, fill=color, width=2)
                elif gtype == 'multilinestring':
                    for line in coords:
                        draw.line(line, fill=color, width=2)
                elif gtype == 'polygon':
                    draw.polygon(coords[0], outline=color, fill=None)
                else:
                    print(f"Unknown geometry type: {gtype}")
        img.save(input_path + "debug_output.png")
        print ("Saved debug image to", input_path + "debug_output.png")

        return vcount

    # Build list of .mvt files
    import os
    files = []
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.mvt"))
    elif os.path.isfile(input_path):
        files = [input_path]
    else:
        raise FileNotFoundError(f"{input_path} not found")

    if not files:
        raise FileNotFoundError("No .mvt files found in input")

    if out_obj is None:
        if os.path.isdir(input_path):
            out_obj = os.path.join(input_path, "merged.obj")
        else:
            out_obj = os.path.splitext(input_path)[0] + ".obj"



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
