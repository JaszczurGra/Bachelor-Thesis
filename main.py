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


def merge_tiles(dir_path, tile_size=(512, 512)):
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

def merge_and_save(dir_path, tile_size=(512, 512)):
    merged_image = merge_tiles(dir_path, tile_size)
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
def main():
    token = load_token()




    lat, lon = 37.7749, -122.4194  # San Francisco, CA
    # lat, lon = 56.38211111,17.07336111 # Zalasewo
    # lat, lon = 56.402, 16.977 #Malta jezioro
  
    x_size = 15
    y_size = 15
    zoom  = 19   # devides by this in x and y direction the whole map 
    output_dir = f"data/SanFrancisco_{zoom}_{x_size}x{y_size}"


    
    
    
    generate_and_save_tiles(lat, lon, zoom, x_size, y_size, token, output_dir=output_dir)
    merge_and_save(output_dir)
    





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

