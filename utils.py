import torch
import rasterio
import numpy as np
from skimage import exposure
from shapely.geometry import Polygon

def load_model(yolov5_path, model_path, device):
    model = torch.hub.load(
        yolov5_path,
        'custom',
        path=model_path,
        source='local',
        _verbose=False,
        device=device
    )
    return model

def calculate_tile_bboxes(
    raster_height: int,
    raster_width: int,
    tile_size: int = 640,
    overlap_ratio: float = 0.0,
    include_edge: bool = False,
):
    tile_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_ratio * tile_size)
    x_overlap = int(overlap_ratio * tile_size)
    while y_max < raster_height:
        x_min = x_max = 0
        y_max = y_min + tile_size
        while x_max < raster_width:
            x_max = x_min + tile_size
            if y_max > raster_height or x_max > raster_width:
                if include_edge:
                    xmax = min(raster_width, x_max)
                    ymax = min(raster_height, y_max)
                    xmin = max(0, xmax - tile_size)
                    ymin = max(0, ymax - tile_size)
                    tile_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                tile_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return tile_bboxes

def preprocess_tile(tile):
    preprocessed_tile_placeholder = np.zeros_like(tile)
    masked_tile = tile[tile != 0]
    p1, p2 = np.percentile(masked_tile, (0.5, 99.5))
    preprocessed_tile = exposure.rescale_intensity(
        image=masked_tile, 
        in_range=(p1, p2), 
        out_range='uint8'
    )
    preprocessed_tile_placeholder[tile != 0] = preprocessed_tile
    return preprocessed_tile_placeholder.astype('uint8')

def shift_preds(tile_bboxes, preds):
    data_array = []
    for i, tile in enumerate(tile_bboxes):
        preds_xyxy = preds.xyxy[i].detach().cpu().numpy()
        preds_xyxy[:, [0, 2]] += tile[0]
        preds_xyxy[:, [1, 3]] += tile[1]
        data_array.extend(preds_xyxy)
    return np.array(data_array)

def geo_tile(src, xmin, ymin, xmax, ymax):
    xmin, ymin = src.xy(ymin,xmin, 'ul')
    xmax, ymax = src.xy(ymax-1,xmax-1, 'lr')
    return Polygon.from_bounds(xmin, ymax, xmax, ymin)


if __name__=='__main__':

    import pandas as pd
    import geopandas as gpd
    from rasterio.windows import Window

    tile_size = 512
    overlap_ratio = 0.1
    include_edge = True

    model = load_model('../yolov5/', '../yolov5/weights/last.pt', 'cpu')
    model.conf = 0.25
    model.iou = 0.45

    with rasterio.open('../yolov5/data/images/test_4326.tif') as src:
        tiles = calculate_tile_bboxes(src.height, src.width, tile_size, overlap_ratio, include_edge)
        windows = [Window(tile[0], tile[1], tile_size, tile_size) for tile in tiles]
        images = [preprocess_tile(src.read(1, window=window)) for window in windows]
        preds = model(images, size=tile_size)

        preds_array = shift_preds(tiles, preds)
        geos = [geo_tile(src, *arr[:4]) for arr in preds_array]
        names = [model.names[int(x)] for x in preds_array[:, -1]]

        df = pd.DataFrame(preds_array).reset_index(drop=True)
        df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class']
        df['name'] = names
        df['geometry'] = geos

        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=src.crs.to_string())
        gdf.to_file('test.geojson', driver='GeoJSON')
