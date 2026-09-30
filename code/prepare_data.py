
# Note:
# Generative AI was used to assist with code development,
# debugging, and refinement.


from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import numpy as np


# Project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANDSLIDE_FILE = DATA_DIR / "Landslide Point" / "ls_center.gpkg"
NON_LANDSLIDE_FILE = DATA_DIR / "Landslide Point" / "non_ls.gpkg"

RASTER_DIR = DATA_DIR / "Landslide Controlling Factors"

OUTPUT_FILE = OUTPUT_DIR / "landslide_ml.csv"



# Predictor rasters

RASTERS = {
    "aspect": RASTER_DIR / "map_aspect.tif",
    "distance_river": RASTER_DIR / "map_distriv.tif",
    "distance_road": RASTER_DIR / "map_distroad.tif",
    "elevation": RASTER_DIR / "map_elev.tif",
    "lithology": RASTER_DIR / "map_geologi.tif",
    "land_use": RASTER_DIR / "map_landuse.tif",
    "plan_curvature": RASTER_DIR / "map_planc.tif",
    "profile_curvature": RASTER_DIR / "map_profc.tif",
    "slope": RASTER_DIR / "map_slope.tif",
    "spi": RASTER_DIR / "map_spi.tif",
    "twi": RASTER_DIR / "map_twi.tif",
}



# Check raster alignment

reference_name = next(iter(RASTERS))
reference_path = RASTERS[reference_name]

with rasterio.open(reference_path) as ref:
    reference_crs = ref.crs
    reference_transform = ref.transform
    reference_width = ref.width
    reference_height = ref.height

for feature_name, raster_path in RASTERS.items():
    with rasterio.open(raster_path) as src:

        if src.crs != reference_crs:
            raise ValueError(
                f"{feature_name}: CRS differs from reference raster."
            )

        if src.width != reference_width or src.height != reference_height:
            raise ValueError(
                f"{feature_name}: Raster dimensions differ from reference raster."
            )

        if not np.allclose(
            tuple(src.transform),
            tuple(reference_transform),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError(
                f"{feature_name}: Raster grid/transform differs from reference raster."
            )

print("All rasters use the same CRS, grid and dimensions.")


# Load and label point data

print("Loading point data...")

landslides = gpd.read_file(LANDSLIDE_FILE)
non_landslides = gpd.read_file(NON_LANDSLIDE_FILE)

landslides["landslide"] = 1
non_landslides["landslide"] = 0

print(f"Landslide points:     {len(landslides)}")
print(f"Non-landslide points: {len(non_landslides)}")

print(f"Landslide CRS:     {landslides.crs}")
print(f"Non-landslide CRS: {non_landslides.crs}")


# Check point CRS

if landslides.crs != non_landslides.crs:
    raise ValueError(
        "Landslide and non-landslide point datasets use different CRS."
    )



# Merge point datasets

points = pd.concat(
    [landslides, non_landslides],
    ignore_index=True,
)

points = gpd.GeoDataFrame(
    points,
    geometry="geometry",
    crs=landslides.crs,
)

print(f"Combined samples: {len(points)}")



# Extract raster values at point locations

for feature_name, raster_path in RASTERS.items():

    print(f"Extracting: {feature_name}")

    

    with rasterio.open(raster_path) as src:

        print(feature_name, "NoData:", src.nodata)

        # Transform points to the raster CRS if necessary
        if points.crs != src.crs:
            points_for_raster = points.to_crs(src.crs)
        else:
            points_for_raster = points

        coordinates = [
            (geom.x, geom.y)
            for geom in points_for_raster.geometry
        ]

        sampled_values = [
            value[0]
            for value in src.sample(coordinates)
        ]

        points[feature_name] = sampled_values

        if src.nodata is not None:
            points.loc[
                points[feature_name] == src.nodata,
                feature_name
            ] = pd.NA




# Prepare output table

points["x"] = points.geometry.x
points["y"] = points.geometry.y

columns = [
    "landslide",
    "x",
    "y",
    *RASTERS.keys(),
]

output = points[columns].copy()



# Save processed dataset

output.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Finished.")
print(f"Saved dataset to:")
print(OUTPUT_FILE)

print()
print("Dataset shape:")
print(output.shape)

print()
print("Missing values:")
print(output.isna().sum())