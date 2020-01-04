import matplotlib.pyplot as plt
from Terrain import Terrain
from Constants import SlopeAlgorithms, SlopeUnits


if __name__ == "__main__":
    # Create a terrain object from a Digital Elevation Model
    terrain = Terrain(dem="example_data\\Alabama_NED.tif", cell_resolution=30)

    # Calculate the terrain outputs from the input DEM
    slope = terrain.calculate_slope(algorithm=SlopeAlgorithms.NEIGHBORHOOD, units=SlopeUnits.DEGREES)
    aspect = terrain.calculate_aspect()
    hillshade = terrain.calculate_hillshade(azimuth=315)
    elevation_profile = terrain.create_elevation_profile(pt1=(33.5, -87), pt2=(31.7, -86))

    # Preview the terrain outputs using pyplot
    slope.preview()
    aspect.preview()
    hillshade.preview()
    elevation_profile.preview()

    # # Save the terrain outputs as local files
    slope.save("example_data\\output\\slope.png")
    aspect.save("example_data\\output\\aspect.png")
    hillshade.save("example_data\\output\\hillshade.png")
    elevation_profile.save("example_data\\output\\elevprof.png")
