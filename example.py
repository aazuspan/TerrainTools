import matplotlib.pyplot as plt
from Terrain import Terrain
from Constants import SlopeAlgorithms, SlopeUnits


if __name__ == "__main__":
    # Create a terrain object from a Digital Elevation Model
    terrain = Terrain(dem="example_data\\black_canyon_dem_small.png", cell_resolution=30)

    # Calculate the terrain outputs from the input DEM
    slope = terrain.calculate_slope(algorithm=SlopeAlgorithms.NEIGHBORHOOD, units=SlopeUnits.DEGREES)
    aspect = terrain.calculate_aspect()
    hillshade = terrain.calculate_hillshade(azimuth=315)

    # Preview the terrain outputs using pyplot
    plt.imshow(slope.array)
    plt.show()
    plt.imshow(aspect.array)
    plt.show()
    plt.imshow(hillshade.array, cmap="Greys_r")
    plt.show()

    # Save the terrain outputs as local files
    slope.save("example_data\\output\\slope.png")
    aspect.save("example_data\\output\\aspect.png")
    hillshade.save("example_data\\output\\hillshade.png")
