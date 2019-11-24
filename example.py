import matplotlib.pyplot as plt
from Terrain import Terrain
from Constants import SlopeAlgorithms, SlopeUnits

if __name__ == "__main__":
    terrain = Terrain(dem="example_data\\black_canyon_dem_small_16bit.png", cell_resolution=30)
    # aspect = terrain.calculate_aspect()
    # slope = terrain.calculate_slope(algorithm=SlopeAlgorithms.NEIGHBORHOOD, units=SlopeUnits.DEGREES)
    hillshade = terrain.calculate_hillshade(azimuth=210)
    hillshade.save("hillshade_210deg.png")
    # aspect.save('aspect.tif')
    # plt.imshow(slope.array)
    # plt.imshow(aspect.array, cmap="Greys_r")
    # plt.show()
    plt.imshow(hillshade.array, cmap="Greys_r")
    plt.show()
