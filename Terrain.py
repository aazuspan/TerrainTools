from matplotlib import pyplot as plt
import cv2
from TerrainOutput import *
from Constants import *


class Terrain:
    def __init__(self, dem, cell_resolution):
        # 2D elevation map array
        self.dem = self.load_dem(dem)
        # TODO: Detect this attribute for georeferenced DEMs
        # Spatial resolution of the cells
        self.cell_resolution = cell_resolution

    @property
    def width(self):
        return self.dem.shape[0]

    @property
    def height(self):
        return self.dem.shape[1]

    def __repr__(self):
        return f"<{self.__class__.__module__}.{self.__class__.__name__} size={self.dem.shape[0], self.dem.shape[1]} " \
               f"at 0x{id(self)}>"

    def load_dem(self, file):
        """
        Return a 2D numpy array from a DEM image

        :param file: Path to the DEM image file
        :return: A 2D numpy array created from the DEM image
        """
        dem = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # cv2 doesn't raise errors if the file is invalid
        if not isinstance(dem, np.ndarray):
            raise FileNotFoundError(f'"{file}" is not a valid image.')

        return dem

    def calculate_slope(self, algorithm=SlopeAlgorithms.MAXIMUM_DOWNHILL_SLOPE, units=SlopeUnits.DEGREES):
        """
        Calculate a slope map from the elevation map

        :param algorithm: Algorithm used to calculate slope.
        :param units: Units of cell values in the returned array. Either degrees or percent slope.
        :return: A 2D numpy array of slope values for the DEM
        """
        slope_array = Slope(self.dem, self.cell_resolution, algorithm, units)
        return slope_array

    def calculate_aspect(self):
        """
        Calculate an aspect map from the elevation map

        :return: A 2D numpy array of aspect values in degrees for the DEM
        """
        aspect_array = Aspect(self.dem, self.cell_resolution)
        return aspect_array

    def calculate_hillshade(self, azimuth=270, altitude=45):
        raise NotImplementedError


if __name__ == "__main__":
    terrain = Terrain(dem="example_data\\black_canyon_dem_small.png", cell_resolution=30)
    aspect = terrain.calculate_aspect()
    slope = terrain.calculate_slope(algorithm=SlopeAlgorithms.NEIGHBORHOOD, units=SlopeUnits.DEGREES)

    plt.imshow(slope.array)
    plt.imshow(aspect.array)
    plt.show()
