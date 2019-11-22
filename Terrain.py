import numpy as np
import cv2
from TerrainOutput import *
from Constants import *


class Terrain:
    def __init__(self, dem):
        # 2D elevation map array
        self.dem = self.load_dem(dem)
        # These will be used to store the terrain outputs as objects once they are created
        self.slope = None
        self.aspect = None
        self.hillshade = None

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
        dem = cv2.imread(file)

        # cv2 doesn't raise errors if the file is invalid
        if not isinstance(dem, np.ndarray):
            raise FileNotFoundError(f'"{file}" is not a valid image.')

        return dem

    def slope(self, algorithm=SlopeAlgorithms.MAXIMUM_DOWNHILL_SLOPE, units=SlopeUnits.DEGREES):
        # Calculate slope and return the image
        slope = Slope(self.dem, algorithm, units)
        # Store the image in the terrain object
        self.slope = slope
        return slope

    def aspect(self):
        pass

    def hillshade(self, azimuth=270, altitude=45):
        pass


terrain = Terrain(dem="example_data\\black_canyon_dem_small.png")
print(terrain)