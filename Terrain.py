from .TerrainOutput import *
from .Constants import *


class Terrain:
    def __init__(self, dem):
        self.dem = self.load_dem(dem)
        # These will be used to store the terrain outputs as objects once they are created
        self.slope = None
        self.aspect = None
        self.hillshade = None

    def load_dem(self, file):
        # Do some validation and convert it to an Image or Numpy array probably
        dem = file
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
