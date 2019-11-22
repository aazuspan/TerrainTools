from abc import ABCMeta, abstractmethod


# Abstract class for all terrain outputs (aspect, slope, hillshade, etc)
class TerrainOutput(metaclass=ABCMeta):
    def __init__(self, dem):
        self.dem = dem
        self.array = self.generate()

    @abstractmethod
    def generate(self):
        # Generate whatever terrain output it is
        return

    def save(self, filename):
        # Save the array as the filename
        pass

    def get_neighbours(self, row, col):
        """
        Get DEM cell values from Moore neighbours of a given coordinate in order z1 - z9. Ignore out of bounds cells.

        :param row: Y coordinate to get neighbours of
        :param col: X coordinate to get neighbours of
        :return: A list of up to 9 cell values from the DEM
        """
        # Moore offset tuples (row, col) in the order z1 - z9 presented by Hickey 1998
        offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (0, 0))

        neighbours = []
        for offset in offsets:
            if not self.is_out_of_bounds(row=row + offset[0], col=col + offset[1]):
                neighbours.append(self.dem[row + offset[0]][col + offset[1]])

        return neighbours

    def is_out_of_bounds(self, row, col):
        """
        Test whether a coordinate is out of bounds of the DEM
        :param row: Y coordinate to test
        :param col: X coordinate to test
        :return: True if coordinate is outside of the DEM, false is coordinate is within the DEM
        """

        if row < 0 or row > self.dem.height - 1 or col < 0 or col > self.dem.width - 1:
            return True
        return False

class Slope(TerrainOutput):
    def __init__(self, dem, algorithm, units):
        self.algorithm = algorithm
        self.units = units
        super().__init__(dem)

    def generate(self):
        return


class Aspect(TerrainOutput):
    def __init__(self, dem):
        super().__init__(dem)

    def generate(self):
        return


class Hillshade(TerrainOutput):
    def __init__(self, dem, azimuth, altitude):
        super().__init__(dem)

    def generate(self):
        return
