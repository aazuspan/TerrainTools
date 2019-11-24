from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import math
from Constants import SlopeAlgorithms, SlopeUnits


# Abstract class for all terrain outputs (aspect, slope, hillshade, etc)
class TerrainOutput(metaclass=ABCMeta):
    def __init__(self, dem, cell_resolution):
        self.dem = dem
        self.cell_resolution = cell_resolution
        self.array = self.generate()

    @abstractmethod
    def generate(self):
        return

    def save(self, filename):
        """
        Save the current slope array as an image

        :param filename: File name and path to save image to
        """
        cv2.imwrite(filename, self.array)

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

        if row < 0 or row > self.dem.shape[0] - 1 or col < 0 or col > self.dem.shape[1] - 1:
            return True
        return False


class Slope(TerrainOutput):
    def __init__(self, dem, cell_resolution, algorithm, units):
        if algorithm not in (SlopeAlgorithms.NEIGHBORHOOD,
                             SlopeAlgorithms.MAXIMUM_SLOPE,
                             SlopeAlgorithms.MAXIMUM_DOWNHILL_SLOPE,
                             SlopeAlgorithms.QUADRATIC_SURFACE):
            raise ValueError('Slope algorithm is invalid')
        else:
            self.algorithm = algorithm

        if units not in (SlopeUnits.DEGREES,
                         SlopeUnits.PERCENT):
            raise ValueError('Slope units are invalid')
        else:
            self.units = units

        super().__init__(dem, cell_resolution)

    def generate(self):
        """
        Generate a slope map using the given algorithm in the given units

        :return: A 2D numpy array where cell values correspond to slope values
        """
        # The slope array must be smaller than the DEM to avoid interpolating border data
        slope_array = np.empty((self.dem.shape[0] - 2, self.dem.shape[1] - 2), dtype=np.uint8)

        for row in range(slope_array.shape[0]):
            for col in range(slope_array.shape[1]):
                # Slope array is inset 1 pixel from the DEM
                z = self.get_neighbours(row + 1, col + 1)
                slope = None

                if self.algorithm == SlopeAlgorithms.NEIGHBORHOOD:
                    # See "The Effect Of Slope Algorithms on Slope Estimates" by Robert J. Hickey 1998
                    slope_ew = ((z[2] + 2 * z[3] + z[4]) - (z[0] + 2 * z[7] + z[6])) / (8 * self.cell_resolution)
                    slope_ns = ((z[0] + 2 * z[1] + z[2]) - (z[6] + 2 * z[5] + z[4])) / (8 * self.cell_resolution)
                    slope = math.sqrt(slope_ew ** 2 + slope_ns ** 2) * 100

                elif self.algorithm in (SlopeAlgorithms.MAXIMUM_SLOPE, SlopeAlgorithms.MAXIMUM_DOWNHILL_SLOPE):
                    if self.algorithm == SlopeAlgorithms.MAXIMUM_SLOPE:
                        # Find the neighbour with the greatest elevation difference from the center cell
                        deltas = [abs(int(z[8]) - int(zi)) for zi in z[:8]]
                    # Max downhill slope is identical except that it does not take absolute value of rise difference
                    else:
                        deltas = [int(z[8]) - int(zi) for zi in z[:8]]

                    max_delta_index = deltas.index(max(deltas))
                    max_delta = max(deltas)

                    # Diagonal cells
                    if max_delta_index in (0, 2, 4, 6):
                        distance = self.cell_resolution * 1.4142
                    else:
                        distance = self.cell_resolution
                    slope = (max_delta / distance) * 100

                elif self.algorithm == SlopeAlgorithms.QUADRATIC_SURFACE:
                    # See "The Effect Of Slope Algorithms on Slope Estimates" by Robert J. Hickey 1998
                    g = (-int(z[7]) + int(z[3])) / (2 * self.cell_resolution)
                    h = (int(z[1]) - int(z[5])) / (2 * self.cell_resolution)
                    slope = math.sqrt(g**2 + h**2) * 100

                # Convert percent slope to degrees if needed
                if self.units == SlopeUnits.DEGREES:
                    slope = math.degrees(math.atan(slope / 100))

                slope_array[row][col] = slope

        return slope_array


class Aspect(TerrainOutput):
    def __init__(self, dem, cell_resolution):
        super().__init__(dem, cell_resolution)

    def generate(self):
        """
        Generate an aspect map in degrees

        :return: A 2D numpy array where cell values correspond to aspect values in degrees
        """
        # The aspect array must be smaller than the DEM to avoid interpolating border data
        aspect_array = np.empty((self.dem.shape[0] - 2, self.dem.shape[1] - 2), dtype=np.uint16)

        for row in range(aspect_array.shape[0]):
            for col in range(aspect_array.shape[1]):
                # Aspect array is inset 1 pixel from the DEM
                z = self.get_neighbours(row + 1, col + 1)

                # See http://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
                slope_ew = ((z[2] + 2 * z[3] + z[4]) - (z[0] + 2 * z[7] + z[6])) / (8 * self.cell_resolution)
                slope_ns = ((z[0] + 2 * z[1] + z[2]) - (z[6] + 2 * z[5] + z[4])) / (8 * self.cell_resolution)
                aspect = math.degrees(math.atan2(slope_ns, slope_ew)) + 180

                if aspect > 90:
                    aspect = 450 - aspect
                else:
                    aspect = 90 - aspect

                aspect_array[row][col] = aspect

        return aspect_array


class Hillshade(TerrainOutput):
    def __init__(self, dem, cell_resolution, azimuth, altitude, slope, aspect):
        self.azimuth = azimuth
        self.altitude = altitude
        # Slope map 2D array
        self.slope = slope
        # Aspect map 2D array
        self.aspect = aspect
        super().__init__(dem, cell_resolution)

    def generate(self):
        # The hillshade array must be smaller than the DEM to avoid interpolating border data
        hillshade_array = np.empty((self.dem.shape[0] - 2, self.dem.shape[1] - 2), dtype=np.uint8)

        # Convert light source altitude in degrees to zenith in radians
        zenith_rad = (90 - self.altitude) * math.pi / 180

        # Convert azimuth to math degrees and then to radians
        azimuth_rad = 360 - self.azimuth + 90 * math.pi / 180
        if azimuth_rad > 2 * math.pi:
            azimuth_rad -= 2 * math.pi

        for row in range(hillshade_array.shape[0]):
            for col in range(hillshade_array.shape[1]):
                slope_rad = math.radians(self.slope[row][col])
                # Hillshade requires 0 to be East instead of North, so subtract 180
                aspect_rad = math.radians(self.aspect[row][col] - 180)

                # See https://pro.arcgis.com/en/pro-app/tool-reference/3d-analyst/how-hillshade-works.htm
                hillshade = 255 * (math.cos(zenith_rad) * math.cos(slope_rad) +
                                   (math.sin(zenith_rad) * math.sin(slope_rad) * math.cos(azimuth_rad - aspect_rad)))

                if hillshade < 0:
                    hillshade = 0

                hillshade_array[row][col] = hillshade

        return hillshade_array
