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
