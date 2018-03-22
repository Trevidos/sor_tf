from Image3D import Image3D
import numpy as np


class ImageIO:

    def __init__(self, file_name):
        self.file_name = file_name

    def read(self, nz, nx, ny):

        print('.read file %s with dim %dx%dx%d' % (self.file_name, nz, nx, ny))

        with open(self.file_name, mode='rb') as file:
            bytes_data = file.read()
            array_data = np.frombuffer(bytes_data,
                                       dtype='b',
                                       count=bytes_data.__len__())

            if bytes_data.__len__() == nx*nz*ny:
                return Image3D(np.reshape(array_data, (nz, nx, ny)))

            return None



