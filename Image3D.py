
class Image3D:

    def __init__(self, array):
        self.buffer = array

    def get_nx(self):
        return self.buffer.shape[1]

    def get_ny(self):
        return self.buffer.shape[2]

    def get_nz(self):
        return self.buffer.shape[0]

    def get_size(self):
        return self.buffer.get_size

    def get_index(self, i, j, k):
        return j + i*self.get_ny() + k*self.get_nx()*self.get_ny()

    def get_indices(self, index):
        nxy = self.get_nx()*self.get_ny()
        k = index // nxy
        k_mod = index % nxy
        i = k_mod // self.get_ny()
        j = k_mod % self.get_ny()
        return [i, j, k]

    def neighbors(self, index):
        indices = self.get_indices(index)
        neighbors = []
        i = indices[0]-1
        if i >= 0:
            neighbors.append(self.get_index(i, indices[1], indices[2]))
        i = indices[0]+1
        if i < self.get_nx():
            neighbors.append(self.get_index(i, indices[1], indices[2]))
        j = indices[1]-1
        if j >= 0:
            neighbors.append(self.get_index(indices[0], j, indices[2]))
        j = indices[1]+1
        if j < self.get_ny():
            neighbors.append(self.get_index(indices[0], j, indices[2]))
        k = indices[2]-1
        if k >= 0:
            neighbors.append(self.get_index(indices[0], indices[1], k))
        k = indices[2]+1
        if k < self.get_nz():
            neighbors.append(self.get_index(indices[0], indices[1], k))
        return neighbors

    def set_value(self, i, j, k, value):
        self.buffer[k, i, j] = value

    def get_value(self, i, j, k):
        return self.buffer[k, i, j]

    def info(self):
        print("Image of size {}!".format(self.get_size()))

    def multiply(self, a):
        self.buffer *= a

    def add(self, image):
        self.buffer += image.buffer

    def get_whole_array(self):
        return self.buffer

    def subtract(self, image):
        self.buffer -= image.buffer

    def get_plane(self, k):
        return self.buffer[k, :, :]

