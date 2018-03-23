from ImageIO import ImageIO
import matplotlib.pyplot as plt
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('file', default='data/output_47.raw')
parser.add_argument('--size', nargs=3, type=int, default=[175, 152, 152])
parser.add_argument('--index', type=int,  default=81)

args = parser.parse_args()
print(args.file)
io = ImageIO(args.file)
size = args.size
img = io.read(size[0], size[1], size[2])

if img is None:
    print("Unable to read this image ->wrong size")
    sys.exit(-1)

print('.plot section %d' % args.index)
plt.matshow(img.get_plane(args.index))
plt.show()

# The corresponding array can be retrieved with the method img.get_whole_array()
