import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from os.path import dirname, realpath, join

dir=join(dirname(realpath(dirname(__file__))), 'Images')
filename=dir+"/MarshOrchid.jpg"

image=mpimg.imread(filename)

print(image.shape)
plt.imshow(image)
plt.show()


