import pixel_art_genertor
from PIL import Image


palette = 'ghost-town.txt'
image = 'garden2.jpg'

img = Image.open(image)
img = pixel_art_genertor.PixelArtGenerator(img)
# img.read_palette(palette)
img.pixelate(height=256)
img.clustering()
img.resize_to_original()
img.show()
img.save(image + '-pixel.jpg')
