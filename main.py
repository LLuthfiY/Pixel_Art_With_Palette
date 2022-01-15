from pixel_art_genertor import PixelArtGenerator as pag
from PIL import Image


palette = 'D:\project\Pixel_Art_With_Palette\\purplemorning8.txt'
image = 'D:\project\Pixel_Art_With_Palette\\red-house.jpg'


img = Image.open(image)
img = pag(img)
img.read_palette(palette)
img.pixelate(height=256)
img.clustering(type_color_for_compare=pag.RGB,
               clustering_method=pag.KMEANSRANDOM)
img.resize_to_original()
img.show()
img.save(image + '-pixel.jpg')
