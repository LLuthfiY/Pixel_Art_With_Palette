import numpy as np
from numpy.core.fromnumeric import reshape
from skimage import color
from PIL import Image
from sklearn.cluster import KMeans
import itertools


class PixelArtGenerator:
    def __init__(self, image: Image) -> None:
        self.image = image
        self.palette = None
        self.resultImage = self.image

    def fromImagetoLab(self, img):
        npimage = np.array(img)
        try:
            npimage = color.rgba2rgb(npimage)
        except:
            pass
        return npimage

    def fromRGBtoImage(self, img):
        return Image.fromarray(img)

    def read_palette(self, palette_name) -> None:
        palette = open(palette_name, 'r')
        self.palette = palette.read().splitlines()
        self.palette = list(filter(lambda x: not (x[0] == ';'), self.palette))
        self.palette = [x[2:] for x in self.palette]

        rgb = []
        for c in self.palette:  # c mean color
            rgb.append(list(int(c[i:i+2], 16) for i in (0, 2, 4)))
        self.palette = rgb

    def getImage(self):
        return self.resultImage

    def clustering(self, cluster=7):
        try:
            cluster = len(self.palette)
        except:
            pass
        print('len cluster', cluster)
        img = self.fromImagetoLab(self.resultImage)
        h, w, _ = img.shape
        img = np.reshape(img, (-1, 3))
        labels, cluster_centers = self.k_mean(img, cluster)
        print(labels.shape, cluster_centers.shape)
        zeros = np.zeros((labels.shape[0], 3))

        try:
            cluster_centers = self.compareClusterCenterWithPallete(
                cluster_centers, self.palette)

        except:
            pass

        for cluster_center in range(len(cluster_centers)):
            zeros[labels == cluster_center] = cluster_centers[cluster_center]
        zeros = np.reshape(zeros, np.array(self.resultImage).shape)
        ims = self.fromRGBtoImage(zeros.astype(np.uint8))
        self.resultImage = ims

    def pixelate(self, height=-1, width=-1):
        h, w, _ = np.array(self.resultImage).shape
        if (height == -1):
            height = int(width * h / w)
        if (width == -1):
            width = int(height * w / h)

        imgSmall = self.resultImage.resize(
            (width, height), resample=Image.BILINEAR)
        #imgSmall = imgSmall.resize(self.image.size, Image.NEAREST)
        self.resultImage = imgSmall

    def show(self):
        self.resultImage.show()

    def k_mean(self, img, clusters):
        img_ = KMeans(clusters, random_state=0).fit(img)
        return (img_.labels_, img_.cluster_centers_)

    def euc(self, arr1, arr2):
        return np.sqrt(abs(arr1[0]-arr2[0])**2 + abs(arr1[1]-arr2[1])**2 + abs(arr1[2]-arr2[2])**2)

    def compareClusterCenterWithPallete(self, cluster_center, palette):
        zeros = np.zeros((len(cluster_center), len(palette)))
        for row in range(len(cluster_center)):
            for col in range(len(palette)):
                zeros[row, col] = self.euc(cluster_center[row], palette[col])
        zeros_reshaped = np.reshape(zeros, (-1, ))
        zeros_reshaped = sorted(zeros_reshaped)
        print('cluster', cluster_center)
        print('palette', palette)
        check = []
        pairs = []
        for el in zeros_reshaped:
            temp = zip(*np.where(zeros == el))
            for t in temp:
                if t[0] not in check:
                    if t[1] not in check:
                        check.append(t[0])
                        check.append(t[1])
                        pairs.append(t)
        print('pair', pairs)
        for pair in pairs:
            cluster_center[pair[0]] = palette[pair[1]]
        return cluster_center

    def resize_to_original(self):
        self.resultImage = self.resultImage.resize(
            self.image.size, Image.NEAREST)

    def save(self, text='result-pixelate.jpg'):
        self.resultImage.save(text)
