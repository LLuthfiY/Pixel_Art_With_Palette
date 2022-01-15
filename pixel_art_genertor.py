import numpy as np
from skimage import color
from PIL import Image
from sklearn.cluster import KMeans, mean_shift
import itertools


class PixelArtGenerator:
    LAB = 2
    RGB = 1
    HSV = 3
    LAB_WITH_CUSTOM_DISTANCES_METHOD = 4
    KMEANS = 1
    KMEANSRANDOM = 2

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

    def clustering(self, cluster=7, type_color_for_compare=1, clustering_method=1):
        func = self.k_mean
        if clustering_method == 2:
            func = self.k_mean_random
        try:
            cluster = len(self.palette)
        except:
            pass
        print('len cluster', cluster)
        img = self.fromImagetoLab(self.resultImage)
        img = np.reshape(img, (-1, 3))
        labels, cluster_centers = func(img, cluster)
        print(labels.shape, cluster_centers.shape)
        zeros = np.zeros((labels.shape[0], 3))

        try:
            cluster_centers = self.compareClusterCenterWithPallete(
                cluster_centers, self.palette, type_color_for_compare)
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

    def k_mean_random(self, img, clusters):
        img_ = KMeans(clusters, random_state=0, init='random').fit(img)
        return (img_.labels_, img_.cluster_centers_)

    def compareClusterCenterWithPallete(self, cluster_center, palette, type_color_for_compare):
        cluster_center_compare = cluster_center
        palette_compare = palette
        func = self.euc

        # change color type based on parameter
        if (type_color_for_compare == 2):
            cluster_center_compare = color.rgb2lab(cluster_center)
            palette_compare = color.rgb2lab(palette)
        if (type_color_for_compare == 4):
            cluster_center_compare = color.rgb2lab(cluster_center)
            palette_compare = color.rgb2lab(palette)
            func = self.custom_euc_for_lab
        if (type_color_for_compare == 3):
            cluster_center_compare = color.rgb2hsv(cluster_center)
            palette_compare = color.rgb2hsv(np.array(palette))

        distances = [func(x, y)
                     for x in cluster_center_compare for y in palette_compare]
        distances_sorted = sorted(distances)

        check1 = []
        check2 = []
        pairs = []

        for index in range(len(distances_sorted)):
            for i in [x for x in range(len(distances)) if distances[x] == distances_sorted[index]]:
                div = i//len(cluster_center)
                mod = i % len(palette)
                if div not in check1 and mod not in check2:
                    check1.append(div)
                    check2.append(mod)
                    pairs.append([div, mod])

        print('pair', pairs)
        for pair in pairs:
            cluster_center[pair[0]] = palette[pair[1]]
        return cluster_center

    def resize_to_original(self):
        self.resultImage = self.resultImage.resize(
            self.image.size, Image.NEAREST)

    def save(self, text='result-pixelate.jpg'):
        self.resultImage.save(text)

    def euc(self, arr1, arr2):
        return np.sqrt(abs(arr1[0]-arr2[0])**2 + abs(arr1[1]-arr2[1])**2 + abs(arr1[2]-arr2[2])**2)

    def custom_euc_for_lab(self, arr1, arr2):
        return np.sqrt(abs(arr1[0]-arr2[0])**2 + abs(arr1[1]-arr2[1])**2) + abs(arr1[2]-arr2[2])**2
