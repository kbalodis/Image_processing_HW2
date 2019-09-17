import cv2
import numpy
from matplotlib import pyplot as plt

def kernela_1D_reizinajums(mat):
    lielums = len(mat)
    matrix = [[0 for x in range(lielums)] for y in range(lielums)]
    i = 0
    for rinda in matrix:
        j = 0
        for _ in rinda:
            matrix[i][j] = mat[i] * mat[j]
            j += 1
        i += 1
    return matrix

def normalizet(kernelis):
    maksimums = 0
    for rinda in kernelis:
        for elem in rinda:
            if maksimums <= elem:
                maksimums = elem
    i = 0
    for rina in kernelis:
        j = 0
        for _ in rinda:
            kernelis[i][j] *= 1.0 / maksimums
            j += 1
        i += 1

    return kernelis

def gausa_vertiba(x, sd):
    return 1 / (numpy.sqrt(2 * numpy.pi) * sd) * numpy.power(numpy.e, ( - numpy.power((x) / sd, 2) / 2))

def gausa_kernelis_2D(lielums=5, standartnovirze=1.):
    kernelis_1D = [0] * lielums
    indekss = 0
    for _ in kernelis_1D:
        kernelis_1D[indekss] = float(indekss) - lielums / 2
        indekss += 1

    indekss = 0
    for elem in kernelis_1D:
        vertiba = gausa_vertiba(elem, standartnovirze)
        kernelis_1D[indekss] = vertiba
        indekss += 1

    kernelis_2D = kernela_1D_reizinajums(kernelis_1D)

    kernelis_2D_normalizets = normalizet(kernelis_2D)
    
    return kernelis_2D_normalizets


def konvolucija(attels, kernelis):
    attels_jauns = numpy.zeros(shape=attels.shape)
    for i in numpy.arange(( len(kernelis) - 1 ) / 2, attels.shape[0] - (len(kernelis) - 1 ) / 2 ):
        for j in numpy.arange(( len(kernelis) - 1 ) / 2, attels.shape[1] - (len(kernelis) - 1 ) / 2 ):
            intensitate = 0.
            for k in numpy.arange( - ( len(kernelis) - 1 ) / 2, ( len(kernelis) - 1 ) / 2 ):
                for l in numpy.arange( - ( len(kernelis) - 1 ) / 2, (len(kernelis) - 1 ) / 2 ):
                    intensitate_orig = attels[i + k][j + l]
                    varbutiba = kernelis[( len(kernelis) - 1 ) / 2 + k][ ( len(kernelis) - 1 ) / 2 + l]
                    intensitate = intensitate + intensitate_orig * varbutiba
            
            attels_jauns[i][j] = intensitate
    return attels_jauns

kernelis = gausa_kernelis_2D()

attels = cv2.imread('images.jpeg', 0)
attels_masivs = numpy.asarray(attels)

attels_jauns = konvolucija(attels_masivs, kernelis)

fig = plt.figure()

fig.add_subplot(121)
plt.title('Originalais attels')
plt.set_cmap('gray')
plt.imshow(attels_masivs)

fig.add_subplot(122)
plt.title('Izpludinatais attels')
plt.set_cmap('gray')
plt.imshow(attels_jauns)

plt.show(block=True)