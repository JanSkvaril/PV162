import numpy as np
from math import pi, sqrt
from scipy.spatial import ConvexHull
from porespy.metrics import regionprops_3D
from skimage.morphology import ball
from scipy.ndimage import binary_opening
from skimage.transform import resize

from .descriptor import DescriptorBase, DescriptorType


class MaskDescriptors3D(DescriptorBase):
    """
    Calculates descriptors of the given mask.
        - mask: 3D numpy array, binary mask

    Returns a dictionary with the following descriptors:
        - surface area
        - volume
        - bbox_volume
        - major axis length
        - minor axis length
        - compactness
        - sphericity
        - elongation
        - convexity
    """

    def Eval(self, image: np.array, mask: np.array):
        result = dict()
        width, height, depth = mask.shape

        props = regionprops_3D(mask)[0]

        result["surface_area"] = props.surface_area
        result["volume"] = np.count_nonzero(mask)

        result["bbox_volume"] = width * height * depth
        result["major_axis"] = props.axis_major_length
        result["minor_axis"] = props.axis_minor_length

        result["compactness"] =\
            (36 * pi * (result["volume"] ** 2)) / (result["surface_area"] ** 3)
        result["sphericity"] = result["compactness"] ** (1/3)

        foreground_points = np.argwhere(mask)
        convex_hull = ConvexHull(foreground_points)
        convex_hull_volume = convex_hull.volume
        result["convexity"] = result["volume"] / convex_hull_volume

        result["elongation"] = result["major_axis"] / result["minor_axis"]

        return result

    def GetName(self) -> str:
        return "Mask descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR


class Mean3D(DescriptorBase):
    """
        Calculates the mean of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask

        Returns scalar representing mean.
    """

    def Eval(self, image: np.array, mask: np.array):
        image_copy = np.copy(image)
        image_copy[mask == 0] = 0

        return np.sum(image_copy) / np.count_nonzero(mask)

    def GetName(self) -> str:
        return "Mean"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class StdDev3D(DescriptorBase):
    """
        Calculates the standard deviation of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask

        Returns scalar representing standard deviation.
    """

    def Eval(self, image: np.array, mask: np.array):
        non_zero_points = np.copy(image)[mask != 0]

        count_points = np.count_nonzero(mask)
        mean = np.sum(non_zero_points) / count_points

        src = non_zero_points.flatten()

        return sqrt(np.sum((src - mean) ** 2) / count_points)

    def GetName(self) -> str:
        return "StdDev"

    def GetType(self) -> DescriptorType:
        return DescriptorType.SCALAR


class Histogram3D(DescriptorBase):
    """
        Computes the histogram of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - bins: number of bins for the histogram, if none, max value of image
          is used

        Returns a 1D numpy array with the histogram.
    """

    def __init__(self, bins: int | None = None):
        self.bins = bins

    def Eval(self, image: np.array, mask: np.array):
        return self.Histogram3D(image, mask, bins=self.bins)

    def Histogram3D(self,
                    image: np.array,
                    mask: np.array,
                    bins: int | None = None) -> np.array:
        if bins is None:
            bins = np.max(image)

        return np.histogram(image[mask != 0], bins=bins)

    def GetName(self) -> str:
        return "Histogram"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR


class HistogramDescriptors3D(DescriptorBase):
    """
        Computes the descriptors of the histogram.
        - histogram: 1D numpy array

        Returns a dictionary with the following descriptors:
        - mean
        - standard deviation
        - variance
        - median
        - max
        - min
        - geometric_mean
        - skewness
        - kurtosis
        - entropy
        - energy
    """

    def __init__(self, bins: int | None = None):
        self.bins = bins

    def Eval(self, image: np.array, mask: np.array):
        histogram, bin_edges = Histogram3D(self.bins).Eval(image, mask)
        return self.HistogramDescriptors3D(histogram, bin_edges)

    def GetName(self) -> str:
        return "Histogram descriptors"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR

    def getKthMoment(self, k: int, values: np.array, histogram: np.array,
                     mean: float, count: int) -> float:

        return np.sum(((values - mean) ** k) * histogram) / count

    def HistogramDescriptors3D(self,
                               histogram: np.array,
                               bin_edges: np.array) -> dict:
        result = dict()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_count = np.sum(histogram)

        result["mean"] = np.sum(bin_centers * histogram) / total_count

        moment2 = self.getKthMoment(2, bin_centers, histogram,
                                    result["mean"], total_count)

        moment3 = self.getKthMoment(3, bin_centers, histogram,
                                    result["mean"], total_count)

        moment4 = self.getKthMoment(4, bin_centers, histogram,
                                    result["mean"], total_count)

        result["var"] = moment2
        result["std"] = sqrt(result["var"])
        result["max"] = bin_edges[len(bin_edges) - 1]
        result["min"] = bin_edges[0]

        cumulative_sum = np.cumsum(histogram)
        median_index = np.argmax(cumulative_sum >= np.sum(histogram) / 2)

        bin_width = bin_edges[1] - bin_edges[0]
        median_bin_left = bin_edges[median_index]
        median_bin_count = histogram[median_index]

        total_counts_before_median = 0
        if median_index > 0:
            total_counts_before_median = cumulative_sum[median_index - 1]

        result["median"] = median_bin_left + ((total_count / 2
                                               - total_counts_before_median)
                                              / median_bin_count) * bin_width
        
        weighted_log = np.log(bin_centers) * histogram
        mean_weighted_log = np.sum(weighted_log) / total_count
        result["gmean"] = np.exp(mean_weighted_log)
        
        result["skewness"] = moment3 / (result["std"] ** 3)
        result["kurtosis"] = (moment4 / (result["var"] ** 2)) - 3

        prob_dist = histogram / total_count
        prob_dist = prob_dist[prob_dist != 0]
        result["entropy"] = -np.sum(prob_dist * np.log2(prob_dist))

        result["energy"] = np.sum(histogram ** 2)

        return result


class Granulometry3D(DescriptorBase):
    """
        Creates granulometric curve from values in the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - max_radius: maximum radius of the structuring element,
        - step: step between sizes of structuring element

        Returns granulometric curve, that shows size distributions of objects
        in the image.
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Granulometry3D(image, mask)

    def GetName(self) -> str:
        return "Granulometry"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR

    def Granulometry3D(self,
                       image: np.array,
                       mask: np.array,
                       max_radius=10,
                       step=1) -> np.array:

        curve = []
        init_volume = np.sum(mask)

        for radius in range(0, max_radius, step):
            st_element = ball(radius)
            opening = binary_opening(mask, structure=st_element)

            opening_volume = np.sum(opening)
            curve.append(opening_volume / init_volume)

        return np.array(curve)


class PowerSpectrum3D(DescriptorBase):
    """
        Calculates the power spectrum of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask.

        returns 3D image containg powerscpetrum
    """

    def Eval(self, image: np.array, mask: np.array) -> np.array:
        img = np.copy(image)
        img[mask == 0] = 0

        fft_image = np.fft.fftn(img)
        power_image = np.abs(fft_image) ** 2

        return np.fft.fftshift(power_image)

    def GetName(self) -> str:
        return "Power spectrum"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX


class Autocorrelation3D(DescriptorBase):
    """
        Calculates the autocorrelation of the image within the mask.

        - image: 3D numpy array
        - mask: 3D numpy array, binary mask.

        returns 3D image containg autocorrelation
    """

    def Eval(self, image: np.array, mask: np.array):
        return self.Autocorrelation3D(image, mask)

    def GetName(self) -> str:
        return "Autocorrelation"

    def GetType(self) -> DescriptorType:
        return DescriptorType.MATRIX

    def Autocorrelation3D(self,
                          image: np.array,
                          mask: np.array,
                          size: int | None = None) -> np.array:
        src = np.copy(image)
        src[mask == 0] = 0

        var = np.var(src)
        data = src - np.mean(src)

        power_img = np.abs(np.fft.fftn(data)) ** 2
        autocorr_img = np.fft.ifftn(power_img).real / var / np.prod(src.shape)

        if size is not None:
            autocorr_img = resize(autocorr_img, (size, size, size))

        return np.fft.fftshift(autocorr_img)


class LocalBinaryPattern3D(DescriptorBase):
    """
        Computes histogram of local binary patterns from image within mask.
    """

    def __init__(self, radius: int = 1):
        self.radius = radius

    def Eval(self, image: np.array, mask: np.array):
        src = np.copy(image)
        src[mask == 0] = 0        
        lbp = self.ComputeLBP3D(src, self.radius)

        hist, bin_edges = Histogram3D(64).Eval(lbp, mask)
        return hist / np.sum(hist), bin_edges

    def GetName(self) -> str:
        return "Local binary pattern"

    def GetType(self) -> DescriptorType:
        return DescriptorType.VECTOR

    def ComputeLBP3D(self, image: np.array, radius: int):
        depth, rows, cols = image.shape
        lbp_image = np.zeros((depth, rows, cols))

        offsets = [(dz, dy, dx) for dz in range(-radius, radius + 1)
                   for dy in range(-radius, radius + 1)
                   for dx in range(-radius, radius + 1)
                   if (dz, dy, dx) != (0, 0, 0)]

        for z in range(radius, depth - radius):
            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[z, i, j]
                    binary_pattern = []

                    for dz, dy, dx in offsets:
                        neighbor = image[z + dz, i + dy, j + dx]
                        binary_pattern.append(1 if neighbor >= center else 0)

                    lbp_value = 0
                    for k in range(len(binary_pattern)):
                        lbp_value += binary_pattern[k] * (2 ** k)

                        lbp_image[z, i, j] = lbp_value

        return lbp_image


class RawMoments3D(DescriptorBase):
    """
        Calculates the moments up to given order or default 4th order
        of the image within the mask.
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - order: (optional) int

        Returns a dictionary with the following descriptors:
        - moments for the corresponding key tuple (p, q, r) representing
          the orders along the x, y, and z axes.
    """

    def __init__(self, order: int = 4):
        self.order = order

    def Eval(self, image: np.array, mask: np.array):
        copy = np.copy(image)
        copy[mask == 0] = 0

        indices = np.indices(mask.shape)
        moments = dict()

        for p in range(self.order + 1):
            for q in range(self.order + 1):
                for r in range(self.order + 1):
                    if p + q + r <= self.order:
                        moment = np.sum((indices[0] ** p) * (indices[1] ** q) *
                                        (indices[2] ** r) * copy)
                        moments[f'{p}{q}{r}'] = moment

        return moments

    def GetName(self) -> str:
        return "RawMoments"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR


class CentralMoments(DescriptorBase):
    """
        Calculates the centroids and central moments from the 2nd up to
        the given order or defaults to the 4th order of the image within
        the mask. Moments are defaulted not to be normalized unless specified
        otherwise.

        Parameters:
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - order: (optional) int
        - normalize: (optional) bool

        Returns a dictionary with the following descriptors:
        - moments for the corresponding key tuple (p, q, r) representing
          the orders along the x, y, and z axes.
        - centroid_x, centroid_y, centroid_z: Centroids of the image volume
          along the x, y, and z axes.
        - skewness_x, skewness_y, skewness_z: Skewness along the axes.
        - kurtosis_x, kurtosis_y, kurtosis_z: Kurtosis along the axes.
    """

    def __init__(self, order: int = 4, normalize: bool = False):
        self.order = order
        self.normalize = normalize

    def Eval(self, image: np.array, mask: np.array):
        copy = np.copy(image)
        copy[mask == 0] = 0

        indices = np.indices(mask.shape)
        central_moments = dict()

        zero_moment = np.sum(copy)
        centroid_x = np.sum(indices[0] * copy) / zero_moment
        centroid_y = np.sum(indices[1] * copy) / zero_moment
        centroid_z = np.sum(indices[2] * copy) / zero_moment

        for p in range(self.order + 1):
            for q in range(self.order + 1):
                for r in range(self.order + 1):
                    if 2 <= p + q + r <= self.order:
                        moment = np.sum(((indices[0] - centroid_x) ** p) *
                                        ((indices[1] - centroid_y) ** q) *
                                        ((indices[2] - centroid_z) ** r) *
                                        copy)

                        if self.normalize and p + q + r >= 2:
                            moment = moment / (zero_moment **
                                               ((1 + (p + q + r)) / 2.0))

                        central_moments[f'{p}{q}{r}'] = moment

        central_moments['centroid_x'] = centroid_x
        central_moments['centroid_y'] = centroid_y
        central_moments['centroid_z'] = centroid_z

        sigma_x = np.sqrt(central_moments['200'])
        sigma_y = np.sqrt(central_moments['020'])
        sigma_z = np.sqrt(central_moments['002'])

        central_moments['m200'] = central_moments['200']
        central_moments['m020'] = central_moments['020']
        central_moments['m002'] = central_moments['002']

        central_moments['skewness_x'] = central_moments['300'] / sigma_x**3
        central_moments['skewness_y'] = central_moments['030'] / sigma_y**3
        central_moments['skewness_z'] = central_moments['003'] / sigma_z**3

        central_moments['kurtosis_x'] = (central_moments['400'] / sigma_x**4)
        central_moments['kurtosis_y'] = (central_moments['040'] / sigma_y**4)
        central_moments['kurtosis_z'] = (central_moments['004'] / sigma_z**4)

        return central_moments

    def GetName(self) -> str:
        return "Central Moments"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR


class HuMoments3D(DescriptorBase):
    """
        Calculates the first seven Hu moments of the image within the mask.
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask

        Returns a dictionary containing the first seven Hu Moments,
        stored as keys 'hu1' to 'hu7'.
    """

    def Eval(self, image: np.array, mask: np.array):
        copy = np.copy(image)
        copy[mask == 0] = 0
        result = dict()

        central_normalized_moments = CentralMoments3D(3, True)
        cnm = central_normalized_moments.Eval(image, mask)

        result['hu1'] = cnm['200'] + cnm['020'] + cnm['002']
        result['hu2'] = (cnm['200'] - cnm['020']) ** 2 + \
                        (2 * cnm['110']) ** 2 + \
                        (cnm['002'] - cnm['020']) ** 2 + \
                        (2 * cnm['101']) ** 2 + \
                        (cnm['111']) ** 2

        result['hu3'] = (cnm['300'] - 3 * cnm['120']) ** 2 + \
                        (3 * cnm['210'] - cnm['030']) ** 2 + \
                        (3 * (cnm['201'] - cnm['021'])) ** 2 + \
                        (cnm['003'] - 3 * cnm['012']) ** 2

        result['hu4'] = (cnm['300'] + cnm['120']) ** 2 + \
                        (cnm['210'] + cnm['030']) ** 2 + \
                        (cnm['201'] + cnm['021']) ** 2 + \
                        (cnm['003'] + cnm['012']) ** 2

        result['hu5'] = (cnm['300'] - 3 * cnm['120']) * \
                        (cnm['300'] + cnm['120']) * \
                        ((cnm['210'] + cnm['201']) ** 2 - 3 *
                         (cnm['030'] + cnm['012']) ** 2) + \
                        (3 * cnm['210'] - cnm['030']) * \
                        (cnm['201'] + cnm['021']) * \
                        (3 * (cnm['120'] + cnm['003']) ** 2 -
                         (cnm['021'] + cnm['012']) ** 2)

        result['hu6'] = (cnm['200'] - cnm['020']) * \
                        ((cnm['300'] + cnm['030']) ** 2 -
                         (cnm['210'] + cnm['201']) ** 2 +
                         4 * cnm['111'] * (cnm['300'] + cnm['030']) *
                         (cnm['210'] + cnm['201'])) + 4 * cnm['110'] * \
                        (cnm['300'] + cnm['030']) * \
                        (cnm['210'] + cnm['201']) + 4 * cnm['101'] * \
                        (cnm['030'] + cnm['003']) * \
                        (cnm['120'] + cnm['012']) + \
                        (cnm['003'] - 3 * cnm['120']) * \
                        (cnm['210'] + cnm['120']) * \
                        (3 * (cnm['030'] + cnm['012']) ** 2 -
                         (cnm['021'] + cnm['003']) ** 2)

        result['hu7'] = (3 * cnm['210'] - cnm['201']) * \
                        (cnm['300'] + cnm['030']) * \
                        ((cnm['300'] + cnm['030']) ** 2 -
                            3 * (cnm['210'] + cnm['201']) ** 2) - \
                        (cnm['120'] - 3 * cnm['012']) * \
                        (cnm['210'] + cnm['201']) * \
                        (3 * (cnm['120'] + cnm['003']) ** 2 -
                            (cnm['021'] + cnm['012']) ** 2) + \
                        (3 * cnm['021'] - cnm['003']) * \
                        (cnm['030'] + cnm['003']) * \
                        ((cnm['030'] + cnm['003']) ** 2 -
                            3 * (cnm['021'] + cnm['012']) ** 2)

        return result

    def GetName(self) -> str:
        return "HuMoments"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR


class GlcmFeatures3D(DescriptorBase):
    """
        Computes the descriptors of the co-ocurrence matrix for a given delta
        (defaults to (x, y, z) = (1, 1, 1)) and distance to check
        for neighbouring channel (defaults to 1)
        - image: 3D numpy array
        - mask: 3D numpy array, binary mask
        - delta: (optional) tuple[int, int, int]
        - distance: (optional) int

        Returns a dictionary with the following descriptors:
        - energy
        - entropy
        - contrast
        - homogeneity
        - correlation
        - shade
        - prominence
    """

    def __init__(self, delta: tuple[int] = (1, 1, 1), distance: int = 1):
        self.delta = delta
        self.distance = distance

    def Eval(self, image: np.array, mask: np.array):
        glcm = self.Glcm3D(image, mask)
        return self.GlcmFeatures3D(glcm)

    def Glcm3D(self, image: np.array, mask: np.array) -> np.array:
        """
            Creates gray level co-ocurrence matrix from values in mask.
            - image: 2D numpy array, with values ranging from 0 to 255
            - mask: 2D numpy array, binary mask
            Returns co-ocurrence matrix
        """
        copy = np.copy(image).astype(np.uint8)
        copy[mask == 0] = 0

        offset = (self.delta[0] * self.distance,
                  self.delta[1] * self.distance,
                  self.delta[2] * self.distance)

        x_max, y_max, z_max = image.shape

        levels = copy.max() + 1

        matrix = np.zeros((levels, levels))

        for (x, y, z), v in np.ndenumerate(copy):
            x_offset = x + offset[0]
            y_offset = y + offset[1]
            z_offset = z + offset[2]

            if (x_offset >= x_max) or (y_offset >= y_max) or (z_offset >=
                                                              z_max):
                continue

            value_at_offset = copy[x_offset, y_offset, z_offset]

            matrix[v, value_at_offset] += 1

        return matrix

    def sgn(self, x: int) -> int:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def GlcmFeatures3D(self, matrix: np.array, mean: bool = True) -> dict:
        """
            Computes the descriptors of the normalized co-ocurrence matrix.
            - matrix: glcm matrix
            - mean: if true, returns the mean of the descriptors
            Returns a dictionary with the following descriptors:
            - energy
            - entropy
            - contrast
            - homogeneity
            - correlation
            - shade
            - prominence
        """
        result = dict()

        matrix = matrix + np.transpose(matrix)
        matrix = matrix / np.sum(matrix)
        matrix_nonzero = matrix + 1e-10

        result['energy'] = np.sum(matrix ** 2)
        result['entropy'] = - np.sum(np.log(matrix_nonzero) * matrix_nonzero)
        result['contrast'] = np.sum([p * (i - j) ** 2 for (i, j), p
                                     in np.ndenumerate(matrix)])
        result['homogenity'] = np.sum([p / (1 + (i - j) ** 2) for (i, j), p
                                       in np.ndenumerate(matrix)])

        glcm_mean_i = np.sum([i * p for (i, _), p in np.ndenumerate(matrix)])
        glcm_mean_j = np.sum([j * p for (_, j), p in np.ndenumerate(matrix)])
        glcm_mean = (glcm_mean_i + glcm_mean_j) / 2

        variance = np.sum([p * (i - glcm_mean_i) ** 2
                           for (i, j), p in np.ndenumerate(matrix)])
        correlation = np.sum([(i - glcm_mean_i) * (j - glcm_mean_j) * p /
                              variance
                              for (i, j), p in np.ndenumerate(matrix)])

        result['correlation'] = correlation

        sigma3 = np.sum([((i - glcm_mean) ** 3 * p) / (variance ** (3/2))
                         for (i, _), p in np.ndenumerate(matrix)])
        A = np.sum([((i + j - 2 * glcm_mean) ** 3 * p) /
                    (sigma3 * (sqrt(2 * (1 + correlation))) ** 3)
                    for (i, j), p in np.ndenumerate(matrix)])
        result['shade'] = self.sgn(A) * abs(A) ** (1/3)

        B = np.sum([((i + j - 2 * glcm_mean) ** 4 * p) /
                    (4 * variance ** 2 * (1 + correlation) ** 2)
                    for (i, j), p in np.ndenumerate(matrix)])
        result['prominence'] = self.sgn(B) * abs(B) ** (1/4)

        return result

    def GetName(self) -> str:
        return "Glcm features"

    def GetType(self) -> DescriptorType:
        return DescriptorType.DICT_SCALAR
