import numpy as np
from .descriptors2d import MaskDecriptors, HistogramDescriptors, \
    Moments, MomentsCentral, MomentsHu, GlcmFeatures, Granulometry, \
    PowerSpectrum, Autocorrelation, LocalBinaryPattern, GaborEnergy
from .descriptors3d import MaskDescriptors3D, HistogramDescriptors3D, \
    Granulometry3D, Autocorrelation3D, LocalBinaryPattern3D, PowerSpectrum3D, \
    GlcmFeatures3D, RawMoments3D, HuMoments3D, CentralMoments3D, GaborFilters


def GetAll(d: int = 2) -> list:
    """
    Returns a list of all descriptors.
    """
    desc = []

    # 2D descriptors
    if d == 2:
        desc.append(MaskDecriptors())
        desc.append(HistogramDescriptors())
        desc.append(Moments())
        desc.append(MomentsCentral())
        desc.append(MomentsHu())
        desc.append(GlcmFeatures())
        desc.append(Granulometry())
        desc.append(PowerSpectrum())
        desc.append(Autocorrelation())
        desc.append(LocalBinaryPattern())
        desc.append(GaborEnergy())
        # desc.append(GaborSpectralHistogram())

    if d == 3:
        desc.append(MaskDescriptors3D())
        desc.append(HistogramDescriptors3D())
        desc.append(Granulometry3D())
        desc.append(Autocorrelation3D())
        desc.append(LocalBinaryPattern3D())
        desc.append(PowerSpectrum3D())
        desc.append(GlcmFeatures3D())
        desc.append(RawMoments3D())
        desc.append(HuMoments3D())
        desc.append(CentralMoments3D())
        desc.append(GaborFilters())
    return desc


def ComputeForAll(image: np.array, mask: np.array, d: int = 2) -> dict:
    """
    Computes all descriptors for the given image and mask.
    returns dictionary with descriptor name as key, value is tuple
    (type, result)
    """
    results = {}
    for desc in GetAll(d):
        results[desc.GetName()] = (desc.GetType(), desc.Eval(image, mask))
    return results
