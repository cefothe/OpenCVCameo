import cv2
import numpy
import utils

def strokeEdges(scr, dst, blurKsize = 7, edgeKsize=5):
    if blurKsize >= 3 :
        blurredScr = cv2.medianBlur(scr,blurKsize)
        graySrc = cv2.cvtColor(blurredScr, cv2.COLOR_RGB2GRAY)
    else:
        graySrc = cv2.cvtColor(scr,cv2.COLOR_GRAY2BGR)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(scr)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels,dst)

class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BFR)."""

    def __init__(self, kernel):
        self._kernel = kernel

    def apply(self, src, dst):
        """Apply the filter with a BGR or gray source/destination."""
        cv2.filter2D(src,-1,self._kernel,dst)

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius. We need to use this filter when we want to kept brightness"""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(kernel)

class FindEdgesFilter(VConvolutionFilter):
    """An edge-finding filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
        VConvolutionFilter.__init__(self,kernel)

class BlurFilter(VConvolutionFilter):
    """A blur filter with a 2-pixel radius. Example for blue filter is simple average filter"""

    def __init__(self):
        kernel = numpy.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04],
                              [0.04, 0.04, 0.04, 0.04, 0.04]])
        VConvolutionFilter.__init__(self,kernel)

class EmbossFilter(VConvolutionFilter):
    """An emboss filter with a 1-pixel radius."""

    def __init__(self):
        kernel = numpy.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [ 0, 1, 2]])
        VConvolutionFilter.__init__(self, kernel)


