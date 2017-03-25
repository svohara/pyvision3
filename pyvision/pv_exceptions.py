"""
Custom exceptions and errors for use in pyvision
author: Stephen O'Hara
created: April 15, 2016
"""


class OutOfBoundsError(ValueError):
    pass


class InvalidImageFile(ValueError):
    pass


class ImageAnnotationError(ValueError):
    pass