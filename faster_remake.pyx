# In your Cython file (.pyx)
from cython.parallel import parallel, prange
import numpy as np
from PIL import Image
cimport numpy as np

def set_value(np.ndarray[np.uint8_t, ndim=2] remake):
    cdef int row = remake.shape[0]
    cdef int col = remake.shape[1]
    cdef int i, j

    # Parallelize only the access, and modify the array after acquiring GIL
    for i in prange(0, row, schedule='dynamic', nogil=True):  # Parallelize outer loop without GIL
        # Ensure modification happens after acquiring the GIL
        for j in range(0, col):
            if remake[i, j] != 0 and remake[i, j] != 255:
                with gil:
                    remake[i, j] = 127

    # Return the NumPy array which can be directly converted to a PIL image
    return remake


def pil_to_binary_mask(pil_image, threshold=0):
    # Convert the PIL image to a numpy array
    np_image = np.array(pil_image)

    # Convert the image to grayscale
    grayscale_image = Image.fromarray(np_image).convert("L")
    image_array = np.array(grayscale_image)

    # Apply thresholding to create a binary mask
    binary_mask = np.where((image_array > threshold) & (image_array < 250), True, False)

    # Create a binary mask and fill it with 1s where the condition is True
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1

    # Scale the mask to 255 and convert back to uint8
    mask = (mask * 255).astype(np.uint8)

    # Convert the mask back to a PIL image
    output_mask = Image.fromarray(mask)

    return output_mask
