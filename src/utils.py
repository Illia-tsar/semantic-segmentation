import numpy as np


def rle_encode(mask):
    """
    Run-Length Encoding
    
    Arguments:
        mask -- numpy array of shape (height, width, 1)
    Returns:
        string with encoded pixels
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join([str(x) for x in runs])


def rle_decode(rle):
    """
    Run-Length Decoding
    
    Arguments:
        rle -- string with encoded pixels
    Returns:
        numpy array of shape (height, width, 1)
    """
    shape = (768, 768)
    mask = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    rle_mask = rle.split()

    # elements with uneven indices are related to start idx
    # elements with even indices are related to lengths
    idx, length = [np.array(x, dtype=int) for x in (rle_mask[::2], rle_mask[1::2])]
    for a, b in zip(idx-1, idx+length-1):
        mask[a:b] = 1
    return mask.reshape(shape).T


def rle_object_size(rle):
    """
    Calculate object size in pixels
    
    Arguments:
        rle -- string with encoded pixels
    Returns:
        int -- quantity of pixels in object
    """
    rle_mask = rle.split()
    # take every second element of array and calculate the sum of obtained list
    object_size = np.sum([int(x) for x in rle_mask[1::2]])
    return object_size


def apply_mask(img, rle):
    """
    Apply mask to img
    
    Arguments:
        img -- numpy array of shape (height, width, 3)
        rle -- string with encoded pixels
    Returns:
        img -- numpy array with mask applied to it of shape (height, width, 3)
    """
    mask = rle_decode(rle)
    return img * mask[:, :, np.newaxis]
