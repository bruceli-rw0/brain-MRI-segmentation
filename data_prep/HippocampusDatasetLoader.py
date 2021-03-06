"""
Module loads the hippocampus dataset into RAM
"""
from pathlib import Path

import numpy as np
from medpy.io import load
from tqdm.auto import tqdm

from utils.utils import med_reshape

def LoadHippocampusData(opt, y_shape, z_shape):
    '''
    This function loads our dataset form disk into memory,
    reshaping output to common size

    Arguments:
        volume {Numpy array} -- 3D array representing the volume

    Returns:
        Array of dictionaries with data stored in seg and image fields as 
        Numpy arrays of shape [AXIAL_WIDTH, Y_SHAPE, Z_SHAPE]
    '''

    image_dir = Path(opt.data_dir)/'images'
    label_dir = Path(opt.data_dir)/'labels'

    files = [f.name for f in sorted(list(Path(opt.data_dir).glob('images/*.nii.gz')))]
    out = []
    for f in tqdm(files):

        # We would benefit from mmap load method here if dataset doesn't fit into memory
        # Images are loaded here using MedPy's load method. We will ignore header 
        # since we will not use it
        image, _ = load(Path(image_dir/f).as_posix())
        label, _ = load(Path(label_dir/f).as_posix())

        # normalize all images (but not labels) so that values are in [0..1] range
        image = image / image.max()

        # We need to reshape data since CNN tensors that represent minibatches
        # in our case will be stacks of slices and stacks need to be of the same size.
        # In the inference pathway we will need to crop the output to that
        # of the input image.
        # Note that since we feed individual slices to the CNN, we only need to 
        # extend 2 dimensions out of 3. We choose to extend coronal and sagittal here
        image = med_reshape(image, new_shape=(image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape=(label.shape[0], y_shape, z_shape)).astype(int)

        # TASK: Why do we need to cast label to int?
        # ANSWER: int datatype is more memory efficient than float
        out.append({"image": image, "seg": label, "filename": f})

    # Hippocampus dataset only takes about 300 Mb RAM, so we can afford to keep it all in RAM
    print(f"Processed {len(out)} files, total {sum([x['image'].shape[0] for x in out])} slices")

    return np.array(out)
