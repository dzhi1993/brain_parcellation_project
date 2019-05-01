from nibabel.gifti import giftiio
import nibabel as nb
import numpy as np

nifti_image = nb.load("SUIT_vertexvol.nii")
index = nifti_image.get_fdata()

print(index.shape)
# img_data = [x.data for x in gifti_image.darrays]
# np_data = np.reshape(img_data, (len(img_data[0]), len(img_data[0][0])))

# print(np_data, np_data.shape)
