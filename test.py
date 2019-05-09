from nibabel.gifti import giftiio
import nibabel as nb
import numpy as np

gifti_image = nb.load("FLAT.coord.gii")
# index = nifti_image.get_fdata()

# print(index.shape)
img_data = [x.data for x in gifti_image.darrays]
#np_data = np.reshape(img_data, (len(img_data[0]), len(img_data[0][0])))

#print(np_data, np_data.shape)
