import cv2
import matplotlib.pyplot as plt
import numpy as np


base_img =  cv2.imread('./sjh.jpeg',cv2.IMREAD_GRAYSCALE)

base_img =  cv2.resize(base_img,(400,400))



eigen_vectors = np.linalg.eig(base_img)[1]
eigen_values = np.linalg.eig(base_img)[0]

identity =  np.identity(400)

eigen_value_for_digonal_matrix =  np.zeros((400,400))


for index, line in enumerate(identity):
    eigen_value_for_digonal_matrix[index,:] = line*eigen_values
    
    

res = np.matmul(eigen_vectors,eigen_value_for_digonal_matrix)
res = np.matmul(res,np.transpose(eigen_vectors))

plt.imshow(np.uint8(res))

plt.imshow(base_img)