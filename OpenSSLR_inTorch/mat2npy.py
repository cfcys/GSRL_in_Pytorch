import numpy as np
# import scipy.io as sio
# data=sio.loadmat('/home/amax/data/xibo/xibobo/DPN_HRA/HybridSN-sample/data/UP/PaviaU.mat')
# print(type(data)) #显示mat的数据类型
# print(list(data)) #只显示mat包含的属性
# print(data["paviaU"]) #显示属性值#
# #找到数值型的属性后，转为npy
# data2=data['paviaU'];
# np.save('/home/amax/data/xibo/OpenSSLR/PaviaU/paviau.npy',data2)

##########
from scipy import io
mat = np.load('/home/amax/data/xibo/OpenSSLR/data/Trento/trento_gt8.npy').astype('int')
io.savemat('/home/amax/data/xibo/OpenSSLR/data/Trento/trento_gt8.mat', {'trento_gt8': mat})

