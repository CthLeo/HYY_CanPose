import cv2
import matplotlib.pyplot as plt
import torch
from detector import *
from PIL import Image

## Settings
name = 'statue'
net_path = '/home/leoh/catkin_ws/src/dope/weights/statue_43.pth'

gpu_id = 0
img_path = '/home/leoh/Dope_dataset/rotate_camera_Statue/1/000045.png'
#img_path = '/home/leoh/Pictures/YCB_Video_0000_Color/000001-color.png'


# Function for visualizing feature maps
def viz_layer(layer, n_filters=9):
    fig = plt.figure(figsize=(20, 20))
    row = 1
    for i in range(n_filters):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i + 1))

    # unloader = transforms.ToPILImage()
    # myimage = layer.clone()
    # myimage = myimage.squeeze(0)
    # myimage = unloader(myimage)
    # plt.imshow(myimage)
    # plt.pause(0.001)

# load color image
in_img = cv2.imread(img_path)
# in_img = cv2.resize(in_img, (640, 480))
in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
# plot image
plt.imshow(in_img)


model = ModelData(name, net_path, gpu_id)
model.load_net_model()
net_model = model.net

# Run network inference
image_tensor = transform(in_img)
image_torch = Variable(image_tensor).cuda().unsqueeze(0)
out, seg = net_model(image_torch)
vertex2 = out[-1][0].cpu()
aff = seg[-1][0].cpu()

# View the vertex and affinities
viz_layer(vertex2)
viz_layer(aff, n_filters=16)

plt.show()