
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from model.R3Dicnet import  DIC
import matplotlib

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    # rate ：倍数
    def __init__(self, dims,rate):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // rate) + 1) * rate - self.ht) % rate
        pad_wd = (((self.wd // rate) + 1) * rate - self.wd) % rate

        self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]



@torch.no_grad()
def main():
    matplotlib.use('TkAgg')
    img1_path = "Star1.tif"
    img2_path = "Star2.tif"
    img_pairs = zip([img1_path], [img2_path])
    network_data = torch.load('../../result/R3DICnet_model_best.pth.tar')
    model = DIC(max_disp=4)
    model.training = True
    model.load_state_dict(network_data['state_dict'])
    model = model.to(device)
    model.eval()
    cudnn.benchmark = True
    for (img1_file, img2_file) in tqdm(img_pairs):
        img1 = cv.imread(img1_file, 0)
        img2 = cv.imread(img2_file, 0)
        img1 = img1 / 255.0
        img2 = img2 / 255.0
        if img1.ndim == 2:
            img1 = img1[np.newaxis, ...]
            img2 = img2[np.newaxis, ...]

            img1 = img1[np.newaxis, ...]
            img2 = img2[np.newaxis, ...]
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()
            padder = InputPadder(img1.shape, 16)
            img1, img2 = padder.pad(img1, img2)
        input1 = img1.to(device)
        input2 = img2.to(device)
        output = model(input1, input2, iters=8)
        output_to_write = output[-1].data.cpu()

        output_to_write = output_to_write.numpy()
        disp_x = output_to_write[0, 0, :, :]
        disp_y = output_to_write[0, 1, :, :]
        disp_y = padder.unpad(disp_y)
        disp_x = padder.unpad(disp_x)
        print(disp_y.shape)
        norm = matplotlib.colors.Normalize(vmin=-0.5, vmax=0.5)
        sc = plt.imshow(disp_y, cmap=plt.cm.jet,norm=norm)
        plt.colorbar()  # 显示色度条

        plt.show()
        np.savetxt('Star_y.csv', disp_y, delimiter=',')

        #
if __name__ == '__main__':
    main()

