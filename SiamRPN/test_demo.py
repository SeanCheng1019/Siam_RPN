from net.net_siamrpn import SiameseAlexNet
import numpy as np
import torch as t
net = SiameseAlexNet()
# input_template = t.from_numpy(np.random.random(size=(1, 3, 127, 127)) * 255).float()
# input_detection = t.from_numpy(np.random.random(size=(1, 3, 271, 271)) * 255).float()
#
# cls_score, reg_score = net(input_template, input_detection)
# print(cls_score, reg_score)
for k,v in net.state_dict().items():
    print(k)