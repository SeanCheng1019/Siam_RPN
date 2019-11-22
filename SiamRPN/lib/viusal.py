import visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class visual:
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(server='http://127.0.0.1', port=8097)
        assert self.vis.check_connection()
        self.counter = 0


    def plot_img(self, img, win=1, name='img'):
        self.vis.image(img.astype('uint8'), win=win, opts={'title': name})

    def plot_imgs(self, img, win=1, name='img'):
        self.vis.images(img, win=win, opts={'title':'multi-channel features'})

    def plot_heatmap(self,img, win=1, name='img'):
        self.vis.heatmap(img, win=win)
