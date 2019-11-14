import visdom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class visual:
    def __init__(self, port=8097):
        self.vis = visdom.Visdom(port=port)
        self.counter = 0

    def plot_img(self, img, win=1, name='img'):
        self.vis.image(img, win=win, opts={'title': name})

