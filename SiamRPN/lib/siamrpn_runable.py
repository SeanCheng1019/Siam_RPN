from tqdm import tqdm
from lib.tracker import SiamRPNTracker

def run_SiamRPN(seq_path, model_path, init_box):
    x, y, w, h = init_box
    tracker = SiamRPNTracker(model_path)
    result = []
