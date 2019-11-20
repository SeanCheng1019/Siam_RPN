from net.config import Config
from net.net_siamrpn import SiameseAlexNet
from lib.tracker import SiamRPNTracker

def main(model_path, datasets_path, vis=True):
    tracker = SiamRPNTracker(model_path)
    # loading testing dataset
    dataset =

if __name__ == '__main__':
    testing_datasets = ''
    main()