from got10k.experiments import *
from lib.tracker import SiamRPNTracker

if __name__ == '__main__':
    # setup tracker
    model_path = '../data/models/siamrpn_epoch_48.pth'
    # model_path = '../model/siamrpn_38.pth'
    # model_path = '../data/models/siamrpn_stmm_epoch_48.pth'
    tracker = SiamRPNTracker(model_path=model_path)
    #tracker = SiamRPNTracker_other(model_path=model_path)
    # setup experiments
    experiments = [
        #ExperimentVOT('/home/csy/dataset/dataset/benchmark/data/vot2016', version=2016),
        ExperimentOTB('/home/csy/dataset/dataset/OTB100/Benchmark', version=2015)
    ]
    for e in experiments:
        e.run(tracker, visualize=False)
        e.report([tracker.name])
