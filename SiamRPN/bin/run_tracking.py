from got10k.experiments import *
from lib.tracker import SiamRPNTracker

if __name__ == '__main__':
    # setup tracker
    model_path = ''
    tracker = SiamRPNTracker(model_path=model_path)
    # setup experiments
    experiments = [
        ExperimentVOT('data/vot2016', version=2016),
        ExperimentOTB('data/OTB', version=2015)
    ]
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
