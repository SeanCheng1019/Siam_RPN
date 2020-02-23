from got10k.experiments import *
from lib.tracker import SiamRPNTracker


if __name__ == '__main__':
    # setup tracker
    # model_path = '../data/models/siamrpn_epoch_43.pth'
    # model_path = '/media/csy/62ac73e0-814c-4dba-b59d-676690aca14b/PycharmProjects/siamrpn-pytorch-master/model/model.pth'
    # model_path = '../data/models/siamrpn_stmm_epoch_47.pth'
    for i in range(40, 41):
        #model_path = '/home/csy/models/siamrpn_{}.pth'.format(i)
        model_path = '../data/models/siamrpn_stmm_select_template_epoch_{}.pth'.format(i)
        tracker = SiamRPNTracker(model_path=model_path)
        # setup experiments
        experiments = [
            # ExperimentVOT('/home/csy/dataset/dataset/benchmark/data/vot2016', version=2016, result_dir='results_stmm_5interval/epoch{}'.format(i),
            #               report_dir='reports_stmm_5interval/epoch{}'.format(i)),
            ExperimentOTB('/home/csy/dataset/dataset/OTB100/Benchmark', version=2015,
                           result_dir='results/noupdate_adaptive/epoch{}'.format(i),
                           report_dir='reports/noupdate_adaptive/epoch{}'.format(i))
            #ExperimentVOT('/home/csy/dataset/dataset/benchmark/data/vot2016', version=2016,
                          #result_dir='results/ricky/origin/epoch{}'.format(i),
                          #report_dir='reports/ricky/origin/epoch{}'.format(i))
        ]
        for e in experiments:
            e.run(tracker, visualize=False)
            e.report([tracker.name])
