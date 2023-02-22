import utils
import numpy as np

from dota import anomalies

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
    average_precision_score, classification_report


def evaluation(outputs, targets, FPS, info=None, **kwargs):
    preds = np.array(utils.flat_list(outputs))
    gts = np.array(utils.flat_list(targets))
    F1_mean, _, F1_one = f1_mean(gts, preds)
    return (
        roc_auc_score(gts, preds),
        average_precision_score(gts, preds),
        F1_one,
        F1_mean,
        accuracy_score(gts, preds > 0.5),
        classification_report(
            gts, preds > 0.5, target_names=['normal', 'anomaly']),
        get_eval_per_class(outputs, targets, info, utils.split_by_class),
        get_eval_per_class(outputs, targets, info, utils.split_by_class_ego),
    )


def print_results(cfg, AUC_frame, PRAUC_frame, f1score, f1_mean, accuracy,
                  report, eval_per_class, eval_per_class_ego):
    if cfg.machine_reading:
        res = []
        pres = ''
        if eval_per_class is not None:
            clss = sorted(eval_per_class.keys())
            fauc = [list(eval_per_class[k])[0] for k in clss]
            fmean = [list(eval_per_class[k])[2] for k in clss]
            res = fmean + fauc
            pres = ' - ' + '{:05f} '*len(fmean) + '- ' + '{:05f} '*len(fauc)
        if eval_per_class_ego is not None:
            def print_with_ego(eval_res, ego):
                clss = sorted(set([cls for cls, _ in eval_res.keys()]))
                fauc_ego = [list(eval_res[(cls, ego)])[0] for cls in clss]
                fmean_ego = [list(eval_res[(cls, ego)])[2] for cls in clss]
                res_ego = fmean_ego + fauc_ego
                pres_ego = ' - ' + '{:05f} '*len(fmean_ego) + \
                    '- ' + '{:05f} '*len(fauc_ego)
                return res_ego, pres_ego

            re0, pre0 = print_with_ego(eval_per_class_ego, 0.)
            re1, pre1 = print_with_ego(eval_per_class_ego, 1.)
            res += re0 + re1
            pres += pre0 + pre1

        print((
            'results {} {:05f} {:05f} {:05f} {:05f} {:05f}' + pres
            ).format(
                cfg.epoch, AUC_frame, PRAUC_frame, f1score, accuracy, f1_mean,
                *res
        ))
    else:
        print()
        print("[Correctness] f-AUC = %.5f" % (AUC_frame))
        print("             PR-AUC = %.5f" % (PRAUC_frame))
        print("           F1-Score = %.5f" % (f1score))
        print("           F1-Mean  = %.5f" % (f1_mean))
        print("           Accuracy = %.5f" % (accuracy))
        #  print("      accident pred = %.5f" % (acc_pred))
        #  print("        normal pred = %.5f" % (nor_pred))
        print()
        print(report)
        print()
        if eval_per_class is not None:
            print('F1-mean per class')
            for key, values in eval_per_class.items():
                print('{:03f} F1-mean class {}'.format(
                    list(values)[2], anomalies[int(key)-1]))
            print()
            print('f-AUC per class')
            for key, values in eval_per_class.items():
                print('{:03f} f-AUC class {}'.format(
                    list(values)[0], anomalies[int(key)-1]))


def f1_mean(gts, preds):
    F1_one = f1_score(gts, preds > 0.5)
    F1_zero = f1_score(
        (gts.astype('bool') == False).astype('long'),
        preds <= 0.5)
    F1_mean = 2 * (F1_one * F1_zero) / (F1_one + F1_zero)
    return F1_mean, F1_zero, F1_one


def get_eval_per_class(outputs, targets, info, split_fun):
    # retrocompat
    if info is None:
        return None
    # outputs/targets split per class
    ot = split_fun(outputs, targets, info)
    return {
        cls: {
            roc_auc_score(vals['targets'], vals['outputs']),
            average_precision_score(vals['targets'], vals['outputs']),
            f1_mean(vals['targets'], vals['outputs'])[0],
            accuracy_score(vals['targets'], vals['outputs'] > 0.5),
        } for cls, vals in ot.items()
    }
