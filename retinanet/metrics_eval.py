import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ann', '--annotations_file', required=True, type=str)
    parser.add_argument('-det', '--detections_file', required=True, type=str)
    parser.add_argument('-area', '--area', nargs=2, type=str, default=['0**2', '1e5**2'])
    return parser


np.warnings.filterwarnings("ignore")


PR_CURVES = [
    # {"iouThr": 0.5, "area": "all", "class": "person", "maxDet": 10},
    # {"iouThr": 0.75, "area": "all", "class": "person", "maxDet": 10}
]


class Params:

    def __init__(self, gt, iouType, area=(0**2, 1e5**2)):
        """
        iouType - one of 'bbox', 'segm'
        """
        area = list(area)
        # список id изображений для подсчета метрик
        # пустой - использовать все
        self.imgIds = []

        self.classes = []

        # пороги IoU
        self.iouThrs = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        # площади объектов, для которых будут вычислeны метрики
        self.areas = {
            "custom": area,
            "all": [0 ** 2, 1e5 ** 2],
            "small": [0 ** 2, 32 ** 2],
            "medium": [32 ** 2, 96 ** 2],
            "large": [96 ** 2, 1e5 ** 2]
        }

        self.maxDets = [1, 10, 100]

        # остальное, как правило, нет причин менять
        self.id_to_class = {cat_id: cat["name"] for cat_id, cat in gt.cats.items()}

        self.class_to_id = {cat["name"]: cat_id for cat_id, cat in gt.cats.items()}
        self.catIds = [self.class_to_id[cls] for cls in self.classes] or list(gt.cats.keys())
        self.useCats = 1
        self.iouType = iouType
        self.useSegm = None
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.areaRngLbl = list(self.areas.keys())
        self.areaRng = [self.areas[k] for k in self.areaRngLbl]
        if not self.imgIds:
            self.imgIds = sorted(gt.getImgIds())


def detection_metrics(coco_gt, coco_dt, params):
    def calk_cond_mean(s, area, cat_id=-1, iouThr="mean", maxDet=-1):
        p = coco_eval.params
        s = s[:, :, list(p.areaRngLbl).index(area), p.maxDets.index(maxDet)]
        if cat_id != -1:
            s = s[:, p.catIds.index(cat_id)]
        if iouThr != "mean":
            s = s[list(p.iouThrs).index(iouThr)]
        valid = s > -1
        return np.mean(s[valid]) if valid.any() else -1

    def AP(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['precision'].mean(axis=1)
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def AR(area, cat_id=-1, iouThr=None, maxDet=-1):
        s = coco_eval.eval['recall']
        return calk_cond_mean(s, area, cat_id, iouThr, maxDet)

    def pr_curve(area, cat_id, iouThr, maxDet):
        p = coco_eval.params
        recall = p.recThrs
        ti = list(p.iouThrs).index(iouThr)
        ki = list(p.catIds).index(cat_id)
        ai = list(p.areaRngLbl).index(area)
        di = list(p.maxDets).index(maxDet)
        precision = coco_eval.eval['precision'][ti, :, ki, ai, di]
        return recall, precision

    coco_eval = COCOeval(coco_gt, coco_dt, params.iouType)
    coco_eval.params = params
    coco_eval.evaluate()
    coco_eval.accumulate()

    metrics = []
    p = coco_eval.params
    for cat_id in p.catIds:
        for area in p.areaRngLbl:
            for maxDet in p.maxDets:
                for iouThr in p.iouThrs:
                    ap = AP(area, cat_id, iouThr, maxDet)
                    ar = AR(area, cat_id, iouThr, maxDet)
                    recall, precision = pr_curve(area, cat_id, iouThr, maxDet)
                    metrics.append({
                        "class": p.id_to_class[cat_id],
                        "area": area,
                        "maxDet": maxDet,
                        "iouThr": iouThr,
                        "AP": ap,
                        "AR": ar,
                        "recall": list(recall),
                        "precision": list(precision)
                    })

    return pd.DataFrame(metrics)


def save_csv(metrics, folder):
    path = os.path.join(folder, "metrics.csv")
    metrics.to_csv(path, index=False)


def save_report(metrics, folder=None):
    f = None
    if folder is not None:
        f = open(os.path.join(folder, "metrics.txt"), "w")

    area_list = sorted(set(metrics["area"]))
    maxDet_list = sorted(set(metrics["maxDet"]))
    iouThr_list = sorted(set(metrics["iouThr"]))

    mean_msg = "[area = {:6s} | IoU = {:<4} | maxDets = {:<3} ]  mAP = {:0.3f}  mAR = {:0.3f}"
    indexed = metrics.set_index(["area", "maxDet"])
    for area in area_list:
        for maxDet in maxDet_list:
            sdf = indexed.loc[(area, maxDet)]
            mAP, mAR = sdf["AP"].mean(), sdf["AR"].mean()
            print(mean_msg.format(area, "mean", maxDet, mAP, mAR), file=f)

            sdf = sdf.reset_index().set_index(["area", "maxDet", "iouThr"])
            for iouThr in iouThr_list:
                ssdf = sdf.loc[(area, maxDet, iouThr)]
                mAP, mAR = ssdf["AP"].mean(), ssdf["AR"].mean()
                print(mean_msg.format(area, iouThr, maxDet, mAP, mAR), file=f)
            print(file=f)

    if f is not None:
        f.close()


def save_pr_curves(metrics, pr_curves, folder):
    indexed = metrics.set_index(["class", "iouThr", "area", "maxDet"])
    fmt = "class={class}-iouThr={iouThr}-area={area}-maxDet={maxDet}.png"
    for p in pr_curves:
        idx = p["class"], p["iouThr"], p["area"], p["maxDet"]
        recall = indexed.loc[idx, "recall"]
        precision = indexed.loc[idx, "precision"]
        plt.clf()
        plt.title("AP = {:.3f}".format(np.mean(precision)))
        plt.plot(recall, precision)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.grid()
        plt.savefig(os.path.join(folder, fmt.format(**p)))


def score_filter(dt_json, args):
    dt_json_new = []
    for ann in dt_json:
        if ann["score"] >= args.score_thr:
            dt_json_new.append(ann)
    return dt_json_new


def evaluate_detections(annotations_file, detections_file, area=(0**2, 1e5**2)):
    coco_gt = COCO(annotations_file)
    with open(detections_file) as f:
        dt_json = json.load(f)
    coco_dt = coco_gt.loadRes(dt_json)
    params = Params(coco_gt, iouType='bbox', area=area)
    metrics = detection_metrics(coco_gt, coco_dt, params)
    return metrics


def get_classes(metrics):
    classes = list()
    for cl in metrics['class']:
        if cl not in classes:
            classes.append(cl)
    return classes


def extract_mAP(metrics, iouThrs=0.5):
    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    indexed = metrics.set_index(["area", "maxDet"])
    area = 'custom'
    maxDet = 100
    mAPs = []
    for iouThr in iouThrs:
        mAP = indexed.loc[(area, maxDet)].reset_index().set_index(["area", "maxDet", "iouThr"]).loc[(area, maxDet, iouThr)]["AP"].mean()
        mAPs.append(mAP)

    if iouThrs_type in (float, int):
        mAPs = mAPs[0]
    return mAPs


def extract_AP(metrics, classes, iouThrs=0.5):
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    classes_type = type(classes)
    if classes_type in (str,):
        classes = (classes,)

    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    APs = []
    area = 'custom'
    maxDet = 100
    for iouThr in iouThrs:
        APs_iouThr = []
        for cl in classes:
            cl_idxes = [idx for idx, value in enumerate(metrics['class']) if value == cl]
            area_idxes = [idx for idx, value in enumerate(metrics['area']) if value == area and idx in cl_idxes]
            maxDet_idxes = [idx for idx, value in enumerate(metrics['maxDet']) if value == maxDet and idx in area_idxes]
            iouThr_idxes = [idx for idx, value in enumerate(metrics['iouThr']) if value == iouThr and idx in maxDet_idxes]
            assert len(iouThr_idxes) == 1
            idx = iouThr_idxes[0]
            APs_iouThr.append(metrics['AP'][idx])
        APs.append(APs_iouThr)

    if iouThrs_type in (float, int):
        APs = APs[0]
    if classes_type in (str,):
        if iouThrs_type in (float, int):
            APs = APs[0]
        else:
            APs = [APs[i][0] for i in range(len(APs))]
    return APs


def get_optimal_score_threshold(metrics, classes, iouThrs=0.5):
    iouThrs_type = type(iouThrs)
    if iouThrs_type in (float, int):
        iouThrs = (iouThrs,)
    classes_type = type(classes)
    if classes_type in (str,):
        classes = (classes,)

    permitted_iouThrs = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
    for iouThr in iouThrs:
        assert iouThr in permitted_iouThrs

    class gt:
        def __init__(self):
            self.cats = dict()
        def getImgIds(self):
            return [0]
    metrics_thrs = Params(gt(), iouType='bbox').recThrs

    opt_thrs = []
    area = 'custom'
    maxDet = 100
    for iouThr in iouThrs:
        opt_thrs_iouThr = []
        for cl in classes:
            cl_idxes = [idx for idx, value in enumerate(metrics['class']) if value == cl]
            area_idxes = [idx for idx, value in enumerate(metrics['area']) if value == area and idx in cl_idxes]
            maxDet_idxes = [idx for idx, value in enumerate(metrics['maxDet']) if value == maxDet and idx in area_idxes]
            iouThr_idxes = [idx for idx, value in enumerate(metrics['iouThr']) if
                            value == iouThr and idx in maxDet_idxes]
            assert len(iouThr_idxes) == 1
            idx = iouThr_idxes[0]
            diff = 2  # >1
            opt_thr = -1
            for i in range(len(metrics['recall'][idx])):
                new_diff = abs(metrics['recall'][idx][i] - metrics['precision'][idx][i])
                if new_diff < diff:
                    diff = new_diff
                    opt_thr = metrics_thrs[i]
            assert opt_thr != -1
            opt_thrs_iouThr.append(opt_thr)
        opt_thrs.append(opt_thrs_iouThr)

    if iouThrs_type in (float, int):
        opt_thrs = opt_thrs[0]
    if classes_type in (str,):
        if iouThrs_type in (float, int):
            opt_thrs = opt_thrs[0]
        else:
            opt_thrs = [opt_thrs[i][0] for i in range(len(opt_thrs))]
    return opt_thrs


def print_metrics(annotations_file, detections_file, area=(0**2, 1e5**2)):
    if area[1] == -1:
        area = (area[0], 1e5**2)
    metrics = evaluate_detections(annotations_file, detections_file, area)
    classes = get_classes(metrics)
    AP = extract_AP(metrics, classes)
    mAP = extract_mAP(metrics, iouThrs=0.5)
    for i in range(len(classes)):
        print('{:15} {}'.format(classes[i], AP[i]))
    print('')
    print('{:15} {}'.format('mAP', mAP))


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.area = list(map(eval, args.area))
    if args.area[1] == -1:
        args.area[1] = 1e5 ** 2
    print_metrics(**vars(args))
