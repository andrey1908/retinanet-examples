import argparse
import json
import os
import xml.etree.ElementTree as xml
from xml.dom import minidom
from PIL import Image
from cvat2coco import cvat2coco
from metrics_eval import print_metrics
from correct_detections import correct_detections


# def build_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-eng', '--engine_file', required=True, type=str)
#     parser.add_argument('-out', '--out_file', required=True, type=str)
#     parser.add_argument('-to', '--predict_to', type=str, choices=['cvat', 'coco'], default='cvat')
#     parser.add_argument('-det_only', '--detections_only', action='store_true')
#     parser.add_argument('-img_fld', '--images_folder', required=True, type=str)
#     parser.add_argument('-img_cl', '--images_and_classes_file', type=str)
#     parser.add_argument('-thr', '--threshold', type=float, default=0.)
#     return parser


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-eng', '--engine-file', required=True, type=str)  # ../models/640x192.plan
    parser.add_argument('-ann', '--annotations-file', required=True, type=str)  # ../data/kitti/val_person.json
    parser.add_argument('-img-fld', '--images-folder', required=True, type=str)  # ./kitti/images
    parser.add_argument('-area', '--area', nargs=2, type=int, default=[40**2, 1e5**2])
    return parser


def get_images_from_coco(annotations_file, images_folder):
    with open(annotations_file, 'r') as f:
        json_dict = json.load(f)
    images = json_dict['images']
    image_names = list()
    image_ids = list()
    for image in images:
        image_names.append(os.path.join(images_folder, image['file_name']))
        image_ids.append(image['id'])
    return image_names, image_ids


def save_image_names_and_ids(image_names, image_ids, file_name):
    assert len(image_names) == len(image_ids)
    lines = list()
    for image_name, image_id in zip(image_names, image_ids):
        line = str(image_id) + ' ' + os.path.join('../data', image_name) + '\n'
        lines.append(line)
    with open(file_name, 'w') as f:
        f.writelines(lines)


def get_classes(dataset='kitti_person'):
    if dataset == 'kitti_person':
        classes = ('Person', 'Cyclist', 'Car', 'Van', 'Tram', 'Truck', 'Misc')
    elif dataset == 'modified_coco':
        classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'traffic_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
    return classes


def convert_detections_to_cvat(det_txt_file, det_cvat_file, classes):
    with open(det_txt_file, 'r') as f:
        lines = f.readlines()
    annotations = xml.Element("annotations")
    meta = xml.SubElement(annotations, "meta")
    task = xml.SubElement(meta, "task")
    size = xml.SubElement(task, "size")
    mode = xml.SubElement(task, "mode").text = "annotation"
    overlap = xml.SubElement(task, "overlap").text = "0"
    flipped = xml.SubElement(task, "flipped").text = "False"
    labels = xml.SubElement(task, "labels")
    for cl in classes:
        label = xml.SubElement(labels, "label")
        name = xml.SubElement(label, "name").text = cl

    num_images = 0
    for line in lines:
        line = line[:-1].split()
        if len(line) == 2:
            image = dict()
            image_name = line[1]
            im = Image.open(image_name)
            image['id'] = line[0]
            image['name'] = image_name
            image['width'] = str(im.size[0])
            image['height'] = str(im.size[1])
            img = xml.SubElement(annotations, "image", image)
            num_images += 1
            continue
        if float(line[0]) < 0.3:
            continue
        bbox = dict()
        bbox['occluded'] = '0'
        bbox['label'] = classes[int(line[5])]
        bbox['xtl'] = line[1]
        bbox['ytl'] = line[2]
        bbox['xbr'] = line[3]
        bbox['ybr'] = line[4]
        bbox['score'] = line[0]
        xml.SubElement(img, "box", bbox)
    size.text = str(num_images)

    rough_string = xml.tostring(annotations, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(det_cvat_file, "w") as f:
        f.writelines(reparsed.toprettyxml(indent="  "))


def validate(engine_file, annotations_file, images_folder, area=(40**2, 1e5**2)):
    if not os.path.exists('../temp'):
        os.mkdir('../temp')
    image_names, image_ids = get_images_from_coco(annotations_file, images_folder)
    save_image_names_and_ids(image_names, image_ids, '../temp/images_and_ids.txt')
    os.system('../extras/cppapi/build/predict ../temp/images_and_ids.txt ' + engine_file + ' ../temp/det.txt')
    convert_detections_to_cvat('../temp/det.txt', '../temp/det.xml', get_classes('kitti_person'))
    cvat2coco('../temp/det.xml', '../temp/det.json', detections_only=True)
    os.system("mv ../temp/det.json ../temp/detections.json")
    print_metrics(annotations_file, '../temp/detections.json', area)

if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    validate(**vars(args))
