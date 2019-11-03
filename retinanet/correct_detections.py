import argparse
import json


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-coco', '--coco-file', required=True, type=str)
    parser.add_argument('-out', '--out-file', required=True, type=str)
    return parser


def correct_detections(coco_file, out_file):
	

    with open(coco_file, 'r') as f:
        detections = json.load(f)
    old_categories = [{"supercategory": "none", "id": 1, "name": "Person"},
                      {"supercategory": "none", "id": 2, "name": "Cyclist"},
                      {"supercategory": "none", "id": 3, "name": "Car"},
                      {"supercategory": "none", "id": 4, "name": "Van"},
                      {"supercategory": "none", "id": 5, "name": "Tram"},
                      {"supercategory": "none", "id": 6, "name": "Truck"},
                      {"supercategory": "none", "id": 7, "name": "Misc"}]
    old_category_name_to_new_id = {'Person': 1, 'Car': 2}
    old_category_id_to_new = dict()
    for old_category in old_categories:
        if old_category['name'] in old_category_name_to_new_id.keys():
            old_category_id_to_new[old_category['id']] = old_category_name_to_new_id[old_category['name']]
    new_detections = list()
    for detection in detections:
        if detection['category_id'] not in old_category_id_to_new.keys():
            continue
        detection['category_id'] = old_category_id_to_new[detection['category_id']]
        new_detections.append(detection)

    with open(out_file, 'w') as f:
        json.dump(new_detections, f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    correct_detections(**vars(args))
