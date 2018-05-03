from cores.utils.data import image_classes
import os
import json
import argparse


def generate_child_labels(dataset_dir, black_list, merge_list):
    image_labels = image_classes(os.path.join(dataset_dir, 'train_val'))
    image_labels = [l for l in image_labels]
    merge_map = {}
    label_map = {}
    for merge_pair in merge_list:
        merge_target, merge_to = merge_pair.split(',')
        merge_map[int(merge_target)] = int(merge_to)
    count = 0
    for lb in image_labels:
        if lb not in black_list and lb not in merge_map:
            if lb in label_map: continue
            label_map[lb] = count
            count += 1
        elif lb not in black_list:
            m_t = merge_map[lb]
            if m_t not in label_map:
                label_map[m_t] = count
                count += 1
            label_map[lb] = label_map[m_t]

    with open(os.path.join(args.dataset_dir, args.output_name + '.json'), 'w') as f:
        json.dump(label_map, f)


def main():
    generate_child_labels(args.dataset_dir, args.blacklist, args.mergelist)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='child_labels'
    )
    parser.add_argument(
        '--blacklist',
        type=int,
        nargs='*',
        default=[]
    )
    parser.add_argument(
        '--mergelist',
        type=str,
        nargs='*',
        default=[]
    )
    args, unargs = parser.parse_known_args()
    main()