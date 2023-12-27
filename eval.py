import json, argparse, os
import numpy as np
import collections
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def calculate_iou(box1, boxes2):
    """
    Calculate the intersection ratio (IoU)
    """
    x1, y1, w1, h1 = box1
    x2 = boxes2[:, 0]
    y2 = boxes2[:, 1]
    w2 = boxes2[:, 2]
    h2 = boxes2[:, 3]

    xmin = np.maximum(x1, x2)
    ymin = np.maximum(y1, y2)
    xmax = np.minimum(x1 + w1, x2 + w2)
    ymax = np.minimum(y1 + h1, y2 + h2)

    intersection = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    union = w1 * h1 + w2 * h2 - intersection

    # Handle cases where the denominator is zero
    iou = np.where(union == 0, 0, intersection / union)

    return iou

def nms_threaded(gt, prediction, iou_threshold, num_threads):
    gt_boxes = np.array([box['bbox'] for box in gt])

    boxes = np.array([box['bbox'] for box in prediction])
    scores = np.array([box['score'] for box in prediction])

    keep_list = []
    remove_list = []
    for idx, i in enumerate(gt):
        gt_box = gt_boxes[idx]
        iou = calculate_iou(gt_box, boxes)
        indices =  np.where(iou > iou_threshold)[0].tolist()
        if indices:
            match_scores = scores[indices]
            sorted_indices = np.argsort(match_scores)[::-1]

            final_indices = []
            for sort_idx in sorted_indices.tolist():
                final_indices.append(indices[sort_idx])

            if final_indices:
                keep_list.append(final_indices[0])
                remove_list += final_indices[1:]

    final_remove_list = []
    for i in remove_list:
        if i not in keep_list:
            final_remove_list.append(i)

    selected_boxes = []
    for idx, i in enumerate(prediction):
        if idx not in final_remove_list:
            selected_boxes.append(i)

    return selected_boxes

def cal_map(gt_data, pred_data):
    
    anno = COCO(gt_data)
    pred = anno.loadRes(pred_data)
    eval = COCOeval(anno, pred, 'bbox')

    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-path", type=str, default="")
    parser.add_argument("--result-path", type=str, default="")
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=16)
   
    args = parser.parse_args()
    gt_path = args.gt_path
    result_path = args.result_path
    image_path = args.image_path
    output_path = args.output_path
    iou_threshold = args.iou_threshold
    num_threads = args.num_threads
    
    dataset_name = gt_path.rsplit('/',1)[-1].split('.')[0]
    print(f"==============Now testing {dataset_name}.")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # load annotation data
    gt_data = json.load(open(gt_path))
    
    before_nms_results = json.load(open(result_path))

    # nms with gt
    print("==============Before NMS:", len(before_nms_results))

    gt_data_ann = gt_data['annotations']
    
    image_dict = collections.defaultdict(list)
    gt_dict = collections.defaultdict(list)

    for i in before_nms_results:
        image_dict[i["image_id"]].append(i)

    for i in gt_data_ann:
        gt_dict[i["image_id"]].append(i)
    
    after_nms_results = []
    # Call multithreaded parallelized NMS functions
    for img, preds in image_dict.items():
        gts = gt_dict[img]
        selected_boxes = nms_threaded(gts, preds, iou_threshold, num_threads)
        after_nms_results += selected_boxes
    
    print("==============After NMS:", len(after_nms_results))
    save_path = os.path.join(output_path, f"{dataset_name}_results.json")
    json.dump(after_nms_results, open(save_path, 'w'), indent=2, ensure_ascii=False)  # save output
    
    print(f"The filtered result is saved to the {save_path} file.")
    
    # Calculate the score
    
    cal_map(gt_path, after_nms_results)
