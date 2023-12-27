import json, argparse, os
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.engine.inference import create_queries_and_maps
from typing import Generator, Sequence, List
from tqdm import tqdm 
import numpy as np
from PIL import Image
import torch
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

def load_model(model_path, cfg_path):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(cfg_path)
    cfg.merge_from_list(["MODEL.WEIGHT", model_path])
    cfg.freeze()
    model = build_detection_model(cfg)
    model.train(False)
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    _ = checkpointer.load(model_path)
    model = model.to('cuda')
    return model, cfg

def chunks(l: Sequence, n: int = 5, return_idx=False) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        if return_idx:
            yield l[i:i + n], (i, i+n)
        else:
            yield l[i:i + n]

def predict(model, original_image, img_feature, texts, cfg):
    
    captions, positive_map_label_to_token_i = create_queries_and_maps(list(range(1, len(texts) + 1)), texts, None,
                                                                      cfg=cfg)

    with torch.no_grad():
        output = model(img_feature, captions=[captions], positive_map=positive_map_label_to_token_i)
        prediction = [o.to(torch.device("cuda")) for o in output][0]
    height, width = original_image.shape[:-1]
    prediction = prediction.resize((width, height))
    return prediction

def post_process(predictions, threshold=0.5):
    scores = predictions.get_field("scores")
    labels = predictions.get_field("labels").tolist()
    thresh = scores.clone()
    for i, lb in enumerate(labels):
        thresh[i] = threshold
    keep = torch.nonzero(scores > thresh).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def batch_predict(model, images, transforms, texts, cfg, thresh):
    result = []
    for image in images:
        temp = []
        if True:
            img_feat = to_image_list(transforms(image), 32).to('cuda')
            predictions = predict(model, image, img_feat, texts, cfg)
            top_predictions = post_process(predictions, thresh)
            bbox = top_predictions.bbox.tolist()
            scores = top_predictions.extra_fields["scores"].tolist()
            labels = top_predictions.extra_fields["labels"].tolist()
            for b, s, l in zip(bbox, scores, labels):
                temp.append({"bbox": b, "score": s, "class": texts[l - 1]})
        result.append(temp)
    return result        
            
def inference_to_results(model, cfg, data: List, prompt: List[str], conf_threshold=0.0, batch_size=1, class_threshold_map: dict = {}):
    
    # load transforms
    transforms = build_transforms(cfg, False)
    
    image_data = [np.array(Image.open(x).convert('RGB'))[:, :, [2, 1, 0]] for x in data]
    resp = []
    for batch in chunks(image_data, batch_size):
        batch_y = batch_predict(model, batch, transforms, prompt, cfg, conf_threshold)
        for ind, k in enumerate(batch_y):
            temp = []
            for z in k:
                x, y, xx, yy, conf, cls = z["bbox"][0], z["bbox"][1], z["bbox"][2], z["bbox"][3], z["score"], z[
                    "class"]
                conf = float(conf)
                if len(class_threshold_map) > 0 and conf > class_threshold_map.get(cls, conf_threshold):
                    if int(x) - int(y) != 0 and int(xx) - int(yy) != 0:
                        temp.append({'xmin': float(x),
                                     'ymin': float(y),
                                     'xmax': float(xx),
                                     'ymax': float(yy),
                                     'conf': conf,
                                     'label': cls})
                else:
                    if int(x) - int(y) != 0 and int(xx) - int(yy) != 0:
                        temp.append({'xmin': float(x),
                                     'ymin': float(y),
                                     'xmax': float(xx),
                                     'ymax': float(yy),
                                     'conf': conf,
                                     'label': cls})
            resp.append(temp)
    return resp

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
    parser.add_argument("--image-path", type=str, default="")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--cfg-path", type=str, default="")
    parser.add_argument("--output-path", type=str, default="")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=16)
   
    args = parser.parse_args()
    gt_path = args.gt_path
    model_path = args.model_path
    cfg_path = args.cfg_path
    image_path = args.image_path
    output_path = args.output_path
    iou_threshold = args.iou_threshold
    num_threads = args.num_threads
    
    dataset_name = gt_path.rsplit('/',1)[-1].split('.')[0]
    print(f"==============Now testing {dataset_name}.")
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    # load model
    model, cfg = load_model(model_path, cfg_path)
    
    # load annotation data
    gt_data = json.load(open(gt_path))
    
    before_nms_results = []
    category_map = {}

    for cat in gt_data["categories"]:
        category_map[cat["name"]] = cat["id"]
        
    for idx, img in enumerate(tqdm(gt_data["images"])):
        prompt = []
        img_path = os.path.join(image_path, img["file_name"])
        if dataset_name in ['coco', 'celebrity', 'logo', 'landmark']:
            for cat in gt_data["categories"]:
                prompt.append(cat["name"])
        else:
            for t in img["text"] + img["neg_text"]:
                prompt.append(t)

        resp = inference_to_results(model, cfg, [img_path]*1, prompt, batch_size=1)

        for dd in range(len(resp[0])):
            pred_image = {"image_id": img["id"]}
            x1, y1, x2, y2 = resp[0][dd]["xmin"], resp[0][dd]["ymin"], resp[0][dd]["xmax"], resp[0][dd]["ymax"]
            w, h = x2 - x1, y2 - y1
            bbox = [int(x1), int(y1), int(w), int(h)]
            label = category_map[resp[0][dd]["label"]]
            score = resp[0][dd]["conf"]
            pred_image["score"], pred_image["category_id"] = round(float(score), 8), label
            pred_image["bbox"] = bbox

            before_nms_results.append(pred_image)

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
