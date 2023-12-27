
# Evaluation Steps

## 1. Data preparation

Download images and annotations from the huggingface. <https://huggingface.co/datasets/omlab/OVDEval>


The dataset is distributed as follows:
```
├── images
│   ├── negation
│   │   └── 64d22ce1e4b011b0db94bd72.jpg
│   │   └── 64d22ce4e4b011b0db94bd98.jpg
│   └── celebrity
│   ├── color
│   └── landmark
│   ├── logo
│   └── material
│   ├── position
│   └── relationship     
├── annotations
│   ├── negation.json
│   └── celebrity.json
│   ├── color.json
│   └── landmark.json
│   ├── logo.json
│   └── material.json
│   ├── position.json
│   └── relationship.json    
```
### Dataset Structure
```
{
  "categories": [
    {
      "supercategory": "object",
      "id": 0,
      "name": "computer without screen on"
    },
    {
      "supercategory": "object",
      "id": 1,
      "name": "computer with screen on"
    }
]
  "annotations": [
    {
      "id": 0,
      "bbox": [
        111,
        117,
        99,
        75
      ],
      "category_id": 0,
      "image_id": 0,
      "iscrowd": 0,
      "area": 7523
    }]
  "images": [
    {
      "file_name": "64d22c6fe4b011b0db94b993.jpg",
      "id": 0,
      "height": 254,
      "width": 340,
      "text": [
        "computer without screen on"  # "text" represents the annotated positive labels of this image.
      ],
      "neg_text": [
        "computer with screen on" # "neg_text" contains fine-grained hard negative labels which are generated according specific sub-tasks.
      ]
    }]
}
```

## 2. Clone this repository 
```
git clone https://github.com/om-ai-lab/OVDEval.git
```

## 3. Result preparation

Explanation: The annotation format of the OVDEval dataset is different from the COCO format. Specifically, each image annotation has different labels, and the "images" section contains "text" and "neg_text". Therefore, inference needs to be done on each image individually.

The specific steps are as follows:

1. Take "text" and "neg_text" as the labels for each image and input them into the model.
2. Use your model to infer the results for each image and save them in the following format. The "category_id" should correspond to the "id" of the category in the annotation file's "annotations" section.
3. Use the processed JSON file and our provided eval.py script to calculate the NMS-AP score.

```
[
  {
    "image_id": 0,
    "score": 0.52312356,
    "category_id": 0,
    "bbox": [
      91,
      0,
      810,
      537
    ]
  },
  {
    "image_id": 0,
    "score": 0.24432115,
    "category_id": 0,
    "bbox": [
      341,
      320,
      433,
      216
    ]
  }
]
```

## 4. Evaluation
```
python eval.py \
    --gt-path annotations/material.json \
    --result-path results.json \
    --image-path images/material/ \
    --output-path output 
```
Parameter Description:

- gt-path: The path to the annotation file downloaded from Hugging Face.
- result-path: The path to the model's output, following the COCO output format.
- image-path: The path to the image corresponding to gt-path.
- output-path: The path to save the processed output file.

## Example

Below, we provide the testing code for the GLIP model on the "material" test set.

```
python eval_glip.py \
    --gt-path annotations/material.json \
    --image-path images/material/ \
    --model-path MODEL/glip_large_model.pth \
    --cfg-path GLIP/configs/pretrain/glip_Swin_L.yaml \
    --output-path output 
```

Example output:
```
==============Now testing material.
==============Before NMS: 87349
==============After NMS: 87349
The filtered result is saved to the output/material_results.json file.
loading annotations into memory...
Done (t=0.03s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.18s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=4.62s).
Accumulating evaluation results...
DONE (t=1.29s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.074
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.094
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.082
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.076
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.089
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.147
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.225
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.231
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.089
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.173
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.239
 ```