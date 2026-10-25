# COCO bbox evaluation

`cocoeval.py` is the official COCO API Python evaluator, retrieved as small source text from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py on 2026-10-25. The only source change is replacing its two removed `np.float` aliases with identical `np.float64` for NumPy 2 compatibility. Its BSD license is included. No dataset, model weights, compiled extension, or dependency was downloaded or installed.

`mask.py` supplies only the bbox `iou` primitive required by the evaluator, with COCO crowd intersection/detection-area semantics. Polygon/RLE segmentation evaluation is deliberately unsupported. Matching, crowd/ignore ordering, maxDets, 101-point PR and accumulation remain in official `cocoeval.py`.
