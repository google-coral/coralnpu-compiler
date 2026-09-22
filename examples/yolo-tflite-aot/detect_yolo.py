# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
from PIL import Image
from examples import utils

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def preprocess_image(image_path, size=640):
  img = Image.open(image_path).convert("RGB").resize((size, size))
  return np.expand_dims(np.array(img, dtype=np.float32), axis=0) / 255.0


def main(size=640):
  args = utils.parse_inference_args()
  script_dir = os.path.dirname(os.path.abspath(__file__))
  image_path = args.input_path or os.path.join(
      script_dir, "../mobilenetv2-jax-aot/cat.jpg")
  input_data = preprocess_image(image_path, size=size)

  predict_func = utils.load_vmfb(args.vmfb, module_name="module")
  output = np.asarray(predict_func(input_data))
  boxes = output[0, :4, :]
  class_scores = output[0, 4:, :]
  best_anchors = class_scores.argmax(axis=1)
  max_per_class = class_scores.max(axis=1)
  top_classes = np.argsort(max_per_class)[-5:][::-1]
  print("\nTop-5 YOLOv8n Detections:")
  for rank, cls_idx in enumerate(top_classes, 1):
    cx, cy, w, h = boxes[:, best_anchors[cls_idx]]
    print(
        f"  {rank}: {COCO_CLASSES[cls_idx]:<15} ({max_per_class[cls_idx]:.4f})"
        f"  box=[cx={cx:.1f}, cy={cy:.1f}, w={w:.1f}, h={h:.1f}]")
  utils.handle_results(max_per_class, args)


if __name__ == "__main__":
  main()
