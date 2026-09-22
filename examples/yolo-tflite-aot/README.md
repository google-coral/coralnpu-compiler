# YOLOv8n TFLite AOT Example

This directory provides two end-to-end flows for compiling and running YOLOv8n (`f32`) on CoralNPU:

1. **640x640 (Original Resolution)**
   - Downloads the standard `640x640` `yolov8n.tflite` (`8.7 GFLOPs`, `8400` anchors) and converts it to TOSA MLIR.
   - **Files**: `export_yolo.py`, `detect_yolo.py`, `test_yolo.sh`, `reference.npy`

2. **160x160 (Reduced Resolution for CI)**
   - Uses the same `yolov8n.pt` weights exported at `160x160` input resolution (`0.54 GFLOPs`, `525` anchors — 16x fewer MACs).
   - **Checked-in Model**: `yolov8n_160.tflite`
   - **Files**: `convert_yolo_160.py`, `export_yolo_160.py`, `detect_yolo_160.py`, `test_yolo_160.sh`, `reference_160.npy`

## Regenerating `yolov8n_160.tflite`

`convert_yolo_160.py` is a self-contained script using [`uv`](https://docs.astral.sh/uv/) that downloads `yolov8n.pt` and exports `yolov8n_160.tflite`:

```bash
uv run examples/yolo-tflite-aot/convert_yolo_160.py
```
