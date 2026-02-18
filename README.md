# PvZ AI Agent

## Model Download

Download the model files from Google Drive:

https://drive.google.com/drive/folders/1zopiD07pOf8oPQTquQhOvWvU8NW2KAur?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

## Where To Place Models

- CNN model (`best.pt`):
  - `model/runs/cnn/best.pt`

- YOLO ONNX model (`best.onnx`) for optimized runtime:
  - `model/runs/detect/train5/weights/best.onnx`

- Optional YOLO PyTorch fallback (`best.pt`):
  - `model/runs/detect/train5/weights/best.pt`

## Notes

- The game loop defaults to:
  - YOLO: `model/runs/detect/train5/weights/best.onnx` (ONNX runtime path)
  - CNN: `model/runs/cnn/best.pt`
- If ONNX is missing and `model/runs/detect/train5/weights/best.pt` exists,
  the game loop automatically falls back to the `.pt` model.
- If you use different paths, pass them with CLI flags in your run command.
