# PvZ AI Agent

## Model Download

Download the model files from Google Drive:

https://drive.google.com/drive/folders/1zopiD07pOf8oPQTquQhOvWvU8NW2KAur?dmr=1&ec=wgc-drive-%5Bmodule%5D-goto

## Where To Place Models

- CNN model (`best.pt`):
  - `model/runs/cnn/best.pt`

- YOLO model (`best.pt`):
  - `model/runs/detect/train5/weights/best.pt`

- Optional YOLO ONNX export (`best.onnx`):
  - `model/runs/detect/train5/weights/best.onnx`

## Notes

- The game loop defaults to:
  - YOLO: `model/runs/detect/train5/weights/best.pt`
  - CNN: `model/runs/cnn/best.pt`
- If you use different paths, pass them with CLI flags in your run command.
