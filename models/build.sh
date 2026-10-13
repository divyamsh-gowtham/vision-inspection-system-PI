#!/bin/sh
# Build TensorRT engines for the PP-OCRv4 pipeline. Engines are tied to this
# TensorRT version and GPU; rebuilt here, never copied between boards.
cd /home/jark/proinspect/models
T=/usr/src/tensorrt/bin/trtexec
nice -n 10 $T --onnx=ppocr_rec.onnx --saveEngine=ppocr_rec.engine --fp16 \
  --minShapes=x:1x3x48x32 --optShapes=x:1x3x48x320 --maxShapes=x:1x3x48x1600 > build_rec.log 2>&1
echo "rec exit $?" >> build_status.txt
nice -n 10 $T --onnx=ppocr_cls.onnx --saveEngine=ppocr_cls.engine --fp16 \
  --minShapes=x:1x3x48x32 --optShapes=x:1x3x48x192 --maxShapes=x:1x3x48x192 > build_cls.log 2>&1
echo "cls exit $?" >> build_status.txt
nice -n 10 $T --onnx=ppocr_det.onnx --saveEngine=ppocr_det.engine --fp16 \
  --minShapes=x:1x3x64x64 --optShapes=x:1x3x640x640 --maxShapes=x:1x3x960x960 > build_det.log 2>&1
echo "det exit $?" >> build_status.txt
echo "done" >> build_status.txt
