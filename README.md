# RF-DETR with USLS

This project shows how we can use SOTA object detection model [RF-DETR][2] with [USLS][1] to detect objects in a video using Rust and ONNX Runtime.

I was testing the code on MacOS with M1 chip, but it should work on other platforms as well.

## Download onnxruntime

```shell
wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz

tar xf onnxruntime-osx-arm64-1.20.1.tgz && rm onnxruntime-osx-arm64-1.20.1.tgz
```

## Download base ONNX model

```shell
wget -O "base_rfdetr.onnx" "https://huggingface.co/onnx-community/rfdetr_base-ONNX/resolve/main/onnx/model.onnx"
wget -O "base_rfdetr_fp16.onnx" "https://huggingface.co/onnx-community/rfdetr_base-ONNX/resolve/main/onnx/model_fp16.onnx"

wget -O "model_q4f16.onnx" "https://huggingface.co/onnx-community/rfdetr_base-ONNX/resolve/main/onnx/model_q4f16.onnx"

# model_int8.onnx does not work:
# Error: Could not find an implementation for ConvInteger(10) node with name '/backbone/backbone.0/encoder/encoder/embeddings/patch_embeddings/projection/Conv_quant'
```

## Download large ONNX model

```shell
wget -O "large_rfdetr.onnx" "https://huggingface.co/onnx-community/rfdetr_large-ONNX/resolve/main/onnx/model.onnx"

wget -O "large_rfdetr_fp16.onnx" "https://huggingface.co/onnx-community/rfdetr_large-ONNX/resolve/main/onnx/model_fp16.onnx"
```

## Download a sample video

```shell
yt-dlp "https://www.youtube.com/watch?v=1_FV-e9mXN0"
```

## Build

```shell
cargo build --release
```

## Run 

```shell
ORT_DYLIB_PATH=./onnxruntime-osx-arm64-1.20.1/lib/libonnxruntime.1.20.1.dylib ./target/release/rf-detr-usls --video-path test_video.mp4 --model-path model_q4f16.onnx
```

[1]: https://github.com/jamjamjon/usls/issues
[2]: https://github.com/roboflow/rf-detr

