# DeepStream NX Integration

A Python-based DeepStream application that performs person detection, tracking, and face recognition, with gRPC integration for the NX Optix plugin.

## Features

- Person detection using YOLOv8
- Object tracking using NVIDIA's multi-object tracker
- Face recognition using ArcFace
- gRPC communication for video input and metadata output
- Support for both RTSP and gRPC video input
- JSON metadata output with bounding boxes, track IDs, and face identities

## Prerequisites

- NVIDIA GPU with CUDA support
- DeepStream SDK 6.0 or later
- Python 3.8 or later
- TensorRT 8.0 or later

## Installation

1. Install DeepStream SDK:
   ```bash
   # Follow NVIDIA's DeepStream installation guide
   # https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate gRPC Python code:
   ```bash
   python -m grpc_tools.protoc -I./protos --python_out=./generated_pb2 --grpc_python_out=./generated_pb2 ./protos/deepstream_grpc.proto
   ```

4. Convert models to TensorRT format:
   ```bash
   # Convert YOLOv8
   trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16

   # Convert ArcFace
   trtexec --onnx=models/arcface.onnx --saveEngine=models/arcface.engine --fp16
   ```

## Configuration

1. Update the configuration in `main.py`:
   - Set `input_type` to 'rtsp' or 'grpc'
   - Configure RTSP URI or gRPC server/client settings
   - Adjust display settings if needed

2. Place model files in the `models/` directory:
   - `yolov8s.engine`
   - `arcface.engine`
   - `yolov8_labels.txt`

3. Configure inference settings in:
   - `configs/config_infer_primary_yolo.txt`
   - `configs/config_infer_secondary_arcface.txt`

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. If using gRPC input:
   - The application will start a gRPC server on port 50051
   - Send video frames using the `FrameStreamer` service

3. If using RTSP input:
   - The application will connect to the specified RTSP stream
   - No additional setup needed

4. Metadata output:
   - The application will send JSON metadata to the configured gRPC endpoint
   - Default port: 50052

## Output Format

The application sends JSON metadata in the following format:

```json
{
    "timestamp": "2024-03-19T12:34:56.789Z",
    "frame_id": 123,
    "source_id": "camera1",
    "objects": [
        {
            "track_id": 1,
            "class": "person",
            "person_bbox": [100, 200, 300, 400],
            "face_identity": "john_doe",
            "face_embedding_available": true,
            "confidence_person": 0.95
        }
    ]
}
```

## Performance

- Target inference time: < 250ms per frame
- Optimized for NVIDIA GPUs using TensorRT
- Batch size: 1 (configurable in inference configs)

## Troubleshooting

1. If you encounter CUDA/GPU issues:
   - Verify CUDA installation
   - Check GPU memory usage
   - Ensure TensorRT models are compatible with your GPU

2. If gRPC communication fails:
   - Check network connectivity
   - Verify port availability
   - Check gRPC server/client configurations

3. If inference is slow:
   - Reduce input resolution
   - Adjust batch size
   - Check GPU utilization

## License

This project is licensed under the MIT License - see the LICENSE file for details. 