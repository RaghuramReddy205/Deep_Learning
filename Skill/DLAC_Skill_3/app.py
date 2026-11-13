import time
import uuid
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
import tritonclient.http as httpclient

app = FastAPI(title="VisionPort Inference Gateway", version="1.0.0")

TRITON_SERVER_URL = "localhost:8000"
MODEL_NAME = "yolov8_traffic_tensorrt"

try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
except Exception as e:
    print(f"Warning: Could not initialize Triton client: {e}")

def preprocess_image(image_bytes: np.ndarray) -> np.ndarray:
    """Resizes, normalizes, and changes channel layout for YOLOv8 input (3, 640, 640)."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image file format.")
    
    # Preprocessing steps
    img_resized = cv2.resize(img, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
    batch_tensor = np.expand_dims(img_transposed, axis=0)      # Add batch dimension
    return batch_tensor

@app.post("/infer/traffic")
async def run_inference(file: UploadFile = File(...)):
    request_id = f"vp-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    
    # 1. API Ingress & Validation
    t_ingress = time.perf_counter()
    image_bytes = await file.read()
    ingress_latency = (time.perf_counter() - t_ingress) * 1000

    try:
        # 2. Preprocessing Stage
        t_prep = time.perf_counter()
        input_data = preprocess_image(image_bytes)
        prep_latency = (time.perf_counter() - t_prep) * 1000

        # 3. Triton & TensorRT Execution Stage
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [httpclient.InferRequestedOutput("output0")]

        t_inference = time.perf_counter()
        response = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)
        inference_latency = (time.perf_counter() - t_inference) * 1000

        result_tensor = response.as_numpy("output0")

        # 4. Postprocessing Stage
        t_post = time.perf_counter()
        # Simulated post-processing / NMS extraction placeholder
        detected_objects = [
            {"class_id": 2, "class_name": "car", "confidence": 0.94, "box": [120, 40, 310, 190]}
        ]
        post_latency = (time.perf_counter() - t_post) * 1000

        total_latency = (time.perf_counter() - start_time) * 1000

        return {
            "status": "success",
            "request_id": request_id,
            "lifecycle_trace": {
                "ingress_ms": round(ingress_latency, 2),
                "preprocessing_ms": round(prep_latency, 2),
                "gpu_inference_ms": round(inference_latency, 2),
                "postprocessing_ms": round(post_latency, 2),
                "total_roundtrip_ms": round(total_latency, 2)
            },
            "predictions": detected_objects
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str.strip(str(e)))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)