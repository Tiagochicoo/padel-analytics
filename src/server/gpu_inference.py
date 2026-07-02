"""
gpu_inference.py — 2-model pipeline (detector + ball_tracknet)
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorrt as trt

# ── CUDA Runtime ──
cudart = ctypes.CDLL("libcudart.so")

def cuda_malloc(size):
    ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(ptr), size)
    if ret != 0: raise RuntimeError(f"cudaMalloc failed: {ret}")
    return ptr.value

def cuda_free(ptr):
    cudart.cudaFree(ctypes.c_void_p(ptr))

def cuda_memcpy_htod(dst, src_np):
    ret = cudart.cudaMemcpy(ctypes.c_void_p(dst), src_np.ctypes.data_as(ctypes.c_void_p), src_np.nbytes, 1)
    if ret != 0: raise RuntimeError("cudaMemcpy H2D failed")

def cuda_memcpy_dtoh(dst_np, src):
    ret = cudart.cudaMemcpy(dst_np.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(src), dst_np.nbytes, 2)
    if ret != 0: raise RuntimeError("cudaMemcpy D2H failed")

MODELS_DIR = Path("/home/tpereira/padel-cv-models")
HEATMAP_COLORS = ["#ff0000","#ff4400","#ff8800","#ffcc00","#ffff00","#88ff00","#00ff00","#00ff88"]

MODEL_DEFS = {
    "detector": {
        "path": MODELS_DIR / "detectors" / "yolo26n_base_fp16.engine",
        "classes": {0: "person"},
        "color": "#58a6ff", "type": "detection",
    },
    "ball_tracknet": {
        "path": MODELS_DIR / "ball_best_fp16.engine",
        "classes": {0: "ball"}, "color": "#ff3333", "type": "tracknet",
    },
}

TRACKNET_H, TRACKNET_W = 288, 512
TRACKNET_N_FRAMES = 9

class TRTEngine:
    def __init__(self, path):
        self.path = Path(path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(self.path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.input_name = self.input_size = None
        self.output_name = self.output_shape = self.output_size = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)
            vol = 1
            for d in shape:
                if d > 0: vol *= d
            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_size = vol * 4
            else:
                self.output_name = name
                self.output_shape = list(shape)
                self.output_size = vol * 4
                self.output_np = np.empty(shape, dtype=np.float32)
        self.d_input = cuda_malloc(self.input_size)
        self.d_output = cuda_malloc(self.output_size)

    def infer(self, img_np):
        img = Image.fromarray(img_np)
        img_resized = img.resize((640, 640), Image.LANCZOS)
        arr = np.array(img_resized, dtype=np.float32)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        arr /= 255.0
        arr = arr.transpose(2,0,1)
        arr = np.ascontiguousarray(arr[np.newaxis,...])
        cuda_memcpy_htod(self.d_input, arr)
        self.context.execute_v2([self.d_input, self.d_output])
        cudart.cudaDeviceSynchronize()
        cuda_memcpy_dtoh(self.output_np, self.d_output)
        return self.output_np.copy()

    def close(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)


class TrackNetEngine:
    def __init__(self, path):
        self.path = Path(path)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(self.path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        _vi = 1*27*TRACKNET_H*TRACKNET_W
        self.input_size = _vi*4
        _vo = 1*8*TRACKNET_H*TRACKNET_W
        self.output_size = _vo*4
        self.output_np = np.empty((1,8,TRACKNET_H,TRACKNET_W), dtype=np.float32)
        self.d_input = cuda_malloc(self.input_size)
        self.d_output = cuda_malloc(self.output_size)
        self.frame_buffer = []

    def add_frame(self, img_np):
        arr = np.array(Image.fromarray(img_np).resize((TRACKNET_W,TRACKNET_H), Image.LANCZOS), dtype=np.float32)
        if arr.ndim == 2: arr = np.stack([arr]*3, axis=-1)
        elif arr.shape[2] == 4: arr = arr[:,:,:3]
        arr /= 255.0
        arr = arr.transpose(2,0,1)
        self.frame_buffer.append(arr)
        if len(self.frame_buffer) > TRACKNET_N_FRAMES:
            self.frame_buffer.pop(0)

    def ready(self):
        return len(self.frame_buffer) == TRACKNET_N_FRAMES

    def infer(self):
        cat = np.concatenate(self.frame_buffer, axis=0)
        inp = np.ascontiguousarray(cat[np.newaxis,...])
        cuda_memcpy_htod(self.d_input, inp)
        self.context.execute_v2([self.d_input, self.d_output])
        cudart.cudaDeviceSynchronize()
        cuda_memcpy_dtoh(self.output_np, self.d_output)
        return self.output_np.copy()

    def parse_ball(self, heatmaps):
        hm = heatmaps[0] if heatmaps.ndim == 4 else heatmaps
        balls = []
        for i in range(min(hm.shape[0], 8)):
            hmap = hm[i]
            mx = float(hmap.max())
            if mx < 0.1: continue
            pos = np.unravel_index(hmap.argmax(), hmap.shape)
            balls.append({"x": pos[1]/TRACKNET_W, "y": pos[0]/TRACKNET_H, "conf": round(mx,4), "idx": i})
        return balls

    def close(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)


def parse_output(output, model_type, class_filter=None, conf_threshold=0.15):
    dets = output[0]
    results = []
    for d in dets:
        obj_conf = float(d[4])
        n = dets.shape[1]
        if model_type == "detection":
            if obj_conf < conf_threshold: continue
            if n == 6:
                cls_id = int(d[5])
                conf = obj_conf
            else:
                cls_scores = d[5:]
                cls_id = int(np.argmax(cls_scores))
                conf = obj_conf * float(cls_scores[cls_id])
                if conf < conf_threshold: continue
            if class_filter is not None and cls_id not in class_filter: continue
            results.append({"bbox": [float(d[0]),float(d[1]),float(d[2]),float(d[3])], "confidence": conf, "class_id": cls_id})
    return results


def draw_overlay(draw, detections, w, h, color, classes):
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except: font = ImageFont.load_default()
    for det in detections:
        cx, cy, bw, bh = det["bbox"]
        x1 = int((cx-bw/2)*w); y1 = int((cy-bh/2)*h)
        x2 = int((cx+bw/2)*w); y2 = int((cy+bh/2)*h)
        cls_name = classes.get(det["class_id"], "")
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        draw.text((x1+2,y1-14), f"{cls_name} {det['confidence']:.2f}", fill=color, font=font)

def draw_ball(draw, balls, w, h):
    for bp in balls:
        bx = int(bp["x"]*w); by = int(bp["y"]*h)
        r = max(3, int(8*min(1.0, bp["conf"])))
        ci = bp["idx"] % len(HEATMAP_COLORS)
        draw.ellipse([bx-r,by-r,bx+r,by+r], fill=HEATMAP_COLORS[ci], outline="white", width=2)

def draw_status_bar(draw, w, h, frame_num, frame_idx, total_frames, model_ms, live=False):
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except: font = ImageFont.load_default()
    x = w - 320; y = 8
    draw.text((x,y), "PadelClutch 2-model GPU", fill="#00ff9d", font=font)
    y += 15
    if live:
        draw.text((x,y), f"● LIVE  Frame {frame_num}", fill="#f85149", font=font)
    else:
        draw.text((x,y), f"Frame {frame_idx}/{total_frames}", fill="#ccc", font=font)
    y += 14
    for name in ["detector", "ball_tracknet"]:
        ms = model_ms.get(name, 0)
        c = MODEL_DEFS.get(name, {}).get("color", "#fff")
        draw.text((x,y), f"{name[:12]:12s} {ms:.0f}ms", fill=c, font=font)
        y += 12
