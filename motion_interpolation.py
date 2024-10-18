import cv2
import torch
import numpy as np
from RIFE_HDv3 import Model
import subprocess
import os
from concurrent.futures import ThreadPoolExecutor

class MotionInterpolator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = None

    def load_model(self):
        print("Loading RIFE model...")
        self.model = Model()
        self.model.load_model('flownet.pkl', -1)
        self.model.eval()
        self.model.device()
        print("RIFE model loaded successfully.")

    def load_image(self, img):
        print("Loading image...")
        return torch.from_numpy(img.transpose(2, 0, 1)).float()[None].to(self.device, non_blocking=True) / 255.0

    def resize_frame_cuda(self, frame, width, height):
        print(f"Resizing frame to {width}x{height}...")
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gpu_resized = cv2.cuda.resize(gpu_frame, (width, height))
            frame_resized = gpu_resized.download()
            print("Frame resized using CUDA.")
        else:
            frame_resized = cv2.resize(frame, (width, height))
            print("Frame resized using CPU.")
        return frame_resized

    def process_frame(self, frame, width, height, expected_frame_size):
        print("Processing frame...")
        frame_resized = self.resize_frame_cuda(frame, width, height)
        actual_frame_size = frame_resized.nbytes
        assert actual_frame_size == expected_frame_size, f"Expected {expected_frame_size} bytes, got {actual_frame_size}"
        print("Frame processed successfully.")
        return frame_resized

    def interpolate_video(self, input_path, output_path, sf=2, prores_quality=3, progress_callback=None):
        print(f"Starting video interpolation: input={input_path}, output={output_path}, sf={sf}, prores_quality={prores_quality}")
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_frame_size = width * height * 3
        print(f"Video properties: fps={fps}, width={width}, height={height}, total_frames={total_frames}")

        output_path = os.path.splitext(output_path)[0] + '.mov'
        print(f"Output path set to: {output_path}")

        ffmpeg_path = os.path.join(os.path.dirname(__file__), 'ffmpeg', 'bin', 'ffmpeg')
        prores_profiles = ["proxy", "lt", "standard", "hq", "4444", "4444xq"]
        prores_profile = prores_profiles[prores_quality - 1]
        command = [
            ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps * sf),
            '-i', '-',
            '-an',
            '-vcodec', 'prores_ks',
            '-profile:v', prores_profile,
            '-pix_fmt', 'yuv422p10le' if prores_profile not in ["4444", "4444xq"] else 'yuva444p10le',
            output_path
        ]
        print("FFmpeg command:", " ".join(command))
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        ret, prev = cap.read()
        if not ret:
            raise RuntimeError("Failed to read the first frame")

        prev = self.load_image(prev)
        frame_count = 0
        total_frames_to_write = total_frames * sf

        print("Starting frame processing loop...")
        try:
            with ThreadPoolExecutor(max_workers=2) as executor, torch.no_grad():
                while True:
                    ret, curr = cap.read()
                    if not ret:
                        print("Reached end of input video.")
                        break

                    curr = self.load_image(curr)

                    print(f"Processing frame {frame_count + 1}/{total_frames}")
                    future = executor.submit(self.process_frame, (prev[0] * 255).byte().cpu().numpy().transpose(1, 2, 0), width, height, expected_frame_size)
                    frame = future.result()
                    process.stdin.write(frame.tobytes())
                    frame_count += 1

                    for i in range(1, sf):
                        print(f"Interpolating frame {frame_count + 1}/{total_frames_to_write}")
                        t = i / sf
                        middle = self.model.inference(prev, curr, t)
                        future_interpolated = executor.submit(self.process_frame, (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0), width, height, expected_frame_size)
                        interpolated_frame = future_interpolated.result()
                        process.stdin.write(interpolated_frame.tobytes())
                        frame_count += 1

                    if progress_callback:
                        progress_callback(frame_count, total_frames_to_write)

                    prev = curr

                print("Processing last frame...")
                frame = (prev[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)
                frame_resized = self.resize_frame_cuda(frame, width, height)
                process.stdin.write(frame_resized.tobytes())
                frame_count += 1

            print("Closing FFmpeg process...")
            process.stdin.close()
            stdout, stderr = process.communicate(timeout=60)  # Wait up to 60 seconds for FFmpeg to finish
            if process.returncode != 0:
                print(f"FFmpeg stdout: {stdout.decode()}")
                print(f"FFmpeg stderr: {stderr.decode()}")
                raise RuntimeError(f"FFmpeg failed with error code {process.returncode}: {stderr.decode()}")

        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            print(f"FFmpeg timed out. stdout: {stdout.decode()}")
            print(f"FFmpeg timed out. stderr: {stderr.decode()}")
            raise RuntimeError("FFmpeg process timed out")
        except Exception as e:
            process.kill()
            stdout, stderr = process.communicate()
            print(f"Exception occurred. FFmpeg stdout: {stdout.decode()}")
            print(f"Exception occurred. FFmpeg stderr: {stderr.decode()}")
            raise RuntimeError(f"Error during FFmpeg encoding: {str(e)}")
        finally:
            cap.release()

        print(f"Video interpolation complete. Total frames written: {frame_count}")
        return frame_count, total_frames_to_write