import os

BINARIES_PATHS = [
    os.path.join(os.path.join(LOADER_DIR, '../../'), 'x64/vc17/bin'),
    os.path.join(os.getenv('CUDA_PATH', 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1'), 'bin')
] + BINARIES_PATHS
