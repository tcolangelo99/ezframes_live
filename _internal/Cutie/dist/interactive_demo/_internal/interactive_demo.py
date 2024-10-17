import logging
import os
import sys

# Set up the Hydra config path
if getattr(sys, 'frozen', False):
    # we are running in a bundle
    bundle_dir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    bundle_dir = os.path.dirname(os.path.abspath(__file__))

# Set the Hydra config path
os.environ['HYDRA_CONFIG_PATH'] = os.path.join(bundle_dir, 'cutie', 'config')

# fix for Windows
if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

from argparse import ArgumentParser
import torch
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from PySide6.QtWidgets import QApplication
import qdarktheme
from gui.main_controller import MainController
from gui.resource_manager import ResourceManager

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument('--images', help='Folders containing input images.', default=None)
    parser.add_argument('--video', help='Video file readable by OpenCV.', default=None)
    parser.add_argument('--workspace', help='Directory for storing buffered images (if needed) and output masks', default=None)
    parser.add_argument('--num_objects', type=int, default=1)
    parser.add_argument('--workspace_init_only', action='store_true', help='Initialize the workspace and exit')
    args = parser.parse_args()
    return args

def run_interactive_demo(video_path=None):
    args = get_arguments()
    if video_path:
        args.video = video_path

    log = logging.getLogger()

    initialize(version_base='1.3.2', config_path="cutie/config", job_name="gui")
    cfg = compose(config_name="gui_config")

    torch.set_grad_enabled(False)
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    args.device = device
    log.info(f'Using device: {device}')

    args = vars(args)
    with open_dict(cfg):
        for k, v in args.items():
            assert k not in cfg, f'Argument {k} already exists in config'
            cfg[k] = v

    resource_manager = ResourceManager(cfg)
    
    try:
        app = QApplication(sys.argv)
        qdarktheme.setup_theme("auto")
        ex = MainController(cfg)  # Pass resource_manager if needed
        if 'workspace_init_only' in cfg and cfg['workspace_init_only']:
            return
        else:
            sys.exit(app.exec())
    finally:
        resource_manager.cleanup()

def main():
    args = get_arguments()
    run_interactive_demo(args.video)

if __name__ == "__main__":
    main()