from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.4.0+cu118'
debug = False
cuda: Optional[str] = '11.8'
git_version = 'e4ee3be4063b7c430974252fdf7db42273388d86'
hip: Optional[str] = None