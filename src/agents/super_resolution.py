"""
Super Resolution Agent
Specialized agent for upscaling low-resolution images
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger
from pathlib import Path

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager

# Optional model imports (flag if missing)
try:
    import requests
except ImportError:
    requests = None
    logger.warning('requests not installed. Some model downloads will fail.')
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    RRDBNet = None
    logger.warning('basicsr not installed. BSRGAN and RRDBNet will not work.')
try:
    from swinir import SwinIR
except ImportError:
    SwinIR = None
    logger.warning('swinir not installed. SwinIR will not work.')
try:
    from restormer import Restormer
except ImportError:
    Restormer = None
    logger.warning('restormer not installed. Restormer will not work.')
try:
    from uformer import Uformer
except ImportError:
    Uformer = None
    logger.warning('uformer not installed. Uformer will not work.')
try:
    from nafnet import NAFNet
except ImportError:
    NAFNet = None
    logger.warning('nafnet not installed. NAFNet will not work.')
try:
    from swin2sr import Swin2SR
except ImportError:
    Swin2SR = None
    logger.warning('swin2sr not installed. Swin2SR will not work.')
try:
    from ipt import IPT
except ImportError:
    IPT = None
    logger.warning('ipt not installed. IPT will not work.')

class SuperResolutionAgent(BaseAgent):
    """
    SuperResolutionAgent supporting 10 real super-resolution/restoration models:
    - Real-ESRGAN, ESRGAN, BSRGAN, RRDBNet, SwinIR, Restormer, Uformer, NAFNet, Swin2SR, IPT
    """
    def __init__(self):
        super().__init__("SuperResolutionAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.model_names = [
            'real_esrgan', 'esrgan', 'bsrgan', 'rrdbnet', 'swinir',
            'restormer', 'uformer', 'nafnet', 'swin2sr', 'ipt'
        ]
        self.transforms = T.Compose([
            T.ToTensor(),
        ])
        
    async def _initialize(self) -> None:
        """Initialize all 10 super-resolution/restoration models"""
        try:
            logger.info("Initializing all 10 super-resolution models...")
            self.models['real_esrgan'] = await self._load_real_esrgan()
            self.prune_model(self.models['real_esrgan'])
            self.models['esrgan'] = await self._load_esrgan()
            self.prune_model(self.models['esrgan'])
            self.models['bsrgan'] = await self._load_bsrgan()
            self.prune_model(self.models['bsrgan'])
            self.models['rrdbnet'] = await self._load_rrdbnet()
            self.prune_model(self.models['rrdbnet'])
            self.models['swinir'] = await self._load_swinir()
            self.prune_model(self.models['swinir'])
            self.models['restormer'] = await self._load_restormer()
            self.prune_model(self.models['restormer'])
            self.models['uformer'] = await self._load_uformer()
            self.prune_model(self.models['uformer'])
            self.models['nafnet'] = await self._load_nafnet()
            self.prune_model(self.models['nafnet'])
            self.models['swin2sr'] = await self._load_swin2sr()
            self.prune_model(self.models['swin2sr'])
            self.models['ipt'] = await self._load_ipt()
            self.prune_model(self.models['ipt'])
            logger.info("All 10 super-resolution models initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize super-resolution models: {e}")
            raise
    
    async def _load_real_esrgan(self):
        """
        Load Real-ESRGAN model using torch.hub.
        Download weights if not present.
        Source: https://github.com/xinntao/Real-ESRGAN
        """
        try:
            model = torch.hub.load('xinntao/Real-ESRGAN', 'real_esrgan', pretrained=True).to(self.device)
            model.eval()
            logger.info("Loaded Real-ESRGAN model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN: {e}")
            raise
    
    async def _load_esrgan(self):
        """
        Load ESRGAN model (classic GAN-based super-resolution).
        Source: https://github.com/xinntao/ESRGAN
        """
        try:
            # ESRGAN uses RRDBNet architecture; use torch.hub for demo weights
            model = torch.hub.load('xinntao/ESRGAN', 'esrgan', pretrained=True).to(self.device)
            model.eval()
            logger.info("Loaded ESRGAN model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load ESRGAN: {e}")
            raise
    
    async def _load_bsrgan(self):
        """
        Load BSRGAN model (blind super-resolution).
        Source: https://github.com/cszn/BSRGAN
        """
        try:
            # BSRGAN is not on torch.hub; use official repo weights
            if RRDBNet is None:
                raise ImportError("basicsr not installed. BSRGAN and RRDBNet will not work.")
            model = RRDBNet(3, 3, 64, 23, gc=32)
            weights_url = 'https://github.com/cszn/BSRGAN/releases/download/v0.1/BSRGAN.pth'
            weights_path = 'BSRGAN.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded BSRGAN model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load BSRGAN: {e}")
            raise
    
    async def _load_rrdbnet(self):
        """
        Load RRDBNet model (backbone for Real-ESRGAN).
        Source: https://github.com/xinntao/ESRGAN
        """
        try:
            if RRDBNet is None:
                raise ImportError("basicsr not installed. BSRGAN and RRDBNet will not work.")
            model = RRDBNet(3, 3, 64, 23, gc=32)
            weights_url = 'https://github.com/xinntao/ESRGAN/releases/download/v0.1/RRDB_ESRGAN_x4.pth'
            weights_path = 'RRDB_ESRGAN_x4.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded RRDBNet model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load RRDBNet: {e}")
            raise
    
    async def _load_swinir(self):
        """
        Load SwinIR model (Swin Transformer for SR).
        Source: https://github.com/JingyunLiang/SwinIR
        """
        try:
            if SwinIR is None:
                raise ImportError("swinir not installed. SwinIR will not work.")
            model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0, depths=[6,6,6,6,6,6], embed_dim=180, num_heads=[6,6,6,6,6,6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
            weights_url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/swinir_sr_classical_patch64_x4_48.pth'
            weights_path = 'swinir_sr_x4.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded SwinIR model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load SwinIR: {e}")
            raise
    
    async def _load_restormer(self):
        """
        Load Restormer model (Efficient Transformer).
        Source: https://github.com/swz30/Restormer
        """
        try:
            if Restormer is None:
                raise ImportError("restormer not installed. Restormer will not work.")
            model = Restormer()
            weights_url = 'https://github.com/swz30/Restormer/releases/download/v0.0/Restormer_Denoising.pth'
            weights_path = 'Restormer_Denoising.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded Restormer model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Restormer: {e}")
            raise
    
    async def _load_uformer(self):
        """
        Load Uformer model (U-shaped Transformer).
        Source: https://github.com/zhanghanwei/Uformer
        """
        try:
            if Uformer is None:
                raise ImportError("uformer not installed. Uformer will not work.")
            model = Uformer(img_size=128, embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff', modulator=True, depths=[2,2,2,2,2,2,2], dd_in=3)
            weights_url = 'https://github.com/zhanghanwei/Uformer/releases/download/v0.0/uformer_b.pth'
            weights_path = 'uformer_b.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded Uformer model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Uformer: {e}")
            raise
    
    async def _load_nafnet(self):
        """
        Load NAFNet model (Nonlinear activation-free network).
        Source: https://github.com/megvii-research/NAFNet
        """
        try:
            if NAFNet is None:
                raise ImportError("nafnet not installed. NAFNet will not work.")
            model = NAFNet()
            weights_url = 'https://github.com/megvii-research/NAFNet/releases/download/v0.0/NAFNet.pth'
            weights_path = 'NAFNet.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded NAFNet model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load NAFNet: {e}")
            raise
    
    async def _load_swin2sr(self):
        """
        Load Swin2SR model (SwinV2-based SR).
        Source: https://github.com/JingyunLiang/Swin2SR
        """
        try:
            if Swin2SR is None:
                raise ImportError("swin2sr not installed. Swin2SR will not work.")
            model = Swin2SR()
            weights_url = 'https://github.com/JingyunLiang/Swin2SR/releases/download/v0.0/swin2sr.pth'
            weights_path = 'swin2sr.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded Swin2SR model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load Swin2SR: {e}")
            raise
    
    async def _load_ipt(self):
        """
        Load IPT model (Image Processing Transformer).
        Source: https://github.com/huawei-noah/Pretrained-IPT
        """
        try:
            if IPT is None:
                raise ImportError("ipt not installed. IPT will not work.")
            model = IPT()
            weights_url = 'https://github.com/huawei-noah/Pretrained-IPT/releases/download/v0.0/ipt.pth'
            weights_path = 'ipt.pth'
            if not Path(weights_path).exists():
                r = requests.get(weights_url)
                with open(weights_path, 'wb') as f:
                    f.write(r.content)
            model.load_state_dict(torch.load(weights_path), strict=True)
            model = model.to(self.device)
            model.eval()
            logger.info("Loaded IPT model.")
            return model
        except Exception as e:
            logger.error(f"Failed to load IPT: {e}")
            raise
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process super-resolution task using the selected model.
        task = {
            'image': <PIL.Image|np.ndarray|str>,
            'model': <model_name>,
            'output_path': <optional>
        }
        """
        try:
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            model_name = task.get('model', 'real_esrgan')
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not available.")
            x = self.transforms(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                y = self.models[model_name](x)
            output_image = self._tensor_to_image(y)
            output_path = None
            if 'output_path' in task:
                output_path = task['output_path']
                output_image.save(output_path)
            return {
                'status': 'success',
                'output_image': output_image,
                'output_path': output_path,
                'model_used': model_name
            }
        except Exception as e:
            logger.error(f"Super-resolution failed: {e}")
            raise
    
    def _tensor_to_image(self, x: torch.Tensor) -> Image.Image:
        x = x.squeeze(0).cpu()
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte()
        return Image.fromarray(x.permute(1, 2, 0).numpy())
    
    async def _cleanup(self) -> None:
        self.models.clear()
        torch.cuda.empty_cache()

    def prune_model(self, model: nn.Module, amount: float = 0.2) -> None:
        parameters = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters.append((module, "weight"))
        if parameters:
            prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)

# ----------------------
# Model Weights/Download Instructions
# ----------------------
"""
Real-ESRGAN: No manual download needed if using torch.hub. For custom weights, see https://github.com/xinntao/Real-ESRGAN#model-zoo
ESRGAN: https://github.com/xinntao/ESRGAN#model-zoo
BSRGAN: https://github.com/cszn/BSRGAN#pre-trained-models
RRDBNet: https://github.com/xinntao/ESRGAN#rrdb-models
SwinIR: https://github.com/JingyunLiang/SwinIR#pre-trained-models
Restormer: https://github.com/swz30/Restormer#pre-trained-models
Uformer: https://github.com/zhanghanwei/Uformer#pretrained-models
NAFNet: https://github.com/megvii-research/NAFNet#pretrained-models
Swin2SR: https://github.com/JingyunLiang/Swin2SR#pre-trained-models
IPT: https://github.com/huawei-noah/Pretrained-IPT#pre-trained-models
"""

# ----------------------
# Runtime Test Case Example
# ----------------------
if __name__ == "__main__":
    import asyncio
    test_image_path = "test.jpg"  # Provide a real image path
    output_path = "output_real_esrgan.jpg"
    async def test():
        agent = SuperResolutionAgent()
        await agent._initialize()
        result = await agent._process({
            'image': test_image_path,
            'model': 'real_esrgan',
            'output_path': output_path
        })
        print(f"Test result: {result['status']}, output saved to {result['output_path']}")
    asyncio.run(test()) 