"""Category-first Grounding DINO + SAM. No scene metadata or GT vocabulary input."""
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import unproject
from .task import COMMON_CATEGORIES


def color_scores(rgb, mask):
    """Weak, revisable appearance features; the VLM judges object parts and lighting."""
    pixels = np.asarray(rgb)[mask]
    if not len(pixels):
        return {}
    hsv = np.asarray(Image.fromarray(pixels.reshape(1, -1, 3)).convert('HSV'))[0]/255.0
    h, s, v = hsv.T
    rules = {'red': ((h < .055) | (h > .94)) & (s > .35) & (v > .25),
             'white': (s < .22) & (v > .62), 'black': v < .22,
             'gray': (s < .22) & (v >= .22) & (v <= .62),
             'brown': (h >= .035) & (h < .14) & (s >= .22) & (v < .72),
             'blue': (h > .50) & (h < .72) & (s > .25)}
    return {k: float(x.mean()) for k, x in rules.items()}


class GroundedSAM:
    def __init__(self, sam_checkpoint, dino_model='IDEA-Research/grounding-dino-tiny',
                 device='cuda:0', threshold=0.28):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from segment_anything import sam_model_registry, SamPredictor
        if not Path(sam_checkpoint).is_file():
            raise FileNotFoundError('SAM ViT-B checkpoint missing: ' + str(sam_checkpoint))
        self.torch, self.device, self.threshold = torch, device, threshold
        # Fail explicitly when cache is incomplete; no hidden large downloads during navigation.
        self.processor = AutoProcessor.from_pretrained(dino_model, local_files_only=True)
        # Transformers 4.41 tries compiling CUDA extensions even when disabled.
        # Use its supported PyTorch implementation; suppress only the eager build probe.
        from transformers.models.grounding_dino import modeling_grounding_dino as gd
        probe = gd.is_ninja_available
        try:
            gd.is_ninja_available = lambda: False
            self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_model, local_files_only=True, disable_custom_kernels=True).to(device).eval()
        finally:
            gd.is_ninja_available = probe
        self.sam = SamPredictor(sam_model_registry['vit_b'](checkpoint=str(sam_checkpoint)).to(device).eval())
        import transformers
        self.manifest = dict(detector=dino_model, revision=getattr(self.detector.config, '_commit_hash', None),
                             sam='vit_b', sam_checkpoint=str(sam_checkpoint),
                             torch=torch.__version__, transformers=transformers.__version__,
                             threshold=threshold, device=device, custom_cuda_kernels=False)

    def detect(self, observation, output, categories=COMMON_CATEGORIES):
        rgb = observation['rgb']
        self.sam.set_image(rgb)
        results = []
        # Separate category prompts avoid color conditioning and label-token ambiguity.
        for category in categories:
            inputs = self.processor(images=Image.fromarray(rgb), text=category+'.', return_tensors='pt').to(self.device)
            with self.torch.inference_mode():
                pred = self.detector(**inputs)
            boxes = self.processor.post_process_grounded_object_detection(
                pred, inputs.input_ids, box_threshold=self.threshold, text_threshold=.25,
                target_sizes=[rgb.shape[:2]])[0]
            from torchvision.ops import nms
            keep = nms(boxes['boxes'], boxes['scores'], .5)
            for index in keep.tolist():
                box = boxes['boxes'][index].detach().cpu().numpy()
                masks, scores, _ = self.sam.predict(box=box, multimask_output=True)
                mask = masks[int(np.argmax(scores))]
                points = unproject(observation['depth'], mask, observation['intrinsics'], observation['camera_to_world'])
                if points is None:
                    continue
                ys, xs = np.nonzero(mask)
                if len(xs) < 40:
                    continue
                stem = observation['id'] + '_%02d' % len(results)
                crop_path, mask_path = Path(output)/(stem+'_crop.jpg'), Path(output)/(stem+'_mask.png')
                # Masked crops prevent a neighboring red object from defining chair color.
                masked = np.where(mask[:, :, None], rgb, np.full_like(rgb, 127))
                Image.fromarray(masked[ys.min():ys.max()+1, xs.min():xs.max()+1]).save(crop_path)
                Image.fromarray(mask.astype('uint8')*255).save(mask_path)
                results.append(dict(category=category, confidence=float(boxes['scores'][index]),
                                    points=points, crop=str(crop_path), mask=str(mask_path),
                                    bbox_2d=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
                                    attributes={'color_scores': color_scores(rgb, mask)}))
        return results
