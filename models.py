"""
Model loading/inference wrappers for Florence-2, Grounding DINO, and SAM 2.1.

Grounding DINO and SAM 2.1 are loaded from the ORIGINAL
github.com/IDEA-Research/Grounded-SAM-2 repo (cloned by setup_environment.sh
into config.GROUNDED_SAM2_DIR), per explicit instruction, rather than the HF
`transformers` ports. Florence-2 is unaffected -- it still loads via
`transformers`.

Design goals driven by the budget constraint:
  - Heavy imports (torch, transformers, groundingdino, sam2) happen INSIDE
    functions, not at module import time, so scripts running with --dry_run
    can import this file's type hints / constants without ever touching
    torch or the cloned repo.
  - Every loader returns a simple object; call free_model() on it when a
    pipeline stage is done to explicitly release GPU memory before the next
    stage loads its own model (requirement: never keep all three resident).
  - Loaders print the device + dtype they end up on so a bad CUDA setup is
    obvious immediately rather than silently falling back to slow CPU.
  - GroundingDinoWrapper.detect() and Sam2Wrapper.mask_from_box() keep the
    exact same call signature as the earlier `transformers`-based version,
    so 03/04_generate_grounded_masks.py need zero changes.
"""

from __future__ import annotations

import logging
from typing import Optional

import config


def _get_device_and_dtype():
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def free_model(*objs) -> None:
    """Explicitly drop references and clear the CUDA cache between stages."""
    import gc
    import torch

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _ensure_grounded_sam2_importable() -> None:
    """Safety net in case the editable installs from setup_environment.sh
    didn't register on this interpreter's path for some reason (e.g. a
    different venv got activated). Cheap to try; a clear ImportError with
    guidance still surfaces below if this doesn't fix it."""
    import sys

    for p in (str(config.GROUNDED_SAM2_DIR), str(config.GROUNDED_SAM2_DIR / "grounding_dino")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _require_file(path, what: str) -> None:
    from pathlib import Path

    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found at {path}. Did setup_environment.sh finish successfully? "
            f"Check its output for the Grounded-SAM-2 clone/install/download steps."
        )


# ---------------------------------------------------------------------------
# Florence-2
# ---------------------------------------------------------------------------
class Florence2Wrapper:
    def __init__(self, logger: Optional[logging.Logger] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        config.configure_hf_cache()
        self.device, self.dtype = _get_device_and_dtype()
        log = logger or logging.getLogger("models")
        log.info(f"Loading Florence-2 ({config.FLORENCE2_MODEL_ID}) on {self.device}/{self.dtype}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.FLORENCE2_MODEL_ID, torch_dtype=self.dtype, trust_remote_code=True
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(
            config.FLORENCE2_MODEL_ID, trust_remote_code=True
        )
        log.info("Florence-2 loaded.")

    def ground_phrase(self, image, prompt_text: str) -> dict:
        """Runs <CAPTION_TO_PHRASE_GROUNDING> against `prompt_text` and returns
        the raw parsed dict: {'bboxes': [[x1,y1,x2,y2], ...], 'labels': [str, ...]}
        """
        import torch

        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self.processor(text=task + prompt_text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype if v.dtype.is_floating_point else v.dtype)
                  for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task, image_size=(image.width, image.height)
        )
        return parsed.get(task, {"bboxes": [], "labels": []})


# ---------------------------------------------------------------------------
# Grounding DINO (original IDEA-Research/GroundingDINO, vendored inside
# Grounded-SAM-2). Uses whatever attention kernel got installed -- the
# compiled CUDA op if setup_environment.sh's build succeeded, or
# GroundingDINO's own built-in pure-PyTorch fallback (slower, same output)
# if it didn't. Either way this class's behavior is identical from here.
# ---------------------------------------------------------------------------
class GroundingDinoWrapper:
    def __init__(self, logger: Optional[logging.Logger] = None):
        _ensure_grounded_sam2_importable()
        _require_file(config.GROUNDING_DINO_CONFIG_PATH, "Grounding DINO config")
        _require_file(config.GROUNDING_DINO_CHECKPOINT_PATH, "Grounding DINO checkpoint")

        import torch
        try:
            from groundingdino.util.inference import load_model
            import groundingdino.datasets.transforms as gdino_T
        except ImportError as e:
            raise ImportError(
                "Failed to import `groundingdino`. This means the editable install of "
                "third_party/Grounded-SAM-2/grounding_dino did not complete -- re-check "
                "setup_environment.sh's [3/7] step output for a build error."
            ) from e

        self.device, _ = _get_device_and_dtype()
        log = logger or logging.getLogger("models")
        log.info(f"Loading Grounding DINO (original repo, SwinT-OGC) on {self.device}...")

        self.model = load_model(
            str(config.GROUNDING_DINO_CONFIG_PATH),
            str(config.GROUNDING_DINO_CHECKPOINT_PATH),
            device=self.device,
        )
        self.model.eval()
        self._transform = gdino_T.Compose([
            gdino_T.RandomResize([800], max_size=1333),
            gdino_T.ToTensor(),
            gdino_T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        log.info("Grounding DINO loaded.")

    def detect(self, image, text_query: str,
               box_threshold: float = None, text_threshold: float = None) -> list[dict]:
        """text_query must be lowercase, period-separated per Grounding DINO's
        expected prompt format, e.g. 'red hat.'. Returns a list of
        {'box': [x1,y1,x2,y2], 'score': float, 'label': str} in pixel coords."""
        from groundingdino.util.inference import predict
        from torchvision.ops import box_convert
        import torch

        box_threshold = box_threshold if box_threshold is not None else config.GROUNDING_DINO_BOX_THRESHOLD
        text_threshold = text_threshold if text_threshold is not None else config.GROUNDING_DINO_TEXT_THRESHOLD

        query = text_query.strip().lower()
        if not query.endswith("."):
            query += "."

        W, H = image.width, image.height
        image_tensor, _ = self._transform(image.convert("RGB"), None)

        boxes, scores, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )

        if boxes.numel() == 0:
            return []

        boxes_pixel = boxes * torch.tensor([W, H, W, H], dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes=boxes_pixel, in_fmt="cxcywh", out_fmt="xyxy").tolist()

        detections = []
        for box, score, phrase in zip(boxes_xyxy, scores.tolist(), phrases):
            detections.append({
                "box": [float(x) for x in box],
                "score": float(score),
                "label": phrase,
            })
        return detections


# ---------------------------------------------------------------------------
# SAM 2.1 (hiera-tiny, original facebookresearch/sam2 code vendored inside
# Grounded-SAM-2). SAM2's own CUDA post-processing op is best-effort by
# default in its setup.py (SAM2_BUILD_ALLOW_ERRORS=1) -- it degrades
# gracefully on its own if compilation fails, no fallback logic needed here.
# ---------------------------------------------------------------------------
class Sam2Wrapper:
    def __init__(self, logger: Optional[logging.Logger] = None):
        _ensure_grounded_sam2_importable()
        _require_file(config.SAM2_CHECKPOINT_PATH, "SAM 2.1 checkpoint")

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(
                "Failed to import `sam2`. This means the editable install of "
                "third_party/Grounded-SAM-2 (`pip install -e .`) did not complete -- "
                "re-check setup_environment.sh's [3/7] step output for a build error."
            ) from e

        self.device, _ = _get_device_and_dtype()
        log = logger or logging.getLogger("models")
        log.info(f"Loading SAM 2.1 hiera-tiny (original repo) on {self.device}...")

        sam2_model = build_sam2(
            config.SAM2_CONFIG_NAME, str(config.SAM2_CHECKPOINT_PATH), device=self.device
        )
        self.predictor = SAM2ImagePredictor(sam2_model)
        log.info("SAM 2.1 loaded.")

    def mask_from_box(self, image, box_xyxy: list[float]):
        """Returns a single-channel boolean numpy mask (H, W) for the given
        box prompt, picking SAM2's highest-IoU-predicted mask candidate."""
        import numpy as np
        import torch

        image_np = np.array(image.convert("RGB"))

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda" else _nullcontext()
        )
        with autocast_ctx:
            self.predictor.set_image(image_np)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box_xyxy),
                multimask_output=True,
            )

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)


def _nullcontext():
    import contextlib

    return contextlib.nullcontext()
