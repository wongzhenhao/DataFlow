import os
import sys
import json
import base64
import cv2
import time
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count
from pathlib import Path

from ..logger import get_logger

_g_initialized = False
_g_lineformer_root = None
_g_config = None
_g_checkpoint = None
_g_device = None
_g_padding_size = 40
_g_border_color = (255, 255, 255)
_g_infer = None
_g_line_utils = None


def _worker_init(lineformer_root: str,
                 config_path: str,
                 checkpoint_path: str,
                 device: str,
                 padding_size: int,
                 border_color: Tuple[int, int, int]):
    global _g_initialized, _g_lineformer_root, _g_config, _g_checkpoint, _g_device
    global _g_infer, _g_line_utils, _g_padding_size, _g_border_color

    if _g_initialized:
        return

    _g_lineformer_root = lineformer_root
    _g_config = config_path
    _g_checkpoint = checkpoint_path
    _g_device = device
    _g_padding_size = padding_size
    _g_border_color = border_color

    if _g_lineformer_root and _g_lineformer_root not in sys.path:
        sys.path.insert(0, _g_lineformer_root)

    import importlib
    _g_infer = importlib.import_module("infer")
    _g_line_utils = importlib.import_module("line_utils")

    _g_infer.load_model(_g_config, _g_checkpoint, _g_device)
    _g_initialized = True


def _process_image_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    res = {
        "image_path": task.get("image_path"),
        "status": "failed",
        "num_lines": 0,
        "line_dataseries": [],
        "visualized_base64": None,
        "json_path": None,
        "vis_path": None,
        "error": None,
    }

    try:
        img_path = task["image_path"]
        return_image = bool(task.get("return_image", False))
        save_json_dir = task.get("save_json_dir")
        save_vis_dir = task.get("save_vis_dir")

        img = cv2.imread(img_path)
        if img is None:
            res["error"] = "Failed to read image"
            return res

        img_padded = cv2.copyMakeBorder(
            img,
            _g_padding_size, _g_padding_size, _g_padding_size, _g_padding_size,
            borderType=cv2.BORDER_CONSTANT,
            value=_g_border_color
        )

        line_dataseries, _ = _g_infer.get_dataseries(
            img_padded, to_clean=False, return_masks=True
        )
        res["line_dataseries"] = line_dataseries
        res["num_lines"] = len(line_dataseries)
        res["status"] = "success"

        stem = Path(img_path).stem

        vis_img = None
        if return_image or save_vis_dir:
            vis_img = _g_line_utils.draw_lines(
                img_padded, _g_line_utils.points_to_array(line_dataseries)
            )

        if return_image and vis_img is not None:
            ok, buf = cv2.imencode(".png", vis_img)
            if ok:
                res["visualized_base64"] = base64.b64encode(buf).decode("utf-8")

        if save_vis_dir and vis_img is not None:
            os.makedirs(save_vis_dir, exist_ok=True)
            vis_path = os.path.join(save_vis_dir, f"{stem}_result.png")
            cv2.imwrite(vis_path, vis_img)
            res["vis_path"] = vis_path

        if save_json_dir:
            os.makedirs(save_json_dir, exist_ok=True)
            json_path = os.path.join(save_json_dir, f"{stem}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(line_dataseries, f, ensure_ascii=False, indent=2)
            res["json_path"] = json_path

        return res

    except Exception as e:
        res["error"] = str(e)
        return res


class APILineFormerServing_local():
    def __init__(
        self,
        lineformer_root: str = "/mnt/DataFlow/wongzhenhao/ChartExtract/figures_paser",
        config_path: str = "/mnt/DataFlow/wongzhenhao/lineextract_clean/core/lineformer_swin_t_config.py",
        checkpoint_path: str = "/mnt/DataFlow/wongzhenhao/lineextract_standalone/weights/iter_3000.pth",
        device: str = "cpu",
        padding_size: int = 40,
        border_color: Tuple[int, int, int] = (255, 255, 255),
        num_workers: Optional[int] = None,
        timeout: int = 1800
    ):
        self.lineformer_root = lineformer_root
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.padding_size = padding_size
        self.border_color = border_color
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.timeout = timeout

        self._pool: Optional[Pool] = None
        self.logger = get_logger()

    def start_serving(self) -> None:
        if self._pool is not None:
            return
        self.logger.info(
            f"Starting LineFormer local serving with {self.num_workers} workers (device={self.device})"
        )
        self._pool = Pool(
            processes=self.num_workers,
            initializer=_worker_init,
            initargs=(
                self.lineformer_root,
                self.config_path,
                self.checkpoint_path,
                self.device,
                self.padding_size,
                self.border_color,
            ),
        )
        self.logger.success("LineFormer workers initialized.")

    def stop_serving(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            finally:
                self._pool = None
                self.logger.info("LineFormer serving stopped.")

    def extract_from_image_paths(
        self,
        image_paths: List[str],
        *,
        return_image: bool = False,
        save_json_dir: Optional[str] = None,
        save_vis_dir: Optional[str] = None,
        chunksize: int = 4
    ) -> List[Dict[str, Any]]:
        if not image_paths:
            return []

        if self._pool is None:
            self.start_serving()

        tasks = [{
            "image_path": p,
            "return_image": return_image,
            "save_json_dir": save_json_dir,
            "save_vis_dir": save_vis_dir,
        } for p in image_paths]

        results: List[Dict[str, Any]] = []
        t0 = time.time()
        try:
            for out in self._pool.imap_unordered(_process_image_worker, tasks, chunksize=chunksize):
                results.append(out)
        except Exception as e:
            self.logger.error(f"Multiprocess execution failed: {e}")
            raise
        finally:
            self.logger.info(
                f"Processed {len(image_paths)} images in {time.time() - t0:.2f}s"
            )

        return results


