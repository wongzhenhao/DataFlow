import os
import json
from typing import Dict, Any, List, Optional

import pandas as pd

from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.core import OperatorABC
from dataflow.utils.storage import DataFlowStorage


@OPERATOR_REGISTRY.register()
class LineSeriesGenerator(OperatorABC):
    """
    使用本地 LineFormer Serving 批量为图片提取折线数据（line dataseries）。

    DataFrame 输入字段（可配置，默认如下）：
    - png_path_key: PNG图片路径字段，默认为 'png_path'（每行一张图）
    - output_dir_key: 输出目录；若为空则使用原图所在目录

    输出：
    - dataframe[output_key]: 该图的 line_dataseries（列表）
    - dataframe[lineformer_json_key]: LineFormer JSON文件路径（可选）
    同时根据参数选择性落盘 JSON 与可视化 PNG
    """

    def __init__(self, lf_serving):
        self.logger = get_logger()
        self.lf_serving = lf_serving

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量为图片提取 LineFormer 数据系列（线条坐标），以 DataFrame 驱动。\n"
                "输入字段：png_path_key(默认 'png_path')，每行一张PNG图。\n"
                "输出字段：output_key(默认 'line_series')为该图的线条数据。"
            )
        return (
            "Batch extract line series using LineFormer in a DataFrame-driven operator. One PNG per row."
        )

    def run(self,
            storage: DataFlowStorage,
            png_path_key: str = "png_path",
            output_dir_key: str = "output_dir",
            output_key: str = "line_series",
            lineformer_json_key: str = "lineformer_json_path",
            save_json: bool = True,
            save_vis: bool = False):
        self.logger.info("Running LineSeriesGenerator...")

        df: pd.DataFrame = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(df)}")

        # 使用传入的 lf_serving 或创建新实例
        if self.lf_serving is None:
            from dataflow.serving.api_lineformer_serving_local import APILineFormerServing_local
            serving = APILineFormerServing_local()
        else:
            serving = self.lf_serving

        # 结果容器
        line_series_list: List[Optional[List]] = [None] * len(df)
        json_path_list: List[Optional[str]] = [None] * len(df)

        # 启动 Serving
        serving.start_serving()

        try:
            # 收集所有有效的 PNG 路径
            batch_image_paths: List[str] = []
            row_indices: List[int] = []

            for idx, row in df.iterrows():
                png_path = row.get(png_path_key)
                if isinstance(png_path, str) and os.path.exists(png_path):
                    batch_image_paths.append(png_path)
                    row_indices.append(idx)
                else:
                    line_series_list[idx] = []
                    json_path_list[idx] = None

            if batch_image_paths:
                self.logger.info(f"Processing {len(batch_image_paths)} PNG images via LineFormer...")

                # 批量处理
                results = serving.extract_from_image_paths(
                    batch_image_paths,
                    return_image=save_vis,
                    save_json_dir=None,
                    save_vis_dir=None,
                    chunksize=4
                )

                # 分配结果回各行
                for idx_pos, (idx, png_path, result) in enumerate(zip(row_indices, batch_image_paths, results)):
                    row = df.loc[idx]
                    
                    if result.get("status") == "success":
                        line_series = result.get("line_dataseries", [])
                        line_series_list[idx] = line_series
                        
                        # 保存 JSON（可选）
                        if save_json:
                            out_dir = row.get(output_dir_key)
                            if not out_dir or not isinstance(out_dir, str):
                                out_dir = os.path.dirname(png_path)
                            os.makedirs(out_dir, exist_ok=True)
                            
                            fname = os.path.basename(png_path)
                            stem = os.path.splitext(fname)[0]
                            json_path = os.path.join(out_dir, f"{stem}_lineformer.json")
                            
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(line_series, jf, ensure_ascii=False, indent=2)
                            
                            json_path_list[idx] = json_path
                            self.logger.info(f"Row {idx}: Saved line series to {json_path}")
                        
                        # 保存可视化（可选）
                        if save_vis and result.get("visualized_base64"):
                            import base64
                            out_dir = row.get(output_dir_key) or os.path.dirname(png_path)
                            os.makedirs(out_dir, exist_ok=True)
                            fname = os.path.basename(png_path)
                            stem = os.path.splitext(fname)[0]
                            vis_path = os.path.join(out_dir, f"{stem}_lineformer_vis.png")
                            
                            with open(vis_path, "wb") as vf:
                                vf.write(base64.b64decode(result["visualized_base64"]))
                            
                            self.logger.info(f"Row {idx}: Saved visualization to {vis_path}")
                    else:
                        line_series_list[idx] = []
                        json_path_list[idx] = None
                        self.logger.warning(f"Row {idx}: Failed to process {png_path}: {result.get('error', 'Unknown error')}")

        finally:
            serving.stop_serving()

        # 回写 DataFrame
        df[output_key] = line_series_list
        if save_json:
            df[lineformer_json_key] = json_path_list
        
        output_file = storage.write(df)
        self.logger.info(f"LineSeriesGenerator completed. Output saved to {output_file}")
        return output_key


