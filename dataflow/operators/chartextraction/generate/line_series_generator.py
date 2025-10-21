import os
import json
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

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
    - figure_info_key: 图表信息字段，默认为 'figure_info'（用于重绘）
    - parser_json_key: Parser JSON路径字段，默认为 'parser_json'（用于坐标映射）

    输出：
    - dataframe[output_key]: 该图的 line_dataseries（列表）
    - dataframe[lineformer_json_key]: LineFormer JSON文件路径（可选）
    - dataframe[replot_key]: 重绘图表路径（可选）
    
    功能：
    1. 提取线条数据（像素坐标）
    2. 保存 JSON 和可视化（可选）
    3. 坐标变换 + 重绘（可选）：通过 curve_fit 拟合像素到真实坐标的映射，生成物理坐标图表
    """

    def __init__(self, lf_serving):
        self.logger = get_logger()
        self.lf_serving = lf_serving

    @staticmethod
    def _linear_model(x, a, b):
        """线性模型：y = ax + b"""
        return a * x + b

    def _build_label_dict(self, parser_data: List) -> Dict[str, List[float]]:
        """
        从 parser_data 构建 label_dict
        label_dict[text] = [x1, y1, x2, y2] (bbox)
        """
        label_dict = {}
        for item in parser_data:
            if len(item) >= 2 and len(item[0]) >= 4 and len(item[1]) >= 1:
                text = item[1][0]
                bbox = item[0]  # [x1, y1, x2, y2]
                # 计算中心点作为 label 位置
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                label_dict[text] = [bbox[0], bbox[1], bbox[2], bbox[3], center_x, center_y]
        return label_dict

    def _fit_coordinate_transform(
        self, 
        figure_info: Dict[str, Any], 
        label_dict: Dict[str, List[float]]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        拟合像素坐标到真实坐标的转换参数
        返回: (x_params, y_params) 其中每个是 (a, b) 或 None
        """
        x_tick_labels = figure_info.get('x_tick_labels', [])
        y_tick_labels = figure_info.get('y_tick_labels', [])
        
        # 收集 x 轴映射点
        x_pixel = []
        x_real = []
        for x_label in x_tick_labels:
            if x_label in label_dict:
                try:
                    x_pixel.append(label_dict[x_label][4])  # center_x
                    x_real.append(float(x_label))
                except (ValueError, IndexError):
                    continue
        
        # 收集 y 轴映射点
        y_pixel = []
        y_real = []
        for y_label in y_tick_labels:
            if y_label in label_dict:
                try:
                    y_pixel.append(label_dict[y_label][5])  # center_y
                    y_real.append(float(y_label))
                except (ValueError, IndexError):
                    continue
        
        # 拟合 x 轴变换
        x_params = None
        if len(x_pixel) >= 2 and len(x_real) >= 2:
            try:
                params, _ = curve_fit(self._linear_model, x_pixel, x_real)
                x_params = tuple(params)
            except Exception as e:
                self.logger.warning(f"Failed to fit x-axis transform: {e}")
        
        # 拟合 y 轴变换
        y_params = None
        if len(y_pixel) >= 2 and len(y_real) >= 2:
            try:
                params, _ = curve_fit(self._linear_model, y_pixel, y_real)
                y_params = tuple(params)
            except Exception as e:
                self.logger.warning(f"Failed to fit y-axis transform: {e}")
        
        return x_params, y_params

    def _transform_line_series(
        self,
        line_series: List[List[Dict[str, float]]],
        x_params: Optional[Tuple[float, float]],
        y_params: Optional[Tuple[float, float]]
    ) -> List[List[Dict[str, float]]]:
        """
        将像素坐标的线条数据转换为真实坐标
        """
        transformed = []
        for line in line_series:
            new_line = []
            for pt in line:
                x = pt['x']
                y = pt['y']
                
                if x_params is not None:
                    x = self._linear_model(x, x_params[0], x_params[1])
                if y_params is not None:
                    y = self._linear_model(y, y_params[0], y_params[1])
                
                new_line.append({'x': float(x), 'y': float(y)})
            transformed.append(new_line)
        return transformed

    def _replot_figure(
        self,
        line_series: List[List[Dict[str, float]]],
        output_path: str,
        figure_info: Optional[Dict[str, Any]] = None,
        figsize: Tuple[float, float] = (8, 6),
        dpi: int = 300
    ):
        """
        根据线条数据重新绘制图表
        """
        plt.figure(figsize=figsize, dpi=dpi)
        
        for line in line_series:
            if len(line) < 2:
                continue
            x_data = [pt['x'] for pt in line]
            y_data = [pt['y'] for pt in line]
            plt.plot(x_data, y_data, linewidth=1.5)
        
        if figure_info:
            x_label = figure_info.get('x_label', '')
            y_label = figure_info.get('y_label', '')
            if x_label:
                plt.xlabel(x_label)
            if y_label:
                plt.ylabel(y_label)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close()

    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "批量为图片提取 LineFormer 数据系列（线条坐标），以 DataFrame 驱动。\n"
                "输入字段：png_path_key(默认 'png_path')，每行一张PNG图。\n"
                "输出字段：output_key(默认 'line_series')为该图的线条数据。\n"
                "可选功能：\n"
                "- save_json: 保存 LineFormer JSON 文件\n"
                "- save_vis: 保存 LineFormer 可视化图片\n"
                "- replot: 将像素坐标转换为真实物理坐标并重新绘图（需要 figure_info 和 parser_json）"
            )
        return (
            "Batch extract line series using LineFormer in a DataFrame-driven operator. "
            "One PNG per row. Optional replot feature to convert pixel coordinates to real coordinates."
        )

    def run(self,
            storage: DataFlowStorage,
            png_path_key: str = "png_path",
            output_dir_key: str = "output_dir",
            output_key: str = "line_series",
            lineformer_json_key: str = "lineformer_json_path",
            save_json: bool = True,
            save_vis: bool = False,
            replot: bool = True,
            replot_key: str = "replot_path",
            figure_info_key: str = "figure_info",
            parser_json_key: str = "parser_json"):
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
        replot_path_list: List[Optional[str]] = [None] * len(df)

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
                    replot_path_list[idx] = None

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
                        
                        # 准备输出目录
                        out_dir = row.get(output_dir_key)
                        if not out_dir or not isinstance(out_dir, str):
                            out_dir = os.path.dirname(png_path)
                        os.makedirs(out_dir, exist_ok=True)
                        
                        fname = os.path.basename(png_path)
                        stem = os.path.splitext(fname)[0]
                        
                        # 保存 JSON（可选）
                        if save_json:
                            json_path = os.path.join(out_dir, f"{stem}_lineformer.json")
                            
                            with open(json_path, "w", encoding="utf-8") as jf:
                                json.dump(line_series, jf, ensure_ascii=False, indent=2)
                            
                            json_path_list[idx] = json_path
                            self.logger.info(f"Row {idx}: Saved line series to {json_path}")
                        
                        # 保存可视化（可选）
                        if save_vis and result.get("visualized_base64"):
                            import base64
                            vis_path = os.path.join(out_dir, f"{stem}_result.png")
                            
                            with open(vis_path, "wb") as vf:
                                vf.write(base64.b64decode(result["visualized_base64"]))
                            
                            self.logger.info(f"Row {idx}: Saved visualization to {vis_path}")
                        
                        # 重绘功能（可选）
                        if replot:
                            try:
                                # 获取 figure_info
                                figure_info = row.get(figure_info_key)
                                if not isinstance(figure_info, dict):
                                    self.logger.warning(f"Row {idx}: No valid figure_info for replotting")
                                    replot_path_list[idx] = None
                                    continue
                                
                                # 获取 parser_json（如果有）
                                parser_json_path = row.get(parser_json_key)
                                label_dict = {}
                                
                                if parser_json_path and isinstance(parser_json_path, str) and os.path.exists(parser_json_path):
                                    with open(parser_json_path, 'r', encoding='utf-8') as f:
                                        parser_data = json.load(f)
                                    label_dict = self._build_label_dict(parser_data)
                                
                                # 拟合坐标变换参数
                                x_params, y_params = None, None
                                if label_dict:
                                    x_params, y_params = self._fit_coordinate_transform(figure_info, label_dict)
                                
                                # 如果没有有效的坐标变换，仍然可以绘制原始像素坐标
                                if x_params is None and y_params is None:
                                    self.logger.warning(f"Row {idx}: No coordinate transform available, plotting in pixel coordinates")
                                
                                # 转换坐标
                                transformed_series = self._transform_line_series(line_series, x_params, y_params)
                                
                                # 重绘图表
                                replot_path = os.path.join(out_dir, f"{stem}_replot.png")
                                self._replot_figure(transformed_series, replot_path, figure_info)
                                replot_path_list[idx] = replot_path
                                
                                self.logger.info(f"Row {idx}: Saved replot to {replot_path}")
                                
                            except Exception as e:
                                self.logger.error(f"Row {idx}: Failed to replot: {e}")
                                replot_path_list[idx] = None
                    else:
                        line_series_list[idx] = []
                        json_path_list[idx] = None
                        replot_path_list[idx] = None
                        self.logger.warning(f"Row {idx}: Failed to process {png_path}: {result.get('error', 'Unknown error')}")

        finally:
            serving.stop_serving()

        # 回写 DataFrame
        df[output_key] = line_series_list
        if save_json:
            df[lineformer_json_key] = json_path_list
        if replot:
            df[replot_key] = replot_path_list
        
        output_file = storage.write(df)
        self.logger.info(f"LineSeriesGenerator completed. Output saved to {output_file}")
        return output_key


