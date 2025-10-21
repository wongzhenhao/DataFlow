import pandas as pd
import os
import json
from typing import Dict, Any, List, Tuple
from dataflow.utils.chartextraction.get_figure import batch_extract_figures_and_components
from dataflow.utils.chartextraction.extract_figure_info import extract_figure_components
from dataflow.utils.registry import OPERATOR_REGISTRY
from dataflow import get_logger
from dataflow.prompts.chartextraction import ChartInfoExtractionPrompt
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OperatorABC
from dataflow.core import LLMServingABC
from dataflow.core.prompt import prompt_restrict

@prompt_restrict(
    ChartInfoExtractionPrompt
)
@OPERATOR_REGISTRY.register()
class FigureInfoGenerator(OperatorABC):
    def __init__(self, vlm_serving: LLMServingABC):
        self.logger = get_logger()
        self.vlm_serving = vlm_serving
        self.prompt_template = ChartInfoExtractionPrompt()
        # 从 prompt 模板获取 JSON schema（用于 OpenAI API Structured Outputs）
        self.json_schema = self.prompt_template.get_json_schema()
        
    
    @staticmethod
    def get_desc(lang: str = "zh"):
        if lang == "zh":
            return (
                "从PDF文档中提取图表并使用VLM生成结构化图表信息。\n"
                "输入参数：\n"
                "- vlm_serving：VLM服务对象（可选），用于图表信息提取\n"
                "- input_pdf_key：PDF文件路径字段名，默认为'pdf_path'\n"
                "- parser_key：UniParser JSON文件路径字段名，默认为'parser_json'\n"
                "- output_dir_key：输出目录字段名，默认为'output_dir'（可选）\n"
                "- output_key：图表信息输出字段名，默认为'figure_info'\n"
                "- expand_rows：是否展开每张图为一行，默认为True\n"
                "输出参数：\n"
                "- expand_rows=True时：每张PNG图作为一行，包含pdf_path、png_path、json_path、figure_filename、figure_info等字段\n"
                "- expand_rows=False时：每个PDF一行，figure_info为{文件名: 图表信息}的字典\n"
                "- 提取的PNG文件和对应的JSON文件保存在output_dir中\n"
                "注意：只处理 _chart.png 图片（包括所有子chart），不处理 figure、caption、legend 等其他组件"
            )
        elif lang == "en":
            return (
                "Extract figures from PDF documents and generate structured chart information using VLM.\n"
                "Input Parameters:\n"
                "- vlm_serving: VLM serving object (optional) for chart information extraction\n"
                "- input_pdf_key: Field name for PDF file path, default is 'pdf_path'\n"
                "- parser_key: Field name for UniParser JSON file path, default is 'parser_json'\n"
                "- output_dir_key: Field name for output directory, default is 'output_dir' (optional)\n"
                "- output_key: Field name for figure info output, default is 'figure_info'\n"
                "- expand_rows: Whether to expand each figure as a row, default is True\n\n"
                "Output Parameters:\n"
                "- expand_rows=True: Each PNG as a row with pdf_path, png_path, json_path, figure_filename, figure_info fields\n"
                "- expand_rows=False: One row per PDF, figure_info as {filename: chart_info} dict\n"
                "- Extracted PNG files and corresponding JSON files saved in output_dir\n"
                "Note: Only processes _chart.png images (including sub-charts), excluding figure, caption, legend components"
            )
        else:
            return (
                "FigureInfoGenerator extracts figures from PDFs and generates structured chart information."
            )

    def _parse_chart_info(self, response: str) -> Dict[str, Any]:
        """
        解析 VLM 响应的 JSON 格式图表信息
        
        Args:
            response: VLM 返回的响应文本
            
        Returns:
            解析后的图表信息字典
        """
        try:
            # 移除可能的 markdown 代码块标记
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # 解析 JSON
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Response text: {response_text[:500]}...")
            return self._get_empty_chart_info()
        except Exception as e:
            self.logger.error(f"Error parsing chart info: {str(e)}")
            return self._get_empty_chart_info()
    
    def _get_empty_chart_info(self) -> Dict[str, Any]:
        """返回空的图表信息"""
        return {
            "x_label": "",
            "y_label": "",
            "x_tick_labels": [],
            "y_tick_labels": [],
            "legend_names": [],
            "legend_shapes": [],
            "legend_colors": [],
            "text": [],
            "figure_type": "",
        }

    def run(self, storage: DataFlowStorage, 
            input_pdf_key: str = "pdf_path", 
            parser_key: str = "parser_json", 
            output_dir_key: str = "output_dir",
            output_key: str = "figure_info",
            expand_rows: bool = True):
        """
        Args:
            expand_rows: 如果为True，每张PNG作为一行输出（复制原PDF元数据）；
                        如果为False，保持原有行为（每个PDF一行，figure_info为字典）
        """
        self.input_pdf_key, self.parser_key, self.output_dir_key, self.output_key = input_pdf_key, parser_key, output_dir_key, output_key
        self.logger.info("Running FigureInfoGenerator...")

        # Load the raw dataframe from the input file
        dataframe = storage.read('dataframe')
        self.logger.info(f"Loading, number of rows: {len(dataframe)}")
        
        # Step 1: 提取所有 PDF 的图表到 PNG
        png_metadata_list = []  # 存储 (row_idx, row_data, png_path, output_dir, filename) 的元组
        figure_info_list = [None] * len(dataframe)  # 预分配结果列表（用于 expand_rows=False）
        
        for idx, row in dataframe.iterrows():
            try:
                pdf_path = row.get(input_pdf_key)
                parser_json_path = row.get(parser_key)
                output_dir = row.get(output_dir_key)
                
                if not pdf_path or not os.path.exists(pdf_path):
                    self.logger.warning(f"Row {idx}: PDF file not found: {pdf_path}")
                    continue
                
                if not parser_json_path or not os.path.exists(parser_json_path):
                    self.logger.warning(f"Row {idx}: Parser JSON file not found: {parser_json_path}")
                    continue
                
                # Create output directory if not specified
                if not output_dir:
                    doi = os.path.splitext(os.path.basename(pdf_path))[0]
                    output_dir = os.path.join(storage.get_workspace(), "extracted_figures", doi)
                
                os.makedirs(output_dir, exist_ok=True)
                
                # Load parser JSON
                with open(parser_json_path, 'r', encoding='utf-8') as f:
                    parser_json = json.load(f)
                
                # Extract figures with components from parser result
                figure_list = extract_figure_components(parser_json)
                
                # Extract figures and components from PDF
                self.logger.info(f"Row {idx}: Extracting figures from {pdf_path}")
                batch_extract_figures_and_components(
                    pdf_path=pdf_path, 
                    output_dir=output_dir, 
                    figures_info=figure_list
                )
                
                # 收集所有提取的 PNG 文件路径（只收集 chart 图，包括所有子chart）
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        # 只处理 chart 图片（包括所有子chart），排除 figure、caption、legend 等
                        if file.endswith(".png") and "_chart.png" in file:
                            pic_path = os.path.join(root, file)
                            # 保存完整的行数据用于后续展开
                            png_metadata_list.append((idx, row.to_dict(), pic_path, output_dir, file))
                
                # 初始化该行的结果为空字典（用于 expand_rows=False）
                figure_info_list[idx] = {}
                
            except Exception as e:
                self.logger.error(f"Row {idx}: Error processing row: {str(e)}")
                continue
        
        # Step 2: 批量调用 VLM 处理所有图片
        png_results = {}  # {(row_idx, filename): (chart_info, json_path, png_path)}
        
        if png_metadata_list:
            self.logger.info(f"Processing {len(png_metadata_list)} PNG files with VLM...")
            
            # 准备所有图片路径
            image_paths = [metadata[2] for metadata in png_metadata_list]
            
            # 批量调用 VLM
            try:
                llm_outputs = self.vlm_serving.generate_from_input(image_paths, self.prompt_template.build_prompt(), json_schema = self.json_schema)
                
                # Step 3: 解析结果并保存
                for (row_idx, row_data, pic_path, output_dir, filename), response in zip(png_metadata_list, llm_outputs):
                    try:
                        # 解析 JSON 响应
                        chart_info = self._parse_chart_info(response)
                        
                        # 保存 JSON 文件
                        json_name = filename.replace(".png", ".json")
                        json_path = os.path.join(output_dir, json_name)
                        with open(json_path, 'w', encoding='utf-8') as jf:
                            json.dump(chart_info, jf, ensure_ascii=False, indent=4)
                        
                        # 保存结果
                        png_results[(row_idx, filename)] = (chart_info, json_path, pic_path)
                        
                        # 同时填充到 figure_info_list（用于 expand_rows=False）
                        if figure_info_list[row_idx] is not None:
                            figure_info_list[row_idx][filename] = chart_info
                        
                        self.logger.info(f"Row {row_idx}: Generated figure info for {filename}")
                        
                    except Exception as e:
                        self.logger.error(f"Row {row_idx}: Error processing {pic_path}: {str(e)}")
                        chart_info = self._get_empty_chart_info()
                        json_path = os.path.join(output_dir, filename.replace(".png", ".json"))
                        png_results[(row_idx, filename)] = (chart_info, json_path, pic_path)
                        if figure_info_list[row_idx] is not None:
                            figure_info_list[row_idx][filename] = chart_info
                
            except Exception as e:
                self.logger.error(f"Error in batch VLM processing: {str(e)}")
                # 如果批量处理失败，填充空结果
                for row_idx, row_data, pic_path, output_dir, filename in png_metadata_list:
                    chart_info = self._get_empty_chart_info()
                    json_path = os.path.join(output_dir, filename.replace(".png", ".json"))
                    png_results[(row_idx, filename)] = (chart_info, json_path, pic_path)
                    if figure_info_list[row_idx] is not None:
                        figure_info_list[row_idx][filename] = chart_info
        
        # Step 4: 构建输出 DataFrame
        if expand_rows:
            # 新模式：每张图一行（展开行）
            new_rows = []
            for (row_idx, row_data, pic_path, output_dir, filename) in png_metadata_list:
                chart_info, json_path, png_path = png_results.get((row_idx, filename), (self._get_empty_chart_info(), "", pic_path))
                
                # 复制原行数据
                new_row = row_data.copy()
                # 添加新字段
                new_row['png_path'] = png_path
                new_row['json_path'] = json_path
                new_row['figure_filename'] = filename
                new_row[output_key] = chart_info
                new_rows.append(new_row)
            
            output_df = pd.DataFrame(new_rows)
            self.logger.info(f"Expanded {len(dataframe)} PDFs to {len(output_df)} figures")
        else:
            # 旧模式：每个PDF一行，figure_info为字典
            dataframe[output_key] = figure_info_list
            output_df = dataframe
        
        # Save the updated dataframe to the output file
        output_file = storage.write(output_df)
        self.logger.info(f"FigureInfoGenerator completed. Output saved to {output_file}")
        return output_key
