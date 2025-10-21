"""
图表提取 Pipeline 示例
从 PDF 中提取图表并生成结构化信息

使用方式类似 prompted_vqa_generator:
1. 在 __init__ 中配置 VLM serving
2. 传入 FigureInfoGenerator
3. 批量处理所有图片
"""
from dataflow.operators.chartextraction import FigureInfoGenerator, LineSeriesGenerator
from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
from dataflow.utils.storage import FileStorage
from dataflow.serving.api_lineformer_serving_local import APILineFormerServing_local



class ChartExtractionPipeline:
    def __init__(self):
        """初始化图表提取 Pipeline"""
        
        # 配置 VLM serving
        self.vlm_serving = APIVLMServing_openai(
            model_name="gpt-4o-mini",  # 或 "gpt-4o" 等其他模型
            api_url="http://123.129.219.111:3000/v1",
            key_name_of_api_key="DF_API_KEY",
            max_workers=5,  # 并发数
            timeout=1800
        )
        # 配置 LineFormer 本地多进程 serving（方案二）
        self.lineformer_serving = APILineFormerServing_local(
            device="cpu",           # 如需 GPU 切换为 "cuda"
            num_workers=2,           # 每进程各自加载一份模型
            padding_size=40
        )
        
        # 配置存储
        self.storage = FileStorage(
            first_entry_file_name="/mnt/DataFlow/wongzhenhao/DataFlow/dataflow/example/ChartExtractionePipeline/example.jsonl",
            cache_path="./cache",
            file_name_prefix="chart_extraction",
            cache_type="json",
        )
        
        # 创建图表信息生成器（传入 VLM serving）
        self.figure_generator = FigureInfoGenerator(vlm_serving=self.vlm_serving)
        # 创建线条数据生成器（内部调用 lineformer_serving）
        self.line_series_generator = LineSeriesGenerator(lf_serving=self.lineformer_serving)
    
    def forward(self):
        """执行图表提取流程"""
        # Step 1: 提取图表并生成结构化信息（每张图一行）
        self.figure_generator.run(
            storage=self.storage.step(),
            input_pdf_key="pdf_path",      # PDF 路径字段名
            parser_key="parser_json",      # UniParser JSON 路径字段名
            output_dir_key="output_dir",   # 输出目录字段名（可选）
            output_key="figure_info",      # 输出字段名
            expand_rows=True               # 展开每张图为一行
        )

        # Step 2: 为每张图提取线条数据并重绘
        self.line_series_generator.run(
            storage=self.storage.step(),
            png_path_key='png_path',       # PNG路径字段名
            output_dir_key='output_dir',   # 输出目录字段名
            output_key='line_series',      # 输出字段名
            lineformer_json_key='lineformer_json_path',  # LineFormer JSON路径字段名
            save_json=True,                # 保存JSON
            save_vis=True,                #  保存可视化
            replot=True,                   # 启用重绘功能
            replot_key='replot_path',      # 重绘图表路径字段名
            figure_info_key='figure_info', # 图表信息字段名
            parser_json_key='parser_json'  # Parser JSON路径字段名
        )


if __name__ == "__main__":
    # 使用方式 1: 使用 Pipeline
    pipeline = ChartExtractionPipeline()
    pipeline.forward()
    
    # 使用方式 2: 直接使用（适合更灵活的配置）
    # from dataflow.operators.chartextraction.generate import FigureInfoGenerator
    # from dataflow.serving.api_vlm_serving_openai import APIVLMServing_openai
    # from dataflow.utils.storage import FileStorage
    # 
    # vlm = APIVLMServing_openai(model_name="gpt-4o-mini")
    # storage = FileStorage(first_entry_file_name="data.json")
    # generator = FigureInfoGenerator()
    # generator.run(storage.step(), vlm_serving=vlm)
