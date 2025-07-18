import os
import sys
import main as gr
import json
import csv
import logging
import datetime
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入doc_merge模块
from examples.doc_merge.doc_merge import (
    DocMergePrompter, DocMergeParser, 
    direct_method, cot, tot, got, got2
)
from graph_of_thoughts import controller, language_models, operations

# 配置日志
logging.basicConfig(level=logging.INFO)

def load_sample_documents(sample_id: int = 0) -> List[str]:
    """加载示例文档"""
    try:
        data_path = os.path.join("examples", "doc_merge", "documents.csv")
        documents = []
        with open(data_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for i, row in enumerate(reader):
                if i == sample_id:
                    # 文档从第3列开始（document1, document2, document3, document4）
                    documents = row[2:6]  # 获取4个文档
                    break
        return documents
    except Exception as e:
        logging.error(f"加载文档失败: {e}")
        return []

def get_available_samples() -> List[int]:
    """获取可用的样本ID"""
    try:
        data_path = os.path.join("examples", "doc_merge", "documents.csv")
        samples = []
        with open(data_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for i, row in enumerate(reader):
                samples.append(i)
        return samples[:10]  # 只返回前10个样本
    except Exception as e:
        logging.error(f"获取样本列表失败: {e}")
        return [0]

def get_method_map():
    """获取方法映射"""
    return {
        "IO": direct_method,
        "CoT": cot, 
        "ToT": tot,
        "GoT": got,
        "GoT2": got2
    }

def get_prompt_content(method_name: str, num_documents: int = 4):
    """
    获取对应方法的提示词内容
    
    :param method_name: 合并方法名称
    :type method_name: str
    :param num_documents: 文档数量
    :type num_documents: int
    :return: 提示词内容
    :rtype: str
    """
    try:
        from examples.doc_merge.doc_merge import DocMergePrompter
        prompter = DocMergePrompter()
        
        method_lower = method_name.lower()
        
        if method_lower == "direct_method":
            return prompter.merge_doc_prompt_start.format(num=num_documents)
        elif method_lower == "cot":
            return prompter.merge_doc_prompt_cot_start.format(num=num_documents)
        elif method_lower == "tot":
            return prompter.improve_summary_prompt_start.format(num=num_documents)
        elif method_lower == "got":
            return prompter.aggregate_full_prompt_base.format(num_ndas=num_documents, num_ndas_summary=3)
        elif method_lower == "got2":
            return prompter.aggregate_sub_prompt_base.format(num_ndas=num_documents)
        else:
            return f"使用 {method_name} 方法合并 {num_documents} 个文档"
    except Exception as e:
        return f"获取提示词失败: {str(e)}"

def run_doc_merge(method_name: str, sample_id: str, custom_docs: str) -> tuple:
    """
    使用Graph of Thoughts方法合并文档
    
    :param method_name: 合并方法名称
    :param sample_id: 样本ID
    :param custom_docs: 自定义文档
    :return: (合并结果, 提示词内容)
    :rtype: tuple
    """
    try:
        # 确定使用哪个文档源
        if custom_docs.strip():
            # 使用自定义文档
            try:
                # 尝试解析JSON格式
                if custom_docs.strip().startswith('['):
                    documents = json.loads(custom_docs)
                else:
                    # 按行分割
                    documents = [doc.strip() for doc in custom_docs.split('\n') if doc.strip()]
                
                if len(documents) < 2:
                    return "错误: 需要至少2个文档进行合并", "无提示词（输入错误）"
            except Exception as e:
                return f"错误: 解析自定义文档失败 - {str(e)}", "无提示词（解析错误）"
        else:
            # 使用示例文档
            try:
                # 从 "样本 0" 格式中提取数字
                sample_num = int(sample_id.split()[-1])
                documents = load_sample_documents(sample_num)
                if not documents:
                    return "错误: 无法加载示例文档", "无提示词（加载失败）"
            except (ValueError, IndexError):
                return "错误: 无效的样本ID格式", "无提示词（格式错误）"
        
        # 检查配置文件
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
            return error_msg, "无提示词（配置错误）"
        
        # 获取方法
        method_map = get_method_map()
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, GoT, GoT2"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        
        # 创建语言模型
        lm = language_models.ChatGPT(
            config_path,
            model_name="deepseek",
            cache=True,
        )
        
        # 重置成本计数器
        lm.cost = 0.0
        
        # 创建操作图
        operations_graph = method()
        
        # 创建控制器
        executor = controller.Controller(
            lm,
            operations_graph,
            DocMergePrompter(),
            DocMergeParser(),
            {
                "documents": documents,
                "current": "",
                "parts": set(),
                "method": method_name.lower(),
            },
        )
        
        # 执行
        executor.run()
        
        # 获取结果
        final_thoughts = executor.get_final_thoughts()
        if final_thoughts and len(final_thoughts) > 0 and len(final_thoughts[0]) > 0:
            result = final_thoughts[0][0].state.get("current", "无结果")
        else:
            result = "无结果"
        
        # 计算成本
        cost = lm.cost
        
        # 获取提示词内容
        prompt_content = get_prompt_content(method_name, len(documents))
        
        # 格式化输出
        output = f"""文档合并结果:

使用方法: {method_name}
文档数量: {len(documents)}
使用成本: ${cost:.6f}
执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型: deepseek
缓存: 启用

合并后的文档:
{result}

详细信息:
- 输入文档长度: {[len(doc) for doc in documents]}
- 输出文档长度: {len(result)}
- 平均文档长度: {sum(len(doc) for doc in documents) / len(documents):.0f} 字符
- 压缩比: {len(result) / sum(len(doc) for doc in documents) * 100:.1f}%"""
        
        return output, prompt_content
        
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"

# 创建Gradio界面
with gr.Blocks(title="Graph of Thoughts 文档合并演示") as demo:
    gr.Markdown("# Graph of Thoughts 文档合并演示")
    gr.Markdown("使用不同的思维方法合并多个NDA文档，并显示使用的提示词")
    
    with gr.Row():
        # 左上：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 输入区域")
            sample_dropdown = gr.Dropdown(
                choices=[f"样本 {i}" for i in get_available_samples()],
                value="样本 0",
                label="选择示例文档",
                info="选择预定义的示例文档集"
            )
            method_dropdown = gr.Dropdown(
                choices=["IO", "CoT", "ToT", "GoT", "GoT2"], 
                value="IO", 
                label="选择合并方法",
                info="IO: 直接合并, CoT: 思维链, ToT: 思维树, GoT: 思维图, GoT2: 改进思维图"
            )
            custom_docs_textbox = gr.Textbox(
                lines=8,
                label="自定义文档（可选）",
                placeholder="输入多个文档，每行一个文档，或使用JSON格式的数组。留空则使用示例文档。",
                info="支持文本格式（每行一个文档）或JSON格式"
            )
            run_button = gr.Button("开始合并", variant="primary")
        
        # 右上：结果区域
        with gr.Column(scale=1):
            gr.Markdown("### 合并结果")
            result_output = gr.Textbox(
                label="合并结果",
                lines=12,
                max_lines=15,
                interactive=False
            )
    
    with gr.Row():
        # 左下：快速示例区域
        with gr.Column(scale=1):
            gr.Markdown("### 快速示例")
            gr.Examples(
                examples=[
                    ["样本 0", "IO", ""],
                    ["样本 1", "CoT", ""],
                    ["样本 2", "ToT", ""],
                    ["样本 3", "GoT", ""],
                    ["样本 4", "GoT2", ""],
                ],
                inputs=[sample_dropdown, method_dropdown, custom_docs_textbox],
                label="点击示例快速填充"
            )
        
        # 右下：提示词区域
        with gr.Column(scale=1):
            gr.Markdown("### 使用的提示词")
            prompt_output = gr.Textbox(
                label="提示词内容",
                lines=12,
                max_lines=15,
                interactive=False
            )
    
    # 连接事件
    run_button.click(
        fn=run_doc_merge,
        inputs=[method_dropdown, sample_dropdown, custom_docs_textbox],
        outputs=[result_output, prompt_output]
    )

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("启动Graph of Thoughts文档合并演示...")
    print("请确保config.json文件存在并包含正确的API配置")
    
    demo.launch(
        server_port=7863, 
        share=False,
        show_error=True
    ) 