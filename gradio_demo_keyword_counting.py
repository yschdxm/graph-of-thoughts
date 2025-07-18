# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import main as gr
import json
import os
import sys
import logging
import datetime
from typing import Dict, List, Callable, Union

# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入Graph of Thoughts相关模块
from graph_of_thoughts import controller, language_models, operations, prompter, parser

# 导入keyword_counting模块
try:
    from examples.keyword_counting.keyword_counting import KeywordCountingPrompter, KeywordCountingParser, direct_method, cot, tot, tot2, got4, got8, gotx
except ImportError:
    # 如果从根目录运行，尝试直接导入
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.keyword_counting.keyword_counting import KeywordCountingPrompter, KeywordCountingParser, direct_method, cot, tot, tot2, got4, got8, gotx

def get_keyword_counting_modules():
    """
    获取keyword_counting模块
    
    :return: 对应的keyword_counting模块
    :rtype: tuple
    """
    return (KeywordCountingPrompter, KeywordCountingParser, 
            {"IO": direct_method, "CoT": cot, "ToT": tot, "ToT2": tot2, "GoT4": got4, "GoT8": got8, "GoTx": gotx})

def get_prompt_content(method_name):
    """
    获取对应方法的提示词内容
    
    :param method_name: keyword_counting方法名称
    :type method_name: str
    :return: 提示词内容
    :rtype: str
    """
    prompter = KeywordCountingPrompter()
    
    method_lower = method_name.lower()
    
    if method_lower == "direct_method":
        return prompter.count_prompt
    elif method_lower == "cot":
        return prompter.count_prompt_cot
    elif method_lower == "tot":
        return prompter.tot_improve_prompt
    elif method_lower == "tot2":
        return prompter.tot_improve_prompt
    elif method_lower == "got4":
        return prompter.got_split_prompt
    elif method_lower == "got8":
        return prompter.got_split_prompt2
    elif method_lower == "gotx":
        return prompter.got_split_prompt3
    else:
        return "不支持的方法"

def run_keyword_counting_with_got(input_text: str, method_name: str) -> tuple:
    """
    使用Graph of Thoughts方法计算文本中国家名称的频率
    
    :param input_text: 输入的文本
    :type input_text: str
    :param method_name: keyword_counting方法名称
    :type method_name: str
    :return: (结果, 提示词内容)
    :rtype: tuple
    """
    try:
        # 验证输入
        if not input_text.strip():
            error_msg = "错误：请输入文本内容。"
            return error_msg, "无提示词（输入错误）"
        
        # 获取对应的keyword_counting模块
        try:
            prompter_class, parser_class, method_map = get_keyword_counting_modules()
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
        
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT4, GoT8, GoTx"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        
        # 获取提示词内容
        prompt_content = get_prompt_content(method_name)
        
        # 创建语言模型
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if not os.path.exists(config_path):
            # 尝试使用模板配置文件
            template_config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts", "language_models", "config_template.json")
            if os.path.exists(template_config_path):
                error_msg = "错误：请复制 config_template.json 为 config.json 并配置您的API密钥。"
                return error_msg, "无提示词（配置错误）"
            else:
                error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
                return error_msg, "无提示词（配置错误）"
            
        lm = language_models.ChatGPT(
            config_path,
            model_name="deepseek",
            cache=True,
        )
        
        # 创建操作图
        # 所有方法都需要all_potential_countries参数
        operations_graph = method([])  # 传入空列表作为all_potential_countries
        
        # 创建执行器
        executor = controller.Controller(
            lm,
            operations_graph,
            prompter_class(),
            parser_class(),
            {
                "original": input_text,
                "ground_truth": "{}",  # 空的地面真值，因为我们只是演示
                "current": "",
                "phase": 0,
                "method": method_name.lower(),
            },
        )
        
        # 运行keyword_counting计算
        executor.run()
        
        # 获取最终结果
        final_thoughts = executor.get_final_thoughts()
        if final_thoughts and final_thoughts[0]:
            result = final_thoughts[0][0].state.get("current", "")
        else:
            result = "未获得结果"
        
        # 计算成本
        cost = lm.cost
        
        # 格式化输出
        output = f"""关键词计数结果:

输入文本: {input_text[:200]}{'...' if len(input_text) > 200 else ''}
计数结果: {result}
使用成本: ${cost:.6f}
使用方法: {method_name}
执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

详细信息:
- 输入长度: {len(input_text)} 个字符
- 模型: deepseek
- 缓存: 启用
- 方法类型: {'直接输入输出' if method_name == 'IO' else '思维链' if method_name == 'CoT' else '思维树' if method_name in ['ToT', 'ToT2'] else '思维图'}"""
        
        return output, prompt_content
        
    except ValueError as e:
        error_msg = f"输入格式错误: {str(e)}。"
        return error_msg, "无提示词（格式错误）"
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"



# 创建Gradio界面
with gr.Blocks(title="Graph of Thoughts 关键词计数演示") as demo:
    gr.Markdown("# Graph of Thoughts 关键词计数演示")
    gr.Markdown("使用不同的思维方法计算文本中国家名称的出现频率，并显示使用的提示词")
    
    with gr.Row():
        # 左上：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 输入区域")
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="例如: The music of Spain and the history of Spain deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
                info="请输入包含国家名称的文本",
                lines=8
            )
            method_dropdown = gr.Dropdown(
                choices=["IO", "CoT", "ToT", "ToT2", "GoT4", "GoT8", "GoTx"], 
                value="IO", 
                label="选择计数方法",
                info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT4/8/x: 思维图"
            )
            run_button = gr.Button("开始计数", variant="primary")
        
        # 右上：结果区域
        with gr.Column(scale=1):
            gr.Markdown("### 计数结果")
            result_output = gr.Textbox(
                label="计数结果",
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
                    ["The music of Spain and the history of Spain deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.", "IO"],
                    ["Alexandra boarded the first flight of her grand journey, starting from Canada. With a globe-trotting itinerary in hand, she was filled with excitement. Her first stop was Mexico, where she marveled at the Mayan ruins. From there, she explored the rainforests of Brazil and danced the tango in Argentina.", "CoT"],
                    ["The adventure led him to the peaks of Peru where he trekked to see the mysteries of Machu Picchu. He then headed to Chile to gaze at the vastness of the Atacama Desert. A quick detour to Uruguay and Paraguay allowed him to experience the vibrancy of the local cultures before returning back to Canada through Peru, Brazil and Mexico.", "ToT"],
                    ["Journeying westward, she admired the art in Italy and sipped coffee in France. The music of Spain and the history of Greece deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia. Italy, Norway, Sweden and Germany will always stay her favourite destinations to visit.", "ToT2"],
                    ["The music of Spain and the history of Spain deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.", "GoT4"],
                ],
                inputs=[input_text, method_dropdown],
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
        fn=run_keyword_counting_with_got,
        inputs=[input_text, method_dropdown],
        outputs=[result_output, prompt_output]
    )

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("启动Graph of Thoughts关键词计数演示...")
    print("请确保config.json文件存在并包含正确的API配置")
    
    demo.launch(
        server_port=7863, 
        share=False,
        show_error=True
    ) 