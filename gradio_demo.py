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

# 导入sorting模块
try:
    from examples.sorting.sorting_032 import SortingPrompter as SortingPrompter32, SortingParser as SortingParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.sorting.sorting_064 import SortingPrompter as SortingPrompter64, SortingParser as SortingParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.sorting.sorting_128 import SortingPrompter as SortingPrompter128, SortingParser as SortingParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.sorting.utils as utils
except ImportError:
    # 如果从根目录运行，尝试直接导入
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.sorting.sorting_032 import SortingPrompter as SortingPrompter32, SortingParser as SortingParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.sorting.sorting_064 import SortingPrompter as SortingPrompter64, SortingParser as SortingParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.sorting.sorting_128 import SortingPrompter as SortingPrompter128, SortingParser as SortingParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.sorting.utils as utils

def get_sorting_modules(length):
    """
    根据输入长度获取对应的排序模块
    
    :param length: 输入数组长度
    :type length: int
    :return: 对应的排序模块
    :rtype: tuple
    """
    if length == 32:
        return (SortingPrompter32, SortingParser32, 
                {"IO": io32, "CoT": cot32, "ToT": tot32, "ToT2": tot232, "GoT": got32})
    elif length == 64:
        return (SortingPrompter64, SortingParser64, 
                {"IO": io64, "CoT": cot64, "ToT": tot64, "ToT2": tot264, "GoT": got64})
    elif length == 128:
        return (SortingPrompter128, SortingParser128, 
                {"IO": io128, "CoT": cot128, "ToT": tot128, "ToT2": tot2128, "GoT": got128})
    else:
        raise ValueError(f"不支持的长度 {length}。支持的长度：32, 64, 128")

def get_prompt_content(length, method_name):
    """
    获取对应长度和方法的提示词内容
    
    :param length: 输入数组长度
    :type length: int
    :param method_name: 排序方法名称
    :type method_name: str
    :return: 提示词内容
    :rtype: str
    """
    if length == 32:
        from examples.sorting.sorting_032 import SortingPrompter
        prompter = SortingPrompter()
    elif length == 64:
        from examples.sorting.sorting_064 import SortingPrompter
        prompter = SortingPrompter()
    elif length == 128:
        from examples.sorting.sorting_128 import SortingPrompter
        prompter = SortingPrompter()
    else:
        return "不支持的长度"
    
    method_lower = method_name.lower()
    
    if method_lower == "direct_method":
        return prompter.sort_prompt
    elif method_lower == "cot":
        return prompter.sort_prompt_cot
    elif method_lower == "tot":
        return prompter.tot_improve_prompt
    elif method_lower == "tot2":
        return prompter.tot_improve_prompt
    elif method_lower == "got":
        return prompter.got_split_prompt
    else:
        return "不支持的方法"

def run_sorting_with_got(input_str: str, method_name: str) -> tuple:
    """
    使用Graph of Thoughts方法对输入数组进行排序
    
    :param input_str: 输入的数字，用逗号分隔
    :type input_str: str
    :param method_name: 排序方法名称
    :type method_name: str
    :return: (排序结果, 提示词内容)
    :rtype: tuple
    """
    try:
        # 解析输入
        nums = [int(x.strip()) for x in input_str.split(",")]
        
        # 验证输入长度
        if len(nums) not in [32, 64, 128]:
            error_msg = f"错误：输入长度必须是32、64或128个数字，当前输入了{len(nums)}个数字。请重新输入正确数量的数字，用逗号分隔。"
            return error_msg, "无提示词（输入错误）"
        
        # 验证数字范围（0-9）
        if not all(0 <= x <= 9 for x in nums):
            error_msg = "错误：所有数字必须在0-9范围内。"
            return error_msg, "无提示词（输入错误）"
        
        # 获取对应的排序模块
        try:
            prompter_class, parser_class, method_map = get_sorting_modules(len(nums))
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
        
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        
        # 获取提示词内容
        prompt_content = get_prompt_content(len(nums), method_name)
        
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
        operations_graph = method()
        
        # 创建执行器
        executor = controller.Controller(
            lm,
            operations_graph,
            prompter_class(),
            parser_class(),
            {
                "original": str(nums),
                "current": "",
                "phase": 0,
                "method": method_name.lower(),
            },
        )
        
        # 运行排序
        executor.run()
        
        # 获取最终结果
        final_thoughts = executor.get_final_thoughts()
        if final_thoughts and final_thoughts[0]:
            result = final_thoughts[0][0].state.get("current", "")
        else:
            result = "未获得结果"
        
        # 计算错误数
        correct_sorted = sorted(nums)
        # 创建状态字典用于错误计算
        state_dict = {
            "original": str(nums),
            "current": result
        }
        error_count = utils.num_errors(state_dict)
        
        # 计算成本
        cost = lm.cost
        
        # 检查是否正确排序
        is_correct = utils.test_sorting(state_dict)
        
        # 格式化输出
        output = f"""排序结果:

输入数组: {nums}
排序结果: {result}
正确排序: {correct_sorted}
错误数: {error_count}
使用成本: ${cost:.6f}
使用方法: {method_name}
排序状态: {'✅ 完全正确' if is_correct else f'❌ 有{error_count}个错误'}

详细信息:
- 输入长度: {len(nums)} 个数字
- 数字范围: {min(nums)} - {max(nums)}
- 执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 模型: deepseek
- 缓存: 启用"""
        
        return output, prompt_content
        
    except ValueError as e:
        error_msg = f"输入格式错误: {str(e)}。请确保输入的是用逗号分隔的数字。"
        return error_msg, "无提示词（格式错误）"
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"

# 创建Gradio界面
with gr.Blocks(title="Graph of Thoughts 排序演示") as demo:
    gr.Markdown("# Graph of Thoughts 排序演示")
    gr.Markdown("使用不同的思维方法对32、64或128个数字进行排序，并显示使用的提示词")
    
    with gr.Row():
        # 左上：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 输入区域")
            input_text = gr.Textbox(
                label="输入数字（用逗号分隔）",
                placeholder="例如: 5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0",
                info="请输入32、64或128个0-9之间的数字，用逗号分隔",
                lines=8
            )
            method_dropdown = gr.Dropdown(
                choices=["IO", "CoT", "ToT", "ToT2", "GoT"], 
                value="IO", 
                label="选择排序方法",
                info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT: 思维图"
            )
            run_button = gr.Button("开始排序", variant="primary")
        
        # 右上：结果区域
        with gr.Column(scale=1):
            gr.Markdown("### 排序结果")
            result_output = gr.Textbox(
                label="排序结果",
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
                    ["5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0", "IO"],
                    ["3,7,0,2,8,1,2,2,2,4,7,8,5,5,3,9,4,3,5,6,6,4,4,5,2,0,9,3,3,9,2,1", "CoT"],
                    ["6,4,5,7,5,6,9,7,6,9,4,6,9,8,1,9,2,4,9,0,7,6,5,6,6,2,8,3,9,5,6,1", "ToT"],
                    ["5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0", "ToT2"],
                    ["5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0", "GoT"],
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
        fn=run_sorting_with_got,
        inputs=[input_text, method_dropdown],
        outputs=[result_output, prompt_output]
    )

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("启动Graph of Thoughts排序演示...")
    print("请确保config.json文件存在并包含正确的API配置")
    
    demo.launch(
        server_port=7862, 
        share=False,
        show_error=True
    ) 