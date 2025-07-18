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

# 导入set_intersection模块
try:
    from examples.set_intersection.set_intersection_032 import SetIntersectionPrompter as SetIntersectionPrompter32, SetIntersectionParser as SetIntersectionParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.set_intersection.set_intersection_064 import SetIntersectionPrompter as SetIntersectionPrompter64, SetIntersectionParser as SetIntersectionParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.set_intersection.set_intersection_128 import SetIntersectionPrompter as SetIntersectionPrompter128, SetIntersectionParser as SetIntersectionParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.set_intersection.utils as utils
except ImportError:
    # 如果从根目录运行，尝试直接导入
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.set_intersection.set_intersection_032 import SetIntersectionPrompter as SetIntersectionPrompter32, SetIntersectionParser as SetIntersectionParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.set_intersection.set_intersection_064 import SetIntersectionPrompter as SetIntersectionPrompter64, SetIntersectionParser as SetIntersectionParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.set_intersection.set_intersection_128 import SetIntersectionPrompter as SetIntersectionPrompter128, SetIntersectionParser as SetIntersectionParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.set_intersection.utils as utils

def get_set_intersection_modules(length):
    """
    根据输入长度获取对应的集合交集模块
    
    :param length: 输入数组长度
    :type length: int
    :return: 对应的集合交集模块
    :rtype: tuple
    """
    if length == 32:
        return (SetIntersectionPrompter32, SetIntersectionParser32, 
                {"IO": io32, "CoT": cot32, "ToT": tot32, "ToT2": tot232, "GoT": got32})
    elif length == 64:
        return (SetIntersectionPrompter64, SetIntersectionParser64, 
                {"IO": io64, "CoT": cot64, "ToT": tot64, "ToT2": tot264, "GoT": got64})
    elif length == 128:
        return (SetIntersectionPrompter128, SetIntersectionParser128, 
                {"IO": io128, "CoT": cot128, "ToT": tot128, "ToT2": tot2128, "GoT": got128})
    else:
        raise ValueError(f"不支持的长度 {length}。支持的长度：32, 64, 128")

def get_prompt_content(length, method_name):
    """
    获取对应长度和方法的提示词内容
    
    :param length: 输入数组长度
    :type length: int
    :param method_name: 集合交集方法名称
    :type method_name: str
    :return: 提示词内容
    :rtype: str
    """
    if length == 32:
        from examples.set_intersection.set_intersection_032 import SetIntersectionPrompter
        prompter = SetIntersectionPrompter()
    elif length == 64:
        from examples.set_intersection.set_intersection_064 import SetIntersectionPrompter
        prompter = SetIntersectionPrompter()
    elif length == 128:
        from examples.set_intersection.set_intersection_128 import SetIntersectionPrompter
        prompter = SetIntersectionPrompter()
    else:
        return "不支持的长度"
    
    method_lower = method_name.lower()
    
    if method_lower == "direct_method":
        return prompter.intersection_prompt
    elif method_lower == "cot":
        return prompter.intersection_prompt_cot
    elif method_lower == "tot":
        return prompter.tot_improve_prompt
    elif method_lower == "tot2":
        return prompter.tot_improve_prompt
    elif method_lower == "got":
        return prompter.got_split_prompt
    else:
        return "不支持的方法"

def parse_set_input(input_str):
    """
    解析集合输入字符串
    
    :param input_str: 输入的集合字符串
    :type input_str: str
    :return: 解析后的数字列表
    :rtype: list
    """
    try:
        # 移除方括号和空格，然后按逗号分割
        cleaned = input_str.strip().replace('[', '').replace(']', '').replace(' ', '')
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    except ValueError:
        raise ValueError("输入格式错误，请确保输入的是用逗号分隔的数字")

def run_set_intersection_with_got(set1_str: str, set2_str: str, method_name: str) -> tuple:
    """
    使用Graph of Thoughts方法计算两个集合的交集
    
    :param set1_str: 第一个集合，用逗号分隔的数字
    :type set1_str: str
    :param set2_str: 第二个集合，用逗号分隔的数字
    :type set2_str: str
    :param method_name: 集合交集方法名称
    :type method_name: str
    :return: (交集结果, 提示词内容)
    :rtype: tuple
    """
    try:
        # 解析输入
        set1_nums = parse_set_input(set1_str)
        set2_nums = parse_set_input(set2_str)
        
        # 验证输入长度
        if len(set1_nums) not in [32, 64, 128] or len(set2_nums) not in [32, 64, 128]:
            error_msg = f"错误：每个集合的长度必须是32、64或128个数字。当前集合1有{len(set1_nums)}个数字，集合2有{len(set2_nums)}个数字。"
            return error_msg, "无提示词（输入错误）"
        
        if len(set1_nums) != len(set2_nums):
            error_msg = f"错误：两个集合的长度必须相同。当前集合1有{len(set1_nums)}个数字，集合2有{len(set2_nums)}个数字。"
            return error_msg, "无提示词（输入错误）"
        
        # 验证数字范围（0-127，因为集合交集示例使用这个范围）
        if not all(0 <= x <= 127 for x in set1_nums + set2_nums):
            error_msg = "错误：所有数字必须在0-127范围内。"
            return error_msg, "无提示词（输入错误）"
        
        # 获取对应的集合交集模块
        try:
            prompter_class, parser_class, method_map = get_set_intersection_modules(len(set1_nums))
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
        
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        
        # 获取提示词内容
        prompt_content = get_prompt_content(len(set1_nums), method_name)
        
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
                "set1": str(set1_nums),
                "set2": str(set2_nums),
                "current": "",
                "phase": 0,
                "method": method_name.lower(),
            },
        )
        
        # 运行集合交集计算
        executor.run()
        
        # 获取最终结果
        final_thoughts = executor.get_final_thoughts()
        if final_thoughts and final_thoughts[0]:
            result = final_thoughts[0][0].state.get("current", "")
        else:
            result = "未获得结果"
        
        # 计算正确的交集
        set1_set = set(set1_nums)
        set2_set = set(set2_nums)
        correct_intersection = sorted(list(set1_set & set2_set))
        
        # 计算错误数
        # 创建状态字典用于错误计算
        state_dict = {
            "set1": str(set1_nums),
            "set2": str(set2_nums),
            "current": result
        }
        error_count = utils.num_errors(state_dict)
        
        # 计算成本
        cost = lm.cost
        
        # 检查是否正确计算交集
        is_correct = utils.test_set_intersection({
            "result": str(correct_intersection),
            "current": result
        })
        
        # 格式化输出
        output = f"""集合交集结果:

集合1: {set1_nums}
集合2: {set2_nums}
AI计算交集: {result}
正确交集: {correct_intersection}
错误数: {error_count}
使用成本: ${cost:.6f}
使用方法: {method_name}
计算状态: {'✅ 完全正确' if is_correct else f'❌ 有{error_count}个错误'}

详细信息:
- 集合长度: {len(set1_nums)} 个数字
- 数字范围: {min(set1_nums + set2_nums)} - {max(set1_nums + set2_nums)}
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
with gr.Blocks(title="Graph of Thoughts 集合交集演示") as demo:
    gr.Markdown("# Graph of Thoughts 集合交集演示")
    gr.Markdown("使用不同的思维方法计算两个集合的交集，并显示使用的提示词")
    
    with gr.Row():
        # 左上：输入区域
        with gr.Column(scale=1):
            gr.Markdown("### 输入区域")
            set1_input = gr.Textbox(
                label="集合1（用逗号分隔的数字）",
                placeholder="例如: 13,16,30,6,21,7,31,15,11,1,24,10,9,3,20,8",
                info="请输入32、64或128个0-127之间的数字，用逗号分隔",
                lines=6
            )
            set2_input = gr.Textbox(
                label="集合2（用逗号分隔的数字）",
                placeholder="例如: 25,24,10,4,27,0,14,12,8,2,29,20,17,19,26,23",
                info="请输入与集合1相同数量的0-127之间的数字，用逗号分隔",
                lines=6
            )
            method_dropdown = gr.Dropdown(
                choices=["IO", "CoT", "ToT", "ToT2", "GoT"], 
                value="IO", 
                label="选择集合交集方法",
                info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT: 思维图"
            )
            run_button = gr.Button("计算交集", variant="primary")
        
        # 右上：结果区域
        with gr.Column(scale=1):
            gr.Markdown("### 交集结果")
            result_output = gr.Textbox(
                label="交集计算结果",
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
                    ["13,16,30,6,21,7,31,15,11,1,24,10,9,3,20,8", "25,24,10,4,27,0,14,12,8,2,29,20,17,19,26,23", "IO"],
                    ["26,40,42,57,15,31,5,32,11,4,24,28,51,54,12,22,33,35,7,13,2,59,8,23,43,16,29,55,25,63,21,18", "16,60,36,48,0,15,5,19,46,24,1,6,61,10,38,53,58,9,44,14,35,63,52,20,27,17,39,47,34,56,40,59", "CoT"],
                    ["115,61,35,103,90,117,86,44,63,45,40,30,74,33,31,1,118,48,38,0,119,51,64,78,15,121,89,101,79,69,120,29", "13,35,20,96,34,18,47,127,126,9,21,16,77,22,111,122,85,73,42,105,123,15,33,59,67,57,104,8,30,89,76,12", "ToT"],
                    ["5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0", "3,7,0,2,8,1,2,2,2,4,7,8,5,5,3,9,4,3,5,6,6,4,4,5,2,0,9,3,3,9,2,1,3,7,0,2,8,1,2,2,2,4,7,8,5,5,3,9,4,3,5,6,6,4,4,5,2,0,9,3,3,9,2,1", "ToT2"],
                    ["6,4,5,7,5,6,9,7,6,9,4,6,9,8,1,9,2,4,9,0,7,6,5,6,6,2,8,3,9,5,6,1,6,4,5,7,5,6,9,7,6,9,4,6,9,8,1,9,2,4,9,0,7,6,5,6,6,2,8,3,9,5,6,1", "8,2,1,3,7,4,0,9,5,1,2,6,8,3,4,7,9,0,1,5,2,8,6,3,4,7,9,1,0,5,2,8,8,2,1,3,7,4,0,9,5,1,2,6,8,3,4,7,9,0,1,5,2,8,6,3,4,7,9,1,0,5,2,8", "GoT"],
                ],
                inputs=[set1_input, set2_input, method_dropdown],
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
        fn=run_set_intersection_with_got,
        inputs=[set1_input, set2_input, method_dropdown],
        outputs=[result_output, prompt_output]
    )

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("启动Graph of Thoughts集合交集演示...")
    print("请确保config.json文件存在并包含正确的API配置")
    
    demo.launch(
        server_port=7863, 
        share=False,
        show_error=True
    ) 