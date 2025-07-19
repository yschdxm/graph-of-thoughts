import asyncio
import gradio as gr
import json
import os
import sys
import logging
import datetime
import csv
from typing import Dict, List, Callable, Union, Any, Set

from matplotlib import pyplot as plt

from examples.doc_merge import plot as doc_merge_plot
from examples.keyword_counting import keyword_counting, plot as keyword_counting_plot
from examples.set_intersection import plot as set_intersection_plot
from examples.sorting import plot as sorting_plot


# 添加项目路径到sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入Graph of Thoughts相关模块
from graph_of_thoughts import controller, language_models, operations, prompter, parser

# ==================== 排序模块导入 ====================
try:
    from examples.sorting.sorting_032 import SortingPrompter as SortingPrompter32, SortingParser as SortingParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.sorting.sorting_064 import SortingPrompter as SortingPrompter64, SortingParser as SortingParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.sorting.sorting_128 import SortingPrompter as SortingPrompter128, SortingParser as SortingParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.sorting.utils as sorting_utils
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.sorting.sorting_032 import SortingPrompter as SortingPrompter32, SortingParser as SortingParser32, direct_method as io32, cot as cot32, tot as tot32, tot2 as tot232, got as got32
    from examples.sorting.sorting_064 import SortingPrompter as SortingPrompter64, SortingParser as SortingParser64, direct_method as io64, cot as cot64, tot as tot64, tot2 as tot264, got as got64
    from examples.sorting.sorting_128 import SortingPrompter as SortingPrompter128, SortingParser as SortingParser128, direct_method as io128, cot as cot128, tot as tot128, tot2 as tot2128, got as got128
    import examples.sorting.utils as sorting_utils

# ==================== 文档合并模块导入 ====================
try:
    from examples.doc_merge.doc_merge import (
        DocMergePrompter, DocMergeParser, 
        direct_method as doc_io, cot as doc_cot, tot as doc_tot, got as doc_got, got2 as doc_got2
    )
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.doc_merge.doc_merge import (
        DocMergePrompter, DocMergeParser, 
        direct_method as doc_io, cot as doc_cot, tot as doc_tot, got as doc_got, got2 as doc_got2
    )

# ==================== 关键词计数模块导入 ====================
try:
    from examples.keyword_counting.keyword_counting import KeywordCountingPrompter, KeywordCountingParser, direct_method as kw_io, cot as kw_cot, tot as kw_tot, tot2 as kw_tot2, got4 as kw_got4, got8 as kw_got8, gotx as kw_gotx
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.keyword_counting.keyword_counting import KeywordCountingPrompter, KeywordCountingParser, direct_method as kw_io, cot as kw_cot, tot as kw_tot, tot2 as kw_tot2, got4 as kw_got4, got8 as kw_got8, gotx as kw_gotx

# ==================== 集合交集模块导入 ====================
try:
    from examples.set_intersection.set_intersection_032 import SetIntersectionPrompter as SetIntersectionPrompter32, SetIntersectionParser as SetIntersectionParser32, direct_method as si_io32, cot as si_cot32, tot as si_tot32, tot2 as si_tot232, got as si_got32
    from examples.set_intersection.set_intersection_064 import SetIntersectionPrompter as SetIntersectionPrompter64, SetIntersectionParser as SetIntersectionParser64, direct_method as si_io64, cot as si_cot64, tot as si_tot64, tot2 as si_tot264, got as si_got64
    from examples.set_intersection.set_intersection_128 import SetIntersectionPrompter as SetIntersectionPrompter128, SetIntersectionParser as SetIntersectionParser128, direct_method as si_io128, cot as si_cot128, tot as si_tot128, tot2 as si_tot2128, got as si_got128
    import examples.set_intersection.utils as set_intersection_utils
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "examples"))
    from examples.set_intersection.set_intersection_032 import SetIntersectionPrompter as SetIntersectionPrompter32, SetIntersectionParser as SetIntersectionParser32, direct_method as si_io32, cot as si_cot32, tot as si_tot32, tot2 as si_tot232, got as si_got32
    from examples.set_intersection.set_intersection_064 import SetIntersectionPrompter as SetIntersectionPrompter64, SetIntersectionParser as SetIntersectionParser64, direct_method as si_io64, cot as si_cot64, tot as si_tot64, tot2 as si_tot264, got as si_got64
    from examples.set_intersection.set_intersection_128 import SetIntersectionPrompter as SetIntersectionPrompter128, SetIntersectionParser as SetIntersectionParser128, direct_method as si_io128, cot as si_cot128, tot as si_tot128, tot2 as si_tot2128, got as si_got128
    import examples.set_intersection.utils as set_intersection_utils

# ==================== 排序功能 ====================
def get_sorting_modules(length):
    if length == 32:
        return (SortingPrompter32, SortingParser32, 
                {"Direct_Method": io32, "CoT": cot32, "ToT": tot32, "ToT2": tot232, "GoT": got32})
    elif length == 64:
        return (SortingPrompter64, SortingParser64, 
                {"Direct_Method": io64, "CoT": cot64, "ToT": tot64, "ToT2": tot264, "GoT": got64})
    elif length == 128:
        return (SortingPrompter128, SortingParser128, 
                {"Direct_Method": io128, "CoT": cot128, "ToT": tot128, "ToT2": tot2128, "GoT": got128})
    else:
        raise ValueError(f"不支持的长度 {length}。支持的长度：32, 64, 128")

def get_sorting_prompt_content(length, method_name, input_str=None, current=None, phase=None):
    """
    获取排序任务的提示词内容
    
    :param length: 输入数字列表的长度 (32, 64, 128)
    :param method_name: 方法名称 (IO, CoT, ToT, ToT2, GoT)
    :param input_str: 原始输入字符串
    :param current: 当前中间结果 (用于ToT/ToT2/GoT)
    :param phase: 当前阶段 (用于GoT)
    :return: 生成的提示词内容
    """
    if length == 32:
        prompter = SortingPrompter32()
    elif length == 64:
        prompter = SortingPrompter64()
    elif length == 128:
        prompter = SortingPrompter128()
    else:
        return f"不支持的长度 {length}"

    method_lower = method_name.lower()
    
    # 准备输入参数
    if input_str is None:
        input_str = "[输入内容]"
    
    # 根据方法类型生成不同的提示词
    if method_lower == "direct_method" or method_lower == "io":
        return prompter.sort_prompt.format(input=input_str)
    elif method_lower == "cot":
        return prompter.sort_prompt_cot.format(input=input_str)
    elif method_lower in ["tot", "tot2"]:
        if current is None or current == "":
            # 初始阶段使用普通排序提示
            return prompter.sort_prompt.format(input=input_str)
        else:
            # 改进阶段使用ToT改进提示
            return prompter.tot_improve_prompt.format(
                input=input_str,
                incorrectly_sorted=current,
                length=length
            )
    elif method_lower == "got":
        if current is None or current == "":
            # 初始阶段使用分割提示
            return prompter.got_split_prompt.format(input=input_str)
        elif phase == 1:
            # 子列表排序阶段使用普通排序提示
            return prompter.sort_prompt.format(input=current)
        else:
            # 合并改进阶段使用ToT改进提示
            return prompter.tot_improve_prompt.format(
                input=input_str,
                incorrectly_sorted=current,
                length=length
            )
    else:
        return "不支持的方法"


def run_sorting(input_str: str, method_name: str, model_name: str = "doubao-lite-32k") -> tuple:
    try:
        nums = [int(x.strip()) for x in input_str.split(",")]
        
        if len(nums) not in [32, 64, 128]:
            error_msg = f"错误：输入长度必须是32、64或128个数字，当前输入了{len(nums)}个数字。"
            return error_msg, "无提示词（输入错误）"
        
        if not all(0 <= x <= 9 for x in nums):
            error_msg = "错误：所有数字必须在0-9范围内。"
            return error_msg, "无提示词（输入错误）"
        
        try:
            prompter_class, parser_class, method_map = get_sorting_modules(len(nums))
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
        if method_name == "IO":
            method_name = "Direct_Method"
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        prompt_content = get_sorting_prompt_content(
            length=len(nums),
            method_name=method_name,
            input_str=str(nums),
            current="",  # 初始没有当前结果
            phase=0      # 初始阶段
        )
        
        config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts/language_models/config.json")
        if not os.path.exists(config_path):
            template_config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts", "language_models", "config_template.json")
            if os.path.exists(template_config_path):
                error_msg = "错误：请复制 config_template.json 为 config.json 并配置您的API密钥。"
                return error_msg, "无提示词（配置错误）"
            else:
                error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
                return error_msg, "无提示词（配置错误）"
            
        operations_graph = method()
        lm = language_models.ChatGPT(config_path, model_name=model_name, cache=True)
        
        # 创建新的事件循环用于异步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            executor = controller.Controller(
                lm, operations_graph, prompter_class(), parser_class(),
                {"original": str(nums), "current": "", "phase": 0, "method": method_name.lower()},
            )
            
            executor.run()
            
            temp_path = os.path.join(os.path.dirname(__file__), "temp.json")
            executor.output_graph(temp_path)
            
            # Read and parse the temp.json file
            with open(temp_path, 'r') as f:
                execution_data = json.load(f)
            
            # Extract the process details
            process_details = []
            for operation in execution_data:
                operation_name = operation.get("operation")
                if "thoughts" in operation:
                    for thought in operation["thoughts"]:
                        if "current" in thought:
                            process_details.append(f"操作 {operation_name}, 阶段 {thought.get('phase', '?')}: {thought['current']}")
            
            # Get the final result from the ground truth evaluator
            final_result = "未获得结果"
            for operation in reversed(execution_data):
                if operation.get("operation") == "ground_truth_evaluator" and "thoughts" in operation:
                    if operation["thoughts"] and "current" in operation["thoughts"][0]:
                        final_result = operation["thoughts"][0]["current"]
                        break
            
            correct_sorted = sorted(nums)
            state_dict = {"original": str(nums), "current": final_result}
            error_count = sorting_utils.num_errors(state_dict)
            cost = lm.cost
            is_correct = sorting_utils.test_sorting(state_dict)
            if method_name == "Direct_Method":
                method_name = "IO"
            output = f"""排序结果:

输入数组: {nums}
排序结果: {final_result}
正确排序: {correct_sorted}
错误数: {error_count}
使用成本: ${cost:.6f}
使用方法: {method_name}
排序状态: {'✅ 完全正确' if is_correct else f'❌ 有{error_count}个错误'}

详细信息:
- 输入长度: {len(nums)} 个数字
- 数字范围: {min(nums)} - {max(nums)}
- 执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 模型: {model_name}

执行过程:
"""
            output += "\n".join(process_details)
            
            return output, prompt_content
            
        finally:
            # 关闭事件循环
            loop.close()
        
    except ValueError as e:
        error_msg = f"输入格式错误: {str(e)}。请确保输入的是用逗号分隔的数字。"
        return error_msg, "无提示词（格式错误）"
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"



# ==================== 文档合并功能 ====================
def load_sample_documents(sample_id: int = 0) -> List[str]:
    try:
        data_path = os.path.join("examples", "doc_merge", "documents.csv")
        documents = []
        with open(data_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                if i == sample_id:
                    documents = row[2:6]
                    break
        return documents
    except Exception as e:
        logging.error(f"加载文档失败: {e}")
        return []

def get_available_samples() -> List[int]:
    try:
        data_path = os.path.join("examples", "doc_merge", "documents.csv")
        samples = []
        with open(data_path, "r", encoding="utf8") as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                samples.append(i)
        return samples[:10]
    except Exception as e:
        logging.error(f"获取样本列表失败: {e}")
        return [0]

def get_doc_merge_method_map():
    return {
        "Direct_Method": doc_io,
        "CoT": doc_cot, 
        "ToT": doc_tot,
        "GoT": doc_got,
        "GoT2": doc_got2
    }

def get_doc_merge_prompt_content(method_name: str, documents: List[str], current: str = None, parts: Set[int] = None) -> str:
    """
    生成文档合并任务的完整提示词内容
    
    :param method_name: 方法名称 (IO/Direct_Method, CoT, ToT, GoT, GoT2)
    :param documents: 要合并的文档列表
    :param current: 当前中间结果 (用于ToT/GoT的改进阶段)
    :param parts: 当前处理的文档部分索引 (用于GoT)
    :return: 完整的提示词内容
    """
    try:
        from examples.doc_merge.doc_merge import DocMergePrompter
        prompter = DocMergePrompter()
        
        method_lower = method_name.lower()
        num_docs = len(documents)
        
        # 统一处理 IO/Direct_Method 名称
        if method_lower == "io":
            method_lower = "direct_method"
        
        # 基础文档块
        def format_doc_blocks(docs, start_idx=1):
            blocks = ""
            for i, doc in enumerate(docs, start=start_idx):
                blocks += prompter.merge_doc_prompt_block.format(document=doc, num=i)
            return blocks
        
        # 基础提示词生成
        if method_lower == "direct_method":
            prompt = prompter.merge_doc_prompt_start.format(num=num_docs)
            prompt += format_doc_blocks(documents)
            return prompt
            
        elif method_lower == "cot":
            prompt = prompter.merge_doc_prompt_cot_start.format(num=num_docs)
            prompt += format_doc_blocks(documents)
            return prompt
            
        elif method_lower == "tot":
            if not current:
                # 初始阶段 - 生成第一个合并版本
                prompt = prompter.merge_doc_prompt_start.format(num=num_docs)
                prompt += format_doc_blocks(documents)
            else:
                # 改进阶段 - 优化现有合并结果
                prompt = prompter.improve_summary_prompt_start.format(num=num_docs)
                prompt += format_doc_blocks(documents)
                prompt += prompter.improve_summary_prompt_end.format(summary=current)
            return prompt
            
        elif method_lower == "got":
            if not parts:
                parts = set(range(num_docs))
                
            if not current:
                # 初始阶段 - 生成部分文档的合并
                selected_docs = [documents[i] for i in sorted(parts)]
                prompt = prompter.merge_doc_prompt_start.format(num=len(parts))
                prompt += format_doc_blocks(selected_docs)
            else:
                # 改进阶段 - 优化部分合并结果
                selected_docs = [documents[i] for i in sorted(parts)]
                prompt = prompter.improve_summary_prompt_start.format(num=len(parts))
                prompt += format_doc_blocks(selected_docs)
                prompt += prompter.improve_summary_prompt_end.format(summary=current)
            return prompt
            
        elif method_lower == "got2":
            if not current:
                # GoT2的初始聚合提示
                prompt = prompter.aggregate_sub_prompt_base.format(num_ndas=num_docs)
                for i, doc in enumerate(documents, 1):
                    prompt += prompter.aggregate_sub_prompt_generate.format(nda=doc, num=i)
            else:
                # GoT2的改进提示
                prompt = prompter.improve_summary_prompt_start.format(num=num_docs)
                prompt += format_doc_blocks(documents)
                prompt += prompter.improve_summary_prompt_end.format(summary=current)
            return prompt
            
        else:
            return f"不支持的方法: {method_name}"
            
    except Exception as e:
        error_msg = f"生成提示词时出错: {str(e)}"
        logging.error(error_msg)
        return error_msg


def run_doc_merge(method_name: str, sample_id: str, custom_docs: str, model_name: str = "deepseek") -> tuple:
    try:
        # 处理输入文档
        if custom_docs.strip():
            try:
                if custom_docs.strip().startswith('['):
                    documents = json.loads(custom_docs)
                else:
                    documents = [doc.strip() for doc in custom_docs.split('\n') if doc.strip()]
                
                if len(documents) < 2:
                    return "错误: 需要至少2个文档进行合并", "无提示词（输入错误）"
            except Exception as e:
                return f"错误: 解析自定义文档失败 - {str(e)}", "无提示词（解析错误）"
        else:
            try:
                sample_num = int(sample_id.split()[-1])
                documents = load_sample_documents(sample_num)
                if not documents:
                    return "错误: 无法加载示例文档", "无提示词（加载失败）"
            except (ValueError, IndexError):
                return "错误: 无效的样本ID格式", "无提示词（格式错误）"
        
        # 检查配置文件
        config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts/language_models/config.json")
        if not os.path.exists(config_path):
            error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
            return error_msg, "无提示词（配置错误）"
        
        # 统一处理 IO/Direct_Method 名称
        if method_name == "IO":
            method_name = "Direct_Method"
            
        # 获取方法映射
        method_map = get_doc_merge_method_map()
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, GoT, GoT2"
            return error_msg, "无提示词（方法错误）"
        
        # 创建事件循环处理异步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 初始化模型和操作图
            lm = language_models.ChatGPT(config_path, model_name=model_name, cache=True)
            method = method_map[method_name]
            operations_graph = method()
            
            # 获取初始提示词
            prompt_content = get_doc_merge_prompt_content(
                method_name=method_name,
                documents=documents,
                current="",
                parts=set()
            )
            
            # 执行控制器
            executor = controller.Controller(
                lm, operations_graph, DocMergePrompter(), DocMergeParser(),
                {
                    "documents": documents, 
                    "current": "", 
                    "parts": set(), 
                    "method": method_name.lower()
                },
            )
            
            executor.run()
            
            # 获取最终结果
            final_thoughts = executor.get_final_thoughts()
            if final_thoughts and len(final_thoughts) > 0 and len(final_thoughts[0]) > 0:
                result = final_thoughts[0][0].state.get("current", "无结果")
            else:
                result = "无结果"
            
            cost = lm.cost
            
            output = f"""文档合并结果:

使用方法: {method_name}
文档数量: {len(documents)}
使用成本: ${cost:.6f}
执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
模型: {model_name}

合并后的文档:
{result}

详细信息:
- 输入文档长度: {[len(doc) for doc in documents]}
- 输出文档长度: {len(result)}
- 平均文档长度: {sum(len(doc) for doc in documents) / len(documents):.0f} 字符
- 压缩比: {len(result) / sum(len(doc) for doc in documents) * 100:.1f}%"""
            
            return output, prompt_content
            
        finally:
            loop.close()
            
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"


# ==================== 关键词计数功能 ====================
def get_keyword_counting_modules():
    return (KeywordCountingPrompter, KeywordCountingParser, 
            {"Direct_Method": kw_io, "CoT": kw_cot, "ToT": kw_tot, 
             "ToT2": kw_tot2, "GoT4": kw_got4, "GoT8": kw_got8, "GoTx": kw_gotx})

def get_keyword_counting_prompt_content(method_name: str, input_text: str = None, 
                                      current: str = None, phase: int = 0, 
                                      sub_text: str = None) -> str:
    """
    获取关键词计数任务的完整提示词内容
    
    :param method_name: 方法名称 (IO/Direct_Method, CoT, ToT, ToT2, GoT4, GoT8, GoTx)
    :param input_text: 输入文本
    :param current: 当前中间结果
    :param phase: 当前阶段 (用于GoT方法)
    :param sub_text: 子文本 (用于GoT方法)
    :return: 生成的提示词内容
    """
    prompter = KeywordCountingPrompter()
    
    # 统一处理 IO/Direct_Method 名称
    method_lower = method_name.lower()
    if method_lower == "io":
        method_lower = "direct_method"
    
    # 处理输入文本
    if input_text is None:
        input_text = "[输入文本]"
    
    if method_lower == "direct_method":
        return prompter.count_prompt.format(input=input_text)
    elif method_lower == "cot":
        return prompter.count_prompt_cot.format(input=input_text)
    elif method_lower in ["tot", "tot2"]:
        if current is None or current == "":
            return prompter.count_prompt_cot.format(input=input_text)
        else:
            return prompter.tot_improve_prompt.format(
                input=input_text,
                incorrect_dict=current
            )
    elif method_lower.startswith("got"):
        if current is None or current == "" and phase == 0:
            # GoT初始阶段 - 分割文本
            if method_lower == "got8":
                return prompter.got_split_prompt2.format(input=input_text)
            elif method_lower == "gotx":
                return prompter.got_split_prompt3.format(input=input_text)
            else:  # GoT4
                return prompter.got_split_prompt.format(input=input_text)
        elif phase == 1:
            # GoT子文本计数阶段
            if method_lower == "gotx":
                return prompter.count_prompt_sentence.format(input=sub_text)
            else:
                return prompter.count_prompt_cot.format(input=sub_text)
        else:
            # GoT改进阶段
            if method_lower == "gotx":
                return prompter.sentence_improve_prompt.format(
                    input=sub_text if sub_text else input_text,
                    incorrect_dict=current
                )
            else:
                return prompter.tot_improve_prompt.format(
                    input=sub_text if sub_text else input_text,
                    incorrect_dict=current
                )
    else:
        return f"不支持的方法: {method_name}"

def run_keyword_counting(input_text: str, method_name: str, model_name: str = "deepseek") -> tuple:
    try:
        if not input_text.strip():
            error_msg = "错误：请输入文本内容。"
            return error_msg, "无提示词（输入错误）"
        
        # 统一处理 IO/Direct_Method 名称
        if method_name == "IO":
            method_name = "Direct_Method"
            
        # 获取模块和方法映射
        try:
            prompter_class, parser_class, method_map = get_keyword_counting_modules()
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
            
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT4, GoT8, GoTx"
            return error_msg, "无提示词（方法错误）"
        
        # 获取初始提示词
        prompt_content = get_keyword_counting_prompt_content(
            method_name=method_name,
            input_text=input_text,
            current="",
            phase=0
        )
        
        # 检查配置文件
        config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts/language_models/config.json")
        if not os.path.exists(config_path):
            error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
            return error_msg, "无提示词（配置错误）"
        
        # 创建新的事件循环处理异步操作
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # 初始化模型和操作图
            lm = language_models.ChatGPT(config_path, model_name=model_name, cache=True)
            method = method_map[method_name]
            
            # 执行控制器
            executor = controller.Controller(
                lm, 
                method([]),  # 传入空列表作为all_potential_countries
                prompter_class(), 
                parser_class(),
                {
                    "original": input_text,
                    "ground_truth": "{}",  # 实际应用中应该提供正确结果
                    "current": "",
                    "phase": 0,
                    "method": method_name.lower()
                },
            )
            
            executor.run()
            
            # 获取最终结果
            final_thoughts = executor.get_final_thoughts()
            if final_thoughts and final_thoughts[0]:
                result = final_thoughts[0][0].state.get("current", "")
            else:
                result = "未获得结果"
            
            cost = lm.cost
            
            output = f"""关键词计数结果:

输入文本: {input_text[:200]}{'...' if len(input_text) > 200 else ''}
计数结果: {result}
使用成本: ${cost:.6f}
使用方法: {method_name}
执行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

详细信息:
- 输入长度: {len(input_text)} 个字符
- 模型: {model_name}
- 方法类型: {'直接输入输出' if method_name == 'Direct_Method' else '思维链' if method_name == 'CoT' else '思维树' if method_name in ['ToT', 'ToT2'] else '思维图'}"""
            
            return output, prompt_content
            
        finally:
            loop.close()
            
    except ValueError as e:
        error_msg = f"输入格式错误: {str(e)}。"
        return error_msg, "无提示词（格式错误）"
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"

# ==================== 集合交集功能 ====================
def get_set_intersection_modules(length):
    if length == 32:
        return (SetIntersectionPrompter32, SetIntersectionParser32, 
                {"Direct_Method": si_io32, "CoT": si_cot32, "ToT": si_tot32, "ToT2": si_tot232, "GoT": si_got32})
    elif length == 64:
        return (SetIntersectionPrompter64, SetIntersectionParser64, 
                {"Direct_Method": si_io64, "CoT": si_cot64, "ToT": si_tot64, "ToT2": si_tot264, "GoT": si_got64})
    elif length == 128:
        return (SetIntersectionPrompter128, SetIntersectionParser128, 
                {"Direct_Method": si_io128, "CoT": si_cot128, "ToT": si_tot128, "ToT2": si_tot2128, "GoT": si_got128})
    else:
        raise ValueError(f"不支持的长度 {length}。支持的长度：32, 64, 128")

def get_set_intersection_prompt_content(length, method_name, set1=None, set2=None, current=None, phase=None):
    if length == 32:
        prompter = SetIntersectionPrompter32()
    elif length == 64:
        prompter = SetIntersectionPrompter64()
    elif length == 128:
        prompter = SetIntersectionPrompter128()
    else:
        return "不支持的长度"
    
    method_lower = method_name.lower()
    
    # Prepare input parameters
    if set1 is None:
        set1 = "[集合1]"
    if set2 is None:
        set2 = "[集合2]"
    
    if method_lower == "direct_method" or method_lower == "io":
        return prompter.intersection_prompt.format(set1=set1, set2=set2)
    elif method_lower == "cot":
        return prompter.intersection_prompt_cot.format(set1=set1, set2=set2)
    elif method_lower in ["tot", "tot2"]:
        if current is None or current == "":
            # Initial phase uses regular intersection prompt
            return prompter.intersection_prompt.format(set1=set1, set2=set2)
        else:
            # Improvement phase uses ToT improve prompt
            return prompter.tot_improve_prompt.format(
                set1=set1,
                set2=set2,
                incorrect_intersection=current
            )
    elif method_lower == "got":
        if current is None or current == "":
            # Initial phase uses split prompt
            return prompter.got_split_prompt.format(input=set2)
        elif phase == 1:
            # Subset intersection phase uses regular intersection prompt
            return prompter.intersection_prompt.format(set1=set1, set2=current)
        else:
            # Merge improvement phase uses ToT improve prompt
            return prompter.tot_improve_prompt.format(
                set1=set1,
                set2=set2,
                incorrect_intersection=current
            )
    else:
        return "不支持的方法"

def parse_set_input(input_str):
    try:
        cleaned = input_str.strip().replace('[', '').replace(']', '').replace(' ', '')
        if not cleaned:
            return []
        return [int(x.strip()) for x in cleaned.split(",") if x.strip()]
    except ValueError:
        raise ValueError("输入格式错误，请确保输入的是用逗号分隔的数字")

def run_set_intersection(set1_str: str, set2_str: str, method_name: str, model_name: str = "deepseek") -> tuple:
    try:
        set1_nums = parse_set_input(set1_str)
        set2_nums = parse_set_input(set2_str)
        
        if len(set1_nums) not in [32, 64, 128] or len(set2_nums) not in [32, 64, 128]:
            error_msg = f"错误：每个集合的长度必须是32、64或128个数字。当前集合1有{len(set1_nums)}个数字，集合2有{len(set2_nums)}个数字。"
            return error_msg, "无提示词（输入错误）"
        
        if len(set1_nums) != len(set2_nums):
            error_msg = f"错误：两个集合的长度必须相同。当前集合1有{len(set1_nums)}个数字，集合2有{len(set2_nums)}个数字。"
            return error_msg, "无提示词（输入错误）"
        
        if not all(0 <= x <= 127 for x in set1_nums + set2_nums):
            error_msg = "错误：所有数字必须在0-127范围内。"
            return error_msg, "无提示词（输入错误）"
        
        # Handle IO/Direct_Method name conversion
        if method_name == "IO":
            method_name = "Direct_Method"
            
        try:
            prompter_class, parser_class, method_map = get_set_intersection_modules(len(set1_nums))
        except ValueError as e:
            return str(e), "无提示词（模块错误）"
        
        if method_name not in method_map:
            error_msg = f"错误：不支持的方法 '{method_name}'。支持的方法：IO, CoT, ToT, ToT2, GoT"
            return error_msg, "无提示词（方法错误）"
        
        method = method_map[method_name]
        prompt_content = get_set_intersection_prompt_content(
            length=len(set1_nums),
            method_name=method_name,
            set1=str(set1_nums),
            set2=str(set2_nums),
            current="",  # Initial empty current result
            phase=0      # Initial phase
        )
        
        config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts/language_models/config.json")
        if not os.path.exists(config_path):
            template_config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts", "language_models", "config_template.json")
            if os.path.exists(template_config_path):
                error_msg = "错误：请复制 config_template.json 为 config.json 并配置您的API密钥。"
                return error_msg, "无提示词（配置错误）"
            else:
                error_msg = "错误：找不到config.json配置文件。请确保配置文件存在。"
                return error_msg, "无提示词（配置错误）"
            
        # Create new event loop for async operations
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            operations_graph = method()
            lm = language_models.ChatGPT(config_path, model_name=model_name, cache=True)
            
            executor = controller.Controller(
                lm, operations_graph, prompter_class(), parser_class(),
                {"set1": str(set1_nums), "set2": str(set2_nums), "current": "", "phase": 0, "method": method_name.lower()},
            )
            
            executor.run()
            
            # Save execution graph to temp file
            temp_path = os.path.join(os.path.dirname(__file__), "temp.json")
            executor.output_graph(temp_path)
            
            # Read and parse the temp.json file
            with open(temp_path, 'r') as f:
                execution_data = json.load(f)
            
            # Extract the process details
            process_details = []
            for operation in execution_data:
                operation_name = operation.get("operation")
                if "thoughts" in operation:
                    for thought in operation["thoughts"]:
                        if "current" in thought:
                            process_details.append(f"操作 {operation_name}, 阶段 {thought.get('phase', '?')}: {thought['current']}")
            
            # Get the final result
            final_thoughts = executor.get_final_thoughts()
            if final_thoughts and final_thoughts[0]:
                result = final_thoughts[0][0].state.get("current", "")
            else:
                result = "未获得结果"
            
            set1_set = set(set1_nums)
            set2_set = set(set2_nums)
            correct_intersection = sorted(list(set1_set & set2_set))
            
            state_dict = {"set1": str(set1_nums), "set2": str(set2_nums), "current": result}
            error_count = set_intersection_utils.num_errors(state_dict)
            cost = lm.cost
            
            is_correct = set_intersection_utils.test_set_intersection({
                "result": str(correct_intersection),
                "current": result
            })
            
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
- 模型: {model_name}
- 缓存: 启用

执行过程:
"""
            output += "\n".join(process_details)
            
            return output, prompt_content
            
        finally:
            # Close the event loop
            loop.close()
        
    except ValueError as e:
        error_msg = f"输入格式错误: {str(e)}。请确保输入的是用逗号分隔的数字。"
        return error_msg, "无提示词（格式错误）"
    except FileNotFoundError as e:
        error_msg = f"配置文件错误: {str(e)}。请检查config.json文件是否存在。"
        return error_msg, "无提示词（文件错误）"
    except Exception as e:
        error_msg = f"运行出错: {str(e)}"
        return error_msg, "无提示词（运行错误）"


def get_available_models():
    """从config.json中获取可用的模型列表"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "graph_of_thoughts/language_models/config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        return list(config.keys())
    except Exception as e:
        print(f"获取可用模型失败: {str(e)}")
        return ["deepseek-v3", "chatgpt", "chatgpt4"]

# ==================== 实验结果绘图功能 ====================
def get_available_result_files():
    """从文件目录中自动获取可用的实验结果文件"""
    result_files = {
        "文档合并": [],
        "关键词计数": [],
        "集合交集": [],
        "数字排序": []
    }
    
    # 定义各实验类型的目录路径
    result_dirs = {
        "文档合并": "examples/doc_merge/results",
        "关键词计数": "examples/keyword_counting/results",
        "集合交集": "examples/set_intersection/results",
        "数字排序": "examples/sorting/results"
    }
    
    # 扫描每个目录，收集结果文件
    for task_type, result_dir in result_dirs.items():
        try:
            if os.path.exists(result_dir):
                # 获取目录下所有文件夹（每个文件夹代表一个实验结果集）
                for dir_name in os.listdir(result_dir):
                    dir_path = os.path.join(result_dir, dir_name)
                    if os.path.isdir(dir_path):
                        # 检查文件夹中是否有.json文件（表示有效的结果）
                        if any(fname.endswith('.json') for fname in os.listdir(dir_path)):
                            result_files[task_type].append(dir_name)
        except Exception as e:
            print(f"扫描{task_type}结果目录时出错: {str(e)}")
    
    # 按时间倒序排序结果文件（最新的在前）
    for task_type in result_files:
        result_files[task_type].sort(reverse=True)
    
    return result_files

def update_result_files(task_type):
    """更新结果文件下拉框选项"""
    result_files = get_available_result_files()
    available_files = result_files.get(task_type, [])
    
    # 返回更新后的下拉框属性和默认值
    return gr.Dropdown(
        choices=available_files,
        value=None,
        label="选择结果文件"
    )

def plot_selected_result(task_type: str, result_file: str):
    """绘制选定的实验结果"""
    try:
        # 根据任务类型确定结果目录
        if task_type == "文档合并":
            dir_path = f"examples/doc_merge/results/{result_file}"
        elif task_type == "关键词计数":
            dir_path = f"examples/keyword_counting/results/{result_file}"
        elif task_type == "集合交集":
            dir_path = f"examples/set_intersection/results/{result_file}"
        elif task_type == "数字排序":
            dir_path = f"examples/sorting/results/{result_file}"
        else:
            return None, "错误：未知的任务类型"
        
        # 检查目录是否存在
        if not os.path.exists(dir_path):
            return None, f"错误：找不到结果目录 {dir_path}"
        

        
        # 根据任务类型设置不同的绘图参数
        if task_type == "文档合并":
            plotting_data = doc_merge_plot.get_plotting_data(dir_path)
            fig = doc_merge_plot.plot_results(
                plotting_data,
                methods_order=["direct_method", "cot", "tot", "got", "got2"],
                model="gradio_demo",
                num_ndas=4,
                display_solved=False,
                y_upper=10,
                display_left_ylabel=True,
                display_right_ylabel=True,
                cost_upper=15,
            )
        elif task_type == "关键词计数":
            plotting_data = keyword_counting_plot.get_plotting_data(dir_path)
            fig = keyword_counting_plot.plot_results(
                plotting_data,
                methods_order=["direct_method", "cot", "tot", "tot2", "got4", "got8", "gotx"],
                model="gradio_demo",
                display_solved=True,
                annotation_offset=-0.3,
                y_upper=35,
                display_left_ylabel=True,
                display_right_ylabel=True,
                cost_upper=9,
            )
        elif task_type == "集合交集":
            plotting_data = set_intersection_plot.get_plotting_data(dir_path)
            fig = set_intersection_plot.plot_results(
                plotting_data,
                methods_order=["direct_method", "cot", "tot", "tot2", "got"],
                model="gradio_demo",
                length=32,
                display_solved=True,
                display_left_ylabel=True,
                display_right_ylabel=True,
            )
        elif task_type == "数字排序":
            plotting_data = sorting_plot.get_plotting_data(dir_path)
            fig = sorting_plot.plot_results(
                plotting_data,
                methods_order=["direct_method", "cot", "tot", "tot2", "got"],
                model="gradio_demo",
                length=32,
                display_solved=True,
                display_left_ylabel=True,
                display_right_ylabel=True,
            )
        
        # 保存临时图片用于显示
        temp_path = "temp_plot.png"
        
        return temp_path, "绘图成功"
    
    except Exception as e:
        return None, f"绘图出错: {str(e)}"
# ==================== Gradio界面 ====================
with gr.Blocks(title="Graph of Thoughts 多功能演示") as demo:
    gr.Markdown("# Graph of Thoughts 多功能演示")
    gr.Markdown("使用不同的思维方法解决多种问题，并显示使用的提示词")
    
    with gr.Tabs():
        # 排序标签页
        with gr.TabItem("数字排序"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入区域")
                    sorting_input = gr.Textbox(
                        label="输入数字（用逗号分隔）",
                        placeholder="例如: 5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0",
                        info="请输入32、64或128个0-9之间的数字，用逗号分隔",
                        lines=8
                    )
                    sorting_method = gr.Dropdown(
                        choices=["IO", "CoT", "ToT", "ToT2", "GoT"], 
                        value="", 
                        label="选择排序方法",
                        info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT: 思维图"
                    )
                    model_selector = gr.Dropdown(
                        choices=get_available_models(),
                        value="",
                        label="选择模型",
                        info="从config.json中选择要使用的模型"
                    )
                    sorting_run = gr.Button("开始排序", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 排序结果")
                    sorting_result = gr.Textbox(
                        label="排序结果",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
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
                        inputs=[sorting_input, sorting_method],
                        label="点击示例快速填充"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 使用的提示词")
                    sorting_prompt = gr.Textbox(
                        label="提示词内容",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            sorting_run.click(
                fn=run_sorting,
                inputs=[sorting_input, sorting_method, model_selector],
                outputs=[sorting_result, sorting_prompt]
            )

        # 文档合并标签页
        with gr.TabItem("文档合并"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入区域")
                    doc_sample = gr.Dropdown(
                        choices=[f"样本 {i}" for i in get_available_samples()],
                        value="",
                        label="选择示例文档",
                        info="选择预定义的示例文档集"
                    )
                    doc_method = gr.Dropdown(
                        choices=["IO", "CoT", "ToT", "GoT", "GoT2"], 
                        value="", 
                        label="选择合并方法",
                        info="IO: 直接合并, CoT: 思维链, ToT: 思维树, GoT: 思维图, GoT2: 改进思维图"
                    )
                    doc_custom = gr.Textbox(
                        lines=8,
                        label="自定义文档（可选）",
                        placeholder="输入多个文档，每行一个文档，或使用JSON格式的数组。留空则使用示例文档。",
                        info="支持文本格式（每行一个文档）或JSON格式"
                    )
                    model_selector = gr.Dropdown(
                        choices=get_available_models(),
                        value="",
                        label="选择模型",
                        info="从config.json中选择要使用的模型"
                    )
                    doc_run = gr.Button("开始合并", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 合并结果")
                    doc_result = gr.Textbox(
                        label="合并结果",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 快速示例")
                    gr.Examples(
                        examples=[
                            ["样本 0", "IO"],
                            ["样本 1", "CoT"],
                            ["样本 2", "ToT"],
                            ["样本 3", "GoT"],
                            ["样本 4", "GoT2"],
                        ],
                        inputs=[doc_sample, doc_method],
                        label="点击示例快速填充"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 使用的提示词")
                    doc_prompt = gr.Textbox(
                        label="提示词内容",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            doc_run.click(
                fn=run_doc_merge,
                inputs=[doc_method, doc_sample, doc_custom, model_selector],
                outputs=[doc_result, doc_prompt]
            )

        # 关键词计数标签页
        with gr.TabItem("关键词计数"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入区域")
                    kw_input = gr.Textbox(
                        label="输入文本",
                        placeholder="例如: The music of Spain and the history of Spain deepened her love for Europe. The Nordic beauty of Norway, Sweden, Finland, and Denmark took her breath away. She danced in Ireland, explored castles in Scotland, and marveled at the architecture in Germany and Russia.",
                        info="请输入包含国家名称的文本",
                        lines=8
                    )
                    kw_method = gr.Dropdown(
                        choices=["IO", "CoT", "ToT", "ToT2", "GoT4", "GoT8", "GoTx"], 
                        value="", 
                        label="选择计数方法",
                        info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT4/8/x: 思维图"
                    )
                    model_selector = gr.Dropdown(
                        choices=get_available_models(),
                        value="",
                        label="选择模型",
                        info="从config.json中选择要使用的模型"
                    )
                    kw_run = gr.Button("开始计数", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 计数结果")
                    kw_result = gr.Textbox(
                        label="计数结果",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
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
                        inputs=[kw_input, kw_method],
                        label="点击示例快速填充"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 使用的提示词")
                    kw_prompt = gr.Textbox(
                        label="提示词内容",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            kw_run.click(
                fn=run_keyword_counting,
                inputs=[kw_input, kw_method, model_selector],
                outputs=[kw_result, kw_prompt]
            )

        # 集合交集标签页
        with gr.TabItem("集合交集"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 输入区域")
                    set1_input = gr.Textbox(
                        label="集合1",
                        placeholder="例如: 13,16,30,6,21,7,31,15,11,1,24,10,9,3,20,8",
                        info="请输入32、64或128个0-127之间的数字，用逗号分隔",
                        lines=6
                    )
                    set2_input = gr.Textbox(
                        label="集合2",
                        placeholder="例如: 25,24,10,4,27,0,14,12,8,2,29,20,17,19,26,23",
                        info="请输入与集合1相同数量的0-127之间的数字，用逗号分隔",
                        lines=6
                    )
                    set_method = gr.Dropdown(
                        choices=["IO", "CoT", "ToT", "ToT2", "GoT"], 
                        value="", 
                        label="选择集合交集方法",
                        info="IO: 直接输入输出, CoT: 思维链, ToT: 思维树, ToT2: 改进思维树, GoT: 思维图"
                    )
                    model_selector = gr.Dropdown(
                        choices=get_available_models(),
                        value="",
                        label="选择模型",
                        info="从config.json中选择要使用的模型"
                    )
                    set_run = gr.Button("计算交集", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 交集结果")
                    set_result = gr.Textbox(
                        label="交集计算结果",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 快速示例")
                    gr.Examples(
                        examples=[
                            ["28,58,36,18,37,31,44,39,34,51,12,56,21,27,7,24,46,1,25,2,41,6,45,29,49,42,35,30,54,55,10,50","4,16,28,46,49,21,58,30,19,1,37,15,3,59,51,24,10,12,34,20,40,44,35,23,36,0,43,54,2,31,57,41","CoT"],
                            ["115,61,35,103,90,117,86,44,63,45,40,30,74,33,31,1,118,48,38,0,119,51,64,78,15,121,89,101,79,69,120,29", "13,35,20,96,34,18,47,127,126,9,21,16,77,22,111,122,85,73,42,105,123,15,33,59,67,57,104,8,30,89,76,12", "ToT"],
                            ["5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0,5,1,0,1,2,0,4,8,1,9,5,1,3,3,9,7,2,4,6,8,0,2,1,5,7,3,9,4,6,8,1,0", "3,7,0,2,8,1,2,2,2,4,7,8,5,5,3,9,4,3,5,6,6,4,4,5,2,0,9,3,3,9,2,1,3,7,0,2,8,1,2,2,2,4,7,8,5,5,3,9,4,3,5,6,6,4,4,5,2,0,9,3,3,9,2,1", "ToT2"],
                            ["6,4,5,7,5,6,9,7,6,9,4,6,9,8,1,9,2,4,9,0,7,6,5,6,6,2,8,3,9,5,6,1,6,4,5,7,5,6,9,7,6,9,4,6,9,8,1,9,2,4,9,0,7,6,5,6,6,2,8,3,9,5,6,1", "8,2,1,3,7,4,0,9,5,1,2,6,8,3,4,7,9,0,1,5,2,8,6,3,4,7,9,1,0,5,2,8,8,2,1,3,7,4,0,9,5,1,2,6,8,3,4,7,9,0,1,5,2,8,6,3,4,7,9,1,0,5,2,8", "GoT"],
                        ],
                        inputs=[set1_input, set2_input, set_method],
                        label="点击示例快速填充"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 使用的提示词")
                    set_prompt = gr.Textbox(
                        label="提示词内容",
                        lines=12,
                        max_lines=15,
                        interactive=False
                    )
            
            set_run.click(
                fn=run_set_intersection,
                inputs=[set1_input, set2_input, set_method, model_selector],
                outputs=[set_result, set_prompt]
            )
        # 在现有的with gr.Tabs()块中添加以下内容
        with gr.TabItem("实验结果绘图"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 选择实验和结果")
                    plot_task_type = gr.Dropdown(
                        choices=["文档合并", "关键词计数", "集合交集", "数字排序"],
                        value="",
                        label="选择实验类型"
                    )
                    plot_result_file = gr.Dropdown(
                        label="选择结果文件",
                        interactive=True
                    )
                    plot_button = gr.Button("绘制结果", variant="primary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 实验结果可视化")
                    plot_output = gr.Image(label="绘图结果", interactive=False)
                    plot_status = gr.Textbox(label="绘图状态", interactive=False)
            
            # 设置交互逻辑
            plot_task_type.change(
                fn=update_result_files,
                inputs=plot_task_type,
                outputs=plot_result_file
            )
            
            plot_button.click(
                fn=plot_selected_result,
                inputs=[plot_task_type, plot_result_file],
                outputs=[plot_output, plot_status]
            )

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("启动Graph of Thoughts多功能演示...")
    print("请确保config.json文件存在并包含正确的API配置")
    
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7862, 
        share=False,
        show_error=True
    )
