import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import re
import logging
import datetime
import json
import csv
from statistics import fmean
import threading
from typing import Dict, List, Callable, Set, Union

from tqdm import tqdm
from graph_of_thoughts import controller, language_models, operations, prompter, parser


class DocMergePrompter(prompter.Prompter):
    """
    DocMergePrompter provides the generation of prompts specific to the document
    merge example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    merge_doc_prompt_start = """Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, maximizing retained information and minimizing redundancy. Output only the created NDA between the tags <Merged> and </Merged>, without any additional text.
Here are NDAs <Doc1> - <Doc{num}>
"""
    merge_doc_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    merge_doc_prompt_cot_start = """Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, maximizing retained information and minimizing redundancy.
You can generate any intermediate thoughts and documents you want, but the final output should be the merged NDA, placed between the two tags <Merged> and </Merged>.
For instance you might want to follow this approach:
1. Split each NDA into their logical subparts.
2. Merge the subparts of the {num} NDAs.
3. Combine the merged subparts into a single NDA.
4. Place the merged NDA between the tags <Merged> and </Merged>.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    improve_summary_prompt_start = """The following NDA <S> merges initial NDAs <Doc1> - <Doc{num}>.
Please improve the summary NDA <S> by adding more information and removing redundancy. Output only the improved NDA, placed between the two tags <Merged> and </Merged>, without any additional text.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    improve_summary_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    improve_summary_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    score_prompt_base = """The following NDA <S> merges NDAs <Doc1> - <Doc{num}>.
Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    score_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    score_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    aggregate_full_prompt_base = """The following NDAs <S1> - <S{num_ndas_summary}> each merge the initial NDAs <Doc1> - <Doc{num_ndas}>.
Combine the merged NDAs <S1> - <S{num_ndas_summary}> into a new one, maximizing their advantages and overall information retention, while minimizing redundancy.
Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.   

Here are the original NDAs <Doc1> - <Doc{num_ndas}>:
"""

    aggregate_full_prompt_block1 = """
<Doc{num}>
{document}
</Doc{num}>
"""
    aggregate_full_prompt_mid = """
Here are the summary NDAs <S1> - <S{num_ndas_summary}>:
"""

    aggregate_full_prompt_block2 = """
<S{num}>
{summary}
</S{num}>
"""

    aggregate_sub_prompt_base = """The following NDAs <S1> - <S{num_ndas}> are summaries of some other NDAs.
Combine them into a new one, make sure to maximize their advantages and overall information retention, while minimizing redundancy.
Output only the new NDA between the tags <Merged> and </Merged>, without any additional text.

Here are NDAs <S1> - <S{num_ndas}>:
"""

    aggregate_sub_prompt_generate = """
NDA <S{num}>:
{nda}
</S{num}>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """

        if len(state_dicts[0]["parts"]) > 0 and len(state_dicts[0]["parts"]) < len(
            state_dicts[0]["documents"]
        ):
            prompt = self.aggregate_sub_prompt_base.format(
                num_ndas=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_sub_prompt_generate.format(
                    nda=state_dict["current"], num=i + 1
                )
            return prompt
        else:
            prompt = self.aggregate_full_prompt_base.format(
                num_ndas=len(state_dicts[0]["documents"]),
                num_ndas_summary=len(state_dicts),
            )
            for i, document in enumerate(state_dicts[0]["documents"]):
                prompt += self.aggregate_full_prompt_block1.format(
                    document=document, num=i + 1
                )
            prompt += self.aggregate_full_prompt_mid.format(
                num_ndas_summary=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_full_prompt_block2.format(
                    summary=state_dict["current"], num=i + 1
                )
            return prompt

    def generate_prompt(
        self,
        num_branches: int,
        documents: List[str],
        method: str,
        parts: Set[str],
        current: str,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param documents: The list of documents to be merged.
        :type documents: List[str]
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param parts: Indices of the already processed document parts.
        :type parts: Set[str]
        :param current: The intermediate solution.
        :type current: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """

        prompt = ""
        if method.startswith("direct_method") or method.startswith("cot"):
            if method.startswith("direct_method"):
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
            else:
                prompt += self.merge_doc_prompt_cot_start.format(num=len(documents))
            for i, document in enumerate(documents):
                prompt += self.merge_doc_prompt_block.format(
                    document=document, num=i + 1
                )
            return prompt
        elif method.startswith("tot"):
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
                for i, document in enumerate(documents):
                    prompt += self.merge_doc_prompt_block.format(
                        document=document, num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(documents),
                )
                for i, document in enumerate(documents):
                    prompt += self.improve_summary_prompt_block.format(
                        document=document, num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        elif method.startswith("got"):
            parts = (
                sorted(list(parts)) if len(parts) > 0 else list(range(len(documents)))
            )
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(parts))
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.merge_doc_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(parts),
                )
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.improve_summary_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        else:
            assert False, "Not implemented yet."

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """

        if len(state_dicts) > 1:
            assert False, "Not implemented yet."
        else:
            # perform individual scoring
            parts = (
                [
                    state_dicts[0]["documents"][part]
                    for part in sorted(list(state_dicts[0]["parts"]))
                ]
                if len(state_dicts[0]["parts"]) > 0
                else state_dicts[0]["documents"]
            )
            prompt = self.score_prompt_base.format(
                num=len(parts),
            )
            for i, part in enumerate(parts):
                prompt += self.score_prompt_block.format(document=part, num=i + 1)
            prompt += self.score_prompt_end.format(
                summary=state_dicts[0]["current"],
            )
            return prompt

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass


class DocMergeParser(parser.Parser):
    """
    DocMergeParser provides the parsing of language model reponses specific to the
    document merge example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        """

        new_states = []
        for text in texts:
            if len(states[0]["parts"]) < len(states[0]["documents"]):
                # subpart aggregation
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_state["parts"] = set()
                for state in states:
                    new_state["parts"] = new_state["parts"] | state["parts"]

                new_states.append(new_state)
            else:
                # full NDA aggregation
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Merged")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        assert len(states) == 1, "Only one state is allowed for scoring."
        if len(states) == 1:
            # individual scoring
            redundancy_scores = []
            retain_scores = []
            for text in texts:
                answer = self.strip_answer_helper(text, "Redundancy")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    redundancy_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple redundancy scores in answer: {text}. Returning the last one."
                    )
                    redundancy_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any redundancy score in answer: {text}. Ignoring this answer."
                    )
                answer = self.strip_answer_helper(text, "Retained")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    retain_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        f"Found multiple retained scores in answer: {text}. Returning the last one."
                    )
                    retain_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        f"Could not find any retained score in answer: {text}. Ignoring this answer."
                    )
            if len(redundancy_scores) == 0 or len(retain_scores) == 0:
                logging.warning(
                    f"Could not find any valid score in any answer. Returning 0.0."
                )
                return [0.0]
            mean_redundancy = fmean(redundancy_scores)
            mean_retain = fmean(retain_scores)
            f1 = 2 * mean_redundancy * mean_retain / (mean_redundancy + mean_retain)
            return [f1]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        pass

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

class ProgressTracker:
    def __init__(self, methods: List[Callable], total_samples: int):
        self.methods = methods
        self.total_samples = total_samples
        self.progress_bars = {}
        self.active_tasks = {}
        self.completed_samples = {}
        self.lock = threading.Lock()
        
    def initialize_progress_bars(self):
        for method in self.methods:
            method_name = method.__name__
            self.progress_bars[method_name] = tqdm(
                total=self.total_samples,
                desc=f"{method_name: <12}",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ({elapsed}/{remaining}) [Active: {postfix[0][active]:<3}]',
                postfix=[{'active': 0}]
            )
            self.active_tasks[method_name] = 0
            self.completed_samples[method_name] = 0
            
    def update_progress(self, method_name: str):
        with self.lock:
            if method_name in self.progress_bars:
                self.completed_samples[method_name] += 1
                self.progress_bars[method_name].n = self.completed_samples[method_name]
                self._update_active_display(method_name)
                self.progress_bars[method_name].refresh()
                
    def update_active_tasks(self, method_name: str, delta: int):
        with self.lock:
            if method_name in self.active_tasks:
                self.active_tasks[method_name] += delta
                self.active_tasks[method_name] = max(0, self.active_tasks[method_name])
                self._update_active_display(method_name)
                
    def _update_active_display(self, method_name: str):
        if method_name in self.progress_bars:
            self.progress_bars[method_name].postfix[0]['active'] = self.active_tasks[method_name]
            self.progress_bars[method_name].refresh()
            
    def close_all(self):
        for bar in self.progress_bars.values():
            bar.close()

def direct_method() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 10

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best_1 = operations.KeepBestN(1, True)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, branch_factor))
        operations_graph.append_operation(operations.Score(3, False))
        keep_best_2 = operations.KeepBestN(1, True)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where full documents
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.Aggregate(5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)

    return operations_graph


def got2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT2 method, where partial
    documents are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    sub_parts = []
    for i in range(0, 4, 2):  # should be at most 16 parts
        sub_text = operations.Selector(
            lambda thoughts, list_id=i: [
                operations.Thought(
                    state={**thoughts[0].state, "parts": {list_id, list_id + 1}}
                )
            ]
        )
        operations_graph.add_operation(sub_text)
        gen_nda = operations.Generate(1, 5)
        gen_nda.add_predecessor(sub_text)
        operations_graph.add_operation(gen_nda)
        score_nda = operations.Score(3, False)
        score_nda.add_predecessor(gen_nda)
        operations_graph.add_operation(score_nda)
        keep_best_nda = operations.KeepBestN(1, True)
        keep_best_nda.add_predecessor(score_nda)
        operations_graph.add_operation(keep_best_nda)

        sub_parts.append(keep_best_nda)

    while len(sub_parts) > 1:
        new_sub_parts = []
        for i in range(0, len(sub_parts), 2):
            if i + 1 == len(sub_parts):
                new_sub_parts.append(sub_parts[i])
                continue
            aggregate = operations.Aggregate(5)
            aggregate.add_predecessor(sub_parts[i])
            aggregate.add_predecessor(sub_parts[i + 1])
            operations_graph.add_operation(aggregate)
            score = operations.Score(3, False)
            score.add_predecessor(aggregate)
            operations_graph.add_operation(score)
            keep_best = operations.KeepBestN(1, True)
            keep_best.add_predecessor(score)
            operations_graph.add_operation(keep_best)

            gen_nda = operations.Generate(1, 5)
            gen_nda.add_predecessor(keep_best)
            operations_graph.add_operation(gen_nda)
            score_nda = operations.Score(3, False)
            score_nda.add_predecessor(gen_nda)
            operations_graph.add_operation(score_nda)
            keep_best_nda = operations.KeepBestN(1, True)
            keep_best_nda.add_predecessor(score_nda)
            keep_best_nda.add_predecessor(keep_best)
            operations_graph.add_operation(keep_best_nda)

            new_sub_parts.append(keep_best_nda)
        sub_parts = new_sub_parts

    return operations_graph


def run_sample_sync(data, method, results_folder, lm_name, progress_tracker):
    try:
        progress_tracker.update_active_tasks(method.__name__, 1)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        lm = language_models.ChatGPT(
            os.path.join(
                os.path.dirname(__file__),
                "../../graph_of_thoughts/language_models/config.json",
            ),
            model_name=lm_name,
            cache=True,
        )
        operations_graph = method()
        executor = controller.Controller(
            lm,
            operations_graph,
            DocMergePrompter(),
            DocMergeParser(),
            {
                "documents": [data[2], data[3], data[4], data[5]],
                "parts": set(),
                "current": "",
                "method": method.__name__,
            },
        )
        executor.run()
        path = os.path.join(
            results_folder,
            method.__name__,
            f"{data[0]}.json",
        )
        for operation in operations_graph.operations:
            for thought in operation.thoughts:
                if "parts" in thought.state:
                    thought.state["parts"] = list(thought.state["parts"])
        executor.output_graph(path)
        
        progress_tracker.update_progress(method.__name__)
        return lm.cost
        
    except Exception as e:
        logging.error(f"Exception in {method.__name__} for data {data[0]}: {e}")
        return 0
    finally:
        progress_tracker.update_active_tasks(method.__name__, -1)
        loop.close()

async def run_method(method, selected_data, results_folder, lm_name, budget_lock, remaining_budget, executor, progress_tracker):
    method_dir = os.path.join(results_folder, method.__name__)
    os.makedirs(method_dir, exist_ok=True)
    
    tasks = []
    for data in selected_data:
        task = run_sample_async(
            data, method, results_folder, lm_name, 
            budget_lock, remaining_budget, executor, 
            progress_tracker
        )
        tasks.append(task)
    
    semaphore = asyncio.Semaphore(THREADS)  # 限制并发数
    
    async def limited_task(task):
        async with semaphore:
            return await task
    
    results = await asyncio.gather(*[limited_task(t) for t in tasks], return_exceptions=True)
    
    total_cost = sum(cost for cost in results if isinstance(cost, (int, float)))
    return total_cost

async def run_sample_async(data, method, results_folder, lm_name, budget_lock, remaining_budget, executor, progress_tracker):
    try:
        async with budget_lock:
            if remaining_budget[0] <= 0.0:
                return 0
        
        loop = asyncio.get_running_loop()
        cost = await loop.run_in_executor(
            executor,
            run_sample_sync,
            data, method, results_folder, lm_name, progress_tracker
        )
        
        async with budget_lock:
            remaining_budget[0] -= cost
        
        return cost
        
    except Exception as e:
        logging.error(f"Exception in {method.__name__} for data {data[0]}: {e}")
        return 0

async def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    orig_budget = budget
    remaining_budget = [budget]
    
    progress_tracker = ProgressTracker(methods, len(data_ids) if data_ids else 50)
    progress_tracker.initialize_progress_bars()
    
    max_workers = len(methods) * THREADS
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    data_path = os.path.join(os.path.dirname(__file__), "documents.csv")
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row[0] = int(row[0])
            data.append(row)

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    budget_lock = asyncio.Lock()
    
    method_tasks = []
    for method in methods:
        task = asyncio.create_task(
            run_method(method, selected_data, results_folder, lm_name, budget_lock, remaining_budget, executor, progress_tracker)
        )
        method_tasks.append(task)
    
    method_costs = await asyncio.gather(*method_tasks)
    
    executor.shutdown(wait=True)
    
    progress_tracker.close_all()
    
    total_cost = sum(method_costs)
    return total_cost

if __name__ == "__main__":
    """
    Input (x1, x2, x3, x4): Four NDAs
    Output (y): A new combined NDA
    Evaluation: According to information coverage without repetition (scored by the LLM)
    """
if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(0, 50)]
    approaches = [direct_method, cot, tot, got, got2]
    THREADS = 5

    try:
        spent = asyncio.run(run(samples, approaches, budget, "deepseek-vocano"))
        logging.info(f"Spent {spent} out of {budget} budget.")
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        raise e
