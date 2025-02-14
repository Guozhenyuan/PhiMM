import re
import os
import sys

from tqdm import tqdm
from fraction import Fraction

# humaneval
import numpy as np
from human_eval.data import HUMAN_EVAL, read_problems
from human_eval.execution import check_correctness
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def extract_answer_number(completion, split_by='Final answer:'):
    # text = completion.split('The answer is: ')
    text = completion.split(split_by)
    if len(text) > 1:
        extract_ans = text[1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', '')) # type: ignore
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None
# def extract_answer_number(text):
#     regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
#     regexes_to_ignore =[
#         ",",
#         "\\$",
#         "(?s).*#### ",
#         "\\.$"
#     ]
#     match = re.findall(regex_pattern, text)
#     if match:
#         match = match[-1]
#         if isinstance(match, tuple):
#             match = [m for m in match if m][0]
#         text = match.strip()

#         for regex in regexes_to_ignore:
#             text = re.sub(regex, "", text)
#         return text
#     else:
#         return None

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data

def get_results(llm,batch_text,sampling_params,chat=True):
    results = []
    for prompt in tqdm(batch_text,desc='Inferencing'):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        if chat==True:
            completions = llm.chat(prompt,sampling_params,use_tqdm=False)
        else:
            completions = llm.generate(prompt,sampling_params,use_tqdm=False)
        # import pdb;pdb.set_trace()
        for output in completions:
            generate_text = output.outputs[0].text
            results.append(generate_text)
    return results

def get_path(args):
    if args.checkpoint == True:
        name = args.model.split('/')[-2] + '_' + args.model.split('/')[-1] + '.json'
    else:
        name = args.model.split('/')[-1] + '.json'
    os.makedirs(args.save_dir,exist_ok=True)
    path = os.path.join(args.save_dir, name)
    return path

def nb_exists(name,base,dataset,json_file):
    # 查看模型的name，base是否存在json文件中
    # 默认json是一个列表
    flag = False
    index = 0
    for id,jf in enumerate(json_file):
        jf_name=jf['model_name']
        jf_base=jf['model_base']
        jf_dataset = jf['dataset']
        if name==jf_name and base==jf_base and dataset==jf_dataset:
            flag=True
            index=id
    return flag,index


'''--------- humaneval function ---------'''

# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def evaluate_functional_correctness(
    sample_file: str,
    k: List[int] = [1, 5, 10, 100],
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        # for sample in tqdm.tqdm(stream_jsonl(sample_file)):
        for sample in tqdm(sample_file):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    return pass_at_k