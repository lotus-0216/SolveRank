import json
import os
from tqdm import tqdm
import collections
import matplotlib.pyplot as plt
import numpy as np
import random

def read_json_or_jsonl(file_path):
    data = []
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'r') as f_r:
            data = json.load(f_r)
    elif txt == 'jsonl':
        with open(file_path, 'r') as f_r:
            for line in f_r:
                data.append(json.loads(line))
    else:
        data = None
    return data

def write_json_or_jsonl(file_path, data, encoding='utf-8'):
    txt = file_path.split('/')[-1].split('.')[-1]
    if txt == 'json':
        with open(file_path, 'w', encoding=encoding) as f_w:
            json.dump(data, f_w, ensure_ascii=False, indent=2)
    elif txt == 'jsonl':
        with open(file_path, 'w', encoding=encoding) as f_w:
            for line in data:
                json_line = json.dumps(line)
                f_w.write(json_line + '\n')
    else:
        print("file_path not exisits")
    # return data
    
def split_jsonl_file(input_path, output_dir, num_parts=20):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    lines_per_part = total_lines // num_parts
    remainder = total_lines % num_parts

    start = 0
    for i in range(num_parts):
        end = start + lines_per_part + (1 if i < remainder else 0)
        part_lines = lines[start:end]
        output_path = os.path.join(output_dir, f"data_part_{i}.jsonl")
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(part_lines)
        print(f"Wrote {len(part_lines)} lines to {output_path}")
        start = end

def merge_split(file_path, file_save_path):
    file_list = []
    for file_name in os.listdir(file_path):
        file_list.append(os.path.join(file_path, file_name))
    data = []
    for f_r in file_list:
        with open(f_r, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    write_json_or_jsonl(file_save_path, data)
    
def save_all_reviewed_code(file_path, file_list_path, file_save_path):
    count = 0
    data_list = read_json_or_jsonl(file_list_path)
    data = read_json_or_jsonl(file_path)
    save_data = []
    for item in tqdm(data):
        save_item_retrieved = []
        save_item = item
        for retrieved in item['retrieve']:
            save_code = []
            src_uid = retrieved['src_uid']
            source_data = next((sample for sample in data_list if sample.get("src_uid") == src_uid), None)
            if source_data and len(source_data['positive_code']) >= 1:
                for code in source_data['positive_code']:
                    save_code.append(code['source_code'])
                retrieved['solution_code'] = save_code
            if len(save_code) >= 1:
                save_item_retrieved.append(retrieved)
                count += len(save_code)
        if len(save_item_retrieved) >= 1:
            item['retrieve'] = save_item_retrieved
            save_data.append(item)
    stop = input(count)
    write_json_or_jsonl(file_save_path, save_data)
    
def save_one_reviewed_code(file_path, file_list_path, file_save_path):
    data_list = read_json_or_jsonl(file_list_path)
    data = read_json_or_jsonl(file_path)
    save_data = []
    for item in tqdm(data):
        save_item_retrieved = []
        save_item = item
        for retrieved in item['retrieve']:
            src_uid = retrieved['src_uid']
            source_data = next((sample for sample in data_list if sample.get("src_uid") == src_uid), None)
            if source_data and len(source_data['positive_code']) >= 1:
                retrieved['solution_code'] = source_data['positive_code'][0]
                save_item_retrieved.append(retrieved)
        if len(save_item_retrieved) >= 1:
            item['retrieve'] = save_item_retrieved
            save_data.append(item)
    write_json_or_jsonl(file_save_path, save_data)
    
def extract_sample_question_same(file_path, file_list_path, save_path):
    data_list = read_json_or_jsonl(file_list_path)
    data = read_json_or_jsonl(file_path)
    save_data = []
    src_list = []
    for item in data_list:
        src_list.append(item['src_uid'])
    for item in data:
        if item['src_uid'] in src_list:
            save_data.append(item)
            src_list.remove(item['src_uid'])
    # print(len(src_list))
    # stop = input()
    for item in data_list:
        if item['src_uid'] in src_list:
            save_data.append(item)
    write_json_or_jsonl(save_path, save_data)
    
def sanitize_code(code):
    while code.startswith("```python") or "```" in code:
        code = code.replace("```python", "")
        code = code.replace("```", "")
    return code.strip()
    
def extract_true_code(file_path, 
                      src_uid_list=[]):
    true_code = []
    data = read_json_or_jsonl(file_path)
    for item in data:
        uid = item['source_data']['src_uid']
        result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
        if not uid in src_uid_list and result == True:
            code = sanitize_code(item['oai_response']['choices'][0]['message']['content'])
            src_uid_list.append(uid)
            true_code.append({
                'src_uid': uid,
                'solution_code': code
            })
    return true_code, src_uid_list

def change_solution_code_for_test(file_path_list, file_path, save_path):
    src_uid_list = []
    true_code_list = []
    for f in file_path_list:
        true_code, src_uid_list = extract_true_code(f, src_uid_list)
        true_code_list += true_code
    data_save = []
    data = read_json_or_jsonl(file_path)
    for item in data:
        src_uid = item['src_uid']
        source_data = next((sample for sample in true_code_list if sample.get("src_uid") == src_uid), None)
        if source_data:
            for i in range(len(item['retrieve'])):
                item['retrieve'][i]["solution_code"] = source_data['solution_code']
            data_save.append(item)
    write_json_or_jsonl(save_path, data_save)
  
def extract_difficulty(file_path):
    difficulty = []
    data = read_json_or_jsonl(file_path)
    for item in data:
        difficulty.append(item['difficulty'])
    sorted_difficulty = sorted(difficulty)
    print(sorted_difficulty)
    
def show_difficulty_pass(file_path):
    diff_all = collections.defaultdict(int)
    diff_correct = collections.defaultdict(int)
    diff_pass = collections.defaultdict()
    data = read_json_or_jsonl(file_path)
    for item in data:
        diff_all[item['source_data']['difficulty']] += 1
        result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
        if result == True:
            diff_correct[item['source_data']['difficulty']] += 1
    for item in diff_all:
        if item in diff_correct:
            diff_pass[item] = float(diff_correct[item] / diff_all[item])
        elif item is not None:
            diff_pass[item] = 0.0
    return diff_pass

def save_difficulty_pass(file_path1, file_path2, file_name):
    diff_pass1 = show_difficulty_pass(file_path1)
    diff_pass2 = show_difficulty_pass(file_path2)

    
    all_difficulties = sorted(set(diff_pass1.keys()).union(diff_pass2.keys()))
    filtered_difficulties = []
    for d in all_difficulties:
        v1 = diff_pass1.get(d, 0.0)
        v2 = diff_pass2.get(d, 0.0)
        if not (v1 == 0.0 and v2 == 0.0):
            filtered_difficulties.append(d)
    # 对齐数据
    y1 = [diff_pass1.get(d, 0.0) for d in filtered_difficulties]
    y2 = [diff_pass2.get(d, 0.0) for d in filtered_difficulties]

    x = range(len(filtered_difficulties))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar([i - width/2 for i in x], y1, width=width, label='without RAG', color='skyblue')
    plt.bar([i + width/2 for i in x], y2, width=width, label='RAG', color='orange')
    
    plt.axvline(x=6.5, color='red', linestyle='--', linewidth=1)
    
    plt.xticks(ticks=x, labels=filtered_difficulties)
    plt.xlabel('Difficulty')
    plt.ylabel('Pass Rate')
    plt.title('Pass Rate by Difficulty (4o-RAG_use_true_code)')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(file_name)
    
def save_difficulty_pass_3(file_path1, file_path2, file_path3):
    diff_pass1 = show_difficulty_pass(file_path1)
    diff_pass2 = show_difficulty_pass(file_path2)
    diff_pass3 = show_difficulty_pass(file_path3)

    # 所有 difficulty 的并集并排序
    all_difficulties = sorted(set(diff_pass1.keys()) | set(diff_pass2.keys()) | set(diff_pass3.keys()))

    # 过滤：去掉三个值都是 0.0 的 difficulty
    filtered_difficulties = []
    for d in all_difficulties:
        v1 = diff_pass1.get(d, 0.0)
        v2 = diff_pass2.get(d, 0.0)
        v3 = diff_pass3.get(d, 0.0)
        if not (v1 == 0.0 and v2 == 0.0 and v3 == 0.0):
            filtered_difficulties.append(d)
    # print(filtered_difficulties)
    
    # 对齐数据
    y1 = [diff_pass1.get(d, 0.0) for d in filtered_difficulties]
    y2 = [diff_pass2.get(d, 0.0) for d in filtered_difficulties]
    y3 = [diff_pass3.get(d, 0.0) for d in filtered_difficulties]

    x = range(len(filtered_difficulties))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar([i - width for i in x], y1, width=width, label='logic', color='skyblue')
    plt.bar(x, y2, width=width, label='dpr', color='orange')
    plt.bar([i + width for i in x], y3, width=width, label='bm25', color='lightgreen')
    
    plt.axvline(x=6.5, color='red', linestyle='--', linewidth=1)

    plt.xticks(ticks=x, labels=filtered_difficulties)
    plt.xlabel('Difficulty')
    plt.ylabel('Pass Rate')
    plt.title('Pass Rate by Difficulty (logic, dpr, bm25)')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("test_4o_three.png")
    
def count_new_pass1(file_path_without_rag, file_path, diff_theta=1400):
    data_without_rag = read_json_or_jsonl(file_path_without_rag)
    data = read_json_or_jsonl(file_path)
    all_item = 0 
    correct = 0
    for item in data_without_rag:
        if item['source_data']['difficulty'] <= diff_theta:
            all_item += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct += 1
    for item in data:
        if item['source_data']['difficulty'] > diff_theta:
            all_item += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct += 1
    return all_item, correct, float(correct / all_item)

def count_new_pass3(file_path, diff_theta=[1400, 2000]):
    assert len(diff_theta) == 2
    data = read_json_or_jsonl(file_path)
    all_item = [0] * 3
    correct = [0] * 3
    for item in data:
        if item['source_data']['difficulty'] <= diff_theta[0]:
            all_item[0] += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct[0] += 1
        elif item['source_data']['difficulty'] <= diff_theta[1]:
            all_item[1] += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct[1] += 1
        else:
            all_item[2] += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct[2] += 1       
    return all_item, correct, [float(c / a) for c, a in zip(correct, all_item)]

def merge_diff_solution(file_path_logic, file_path_rag, file_save):
    data_save = []
    data_logic = read_json_or_jsonl(file_path_logic)
    data_rag = read_json_or_jsonl(file_path_rag)
    for item in data_logic:
        src_uid = item['src_uid']
        item_rag = next((sample for sample in data_rag if sample.get('src_uid') == src_uid), None)
        if item_rag:
            retrieve1 = item_rag['retrieve'][0]
            retrieve2 = item['retrieve'][0]
            data_save.append(item)
            data_save[-1]['retrieve'] = [retrieve1, retrieve2]
    write_json_or_jsonl(file_save, data_save) 
    
def detection_longth(data_path):
    data = read_json_or_jsonl(data_path)
    longth = collections.defaultdict(int)
    for item in data:
        longth[len(item['retrieve'])] += 1
    return longth


def plot_comparison(pass_rat, pass_rat_logic, pass_rat_dpr, pass_rat_bm25):
    categories = ["<=1400", "<=2000", ">2000"]
    methods = ["logic", "dpr", "bm25", "without RAG"]
    raw_values = [pass_rat_logic, pass_rat_dpr, pass_rat_bm25, pass_rat]
    
    values = [[v * 100 for v in method] for method in raw_values]

    x = np.arange(len(categories))
    width = 0.2
    # colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(len(methods)):
        ax.bar(x + i * width, values[i], width, label=methods[i], color=colors[i])

    ax.set_ylabel('pass@1(%)', fontsize=12)
    ax.set_title('Performance of different methods at different levels of difficulty', fontsize=14)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend()

    for i in range(len(methods)):
        for j in range(len(categories)):
            ax.text(x[j] + i * width, values[i][j] + 1, f'{values[i][j]:.1f}%', 
                    ha='center', va='bottom', fontsize=9)
    
    line_colors = ['#FF5733', '#33C3FF', '#28B463']
    for i, y in enumerate(values[0]):
        left = x[i] - 0.2
        right = x[i] + width * 3.8  # 四根柱子 + 一点余量
        ax.hlines(y, xmin=left, xmax=right, colors=line_colors[i], linestyles='--', linewidth=1.5)


    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("test_diff_3_gpt35.png")

def coverage(file_path, file_path_true):
    data_rag = read_json_or_jsonl(file_path)
    data_list = read_json_or_jsonl(file_path_true)
    all_num = 0
    cover = 0
    diff_count = []
    for item in data_rag:
        all_num += 1
        src_uid = item['src_uid']
        source_data = next((sample for sample in data_list if sample.get("src_uid") == src_uid), None)
        retrieved_uid_list = [code['src_uid'] for code in source_data['retrieve']]
        if item['retrieve'][0]['src_uid'] in retrieved_uid_list:
            cover += 1
            diff_count.append(item['difficulty'])
    return all_num, cover, float(cover / all_num), diff_count

# def coverage(file_path, file_path_true):
#     data_rag = read_json_or_jsonl(file_path)
#     data_list_ref = read_json_or_jsonl(file_path_true)
#     data_list = []
#     for item in data_list_ref:
#         data_list.append(item['source_data'])
#     all_num = 0
#     cover = 0
#     diff_count = []
#     for item in data_rag:
#         all_num += 1
#         src_uid = item['source_data']['src_uid']
#         source_data = next((sample for sample in data_list if sample.get("src_uid") == src_uid), None)
#         retrieved_uid_list = [code['src_uid'] for code in source_data['retrieve']]
#         if item['retrieve'][0]['src_uid'] in retrieved_uid_list:
#             cover += 1
#             diff_count.append(item['source_data']['difficulty'])
#     return all_num, cover, float(cover / all_num), diff_count

def select_flaw_retrieve(file_path, file_path_list, file_save):
    data = read_json_or_jsonl(file_path)
    data_list = read_json_or_jsonl(file_path_list)
    data_save = []
    for item in data:
        # stop = input(item['retrieve'][0])
        src_uid = item['src_uid']
        target = next((surce_data for surce_data in data_list if surce_data.get("src_uid") == src_uid), None)
        src_re = target['retrieve'][0]['src_uid']
        # stop = input(src_re)
        if item['retrieve'][0]['src_uid'] != src_re:
            data_save.append(item)
    print(len(data_save))
    stop = input()
    write_json_or_jsonl(file_save, data_save)

def extract_difficult(file_path, save_path):
    data = read_json_or_jsonl(file_path)
    save_data = []
    for item in data:
        print(len(item['retrieve']))
        diffic = item['difficulty']
        if diffic <= 1400:
            save_data.append(item)
    write_json_or_jsonl(save_path, save_data)
    
def compute_part_solution(file_path, file_list_path):
    data = read_json_or_jsonl(file_path)
    data_list = read_json_or_jsonl(file_list_path)
    src_list = []
    all_sample = 0
    correct = 0
    for item in data_list:
        src_list.append(item['src_uid'])
    print(len(src_list))
    for item in data:
        if item['source_data']['src_uid'] in src_list:
            all_sample += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct += 1
    print(correct / all_sample)
    
def count_diff_num(file_path):
    data = read_json_or_jsonl(file_path)
    diff_count = [0] * 3
    src_list = []
    for item in data:
        diff = item['difficulty']
        if diff <= 1400:
            diff_count[0] += 1
        elif diff <=2000:
            diff_count[1] += 1
        else:
            diff_count[2] += 1
            src_list.append(item['src_uid'])
    return diff_count, src_list

def compute_pass_list(file_path, src_list):
    data = read_json_or_jsonl(file_path)
    all_count = 0
    correct_count = 0
    for item in data:
        if item['source_data']['src_uid'] in src_list:
            all_count += 1
            result = all(x["exec_outcome"] == "PASSED" for x in item['unit_test_results'][0])
            if result == True:
                correct_count += 1            
    print(correct_count / all_count)

def show_json_item_collection(file_path, item_name=None):
    data = read_json_or_jsonl(file_path)
    if item_name is not None:
        collection_item = collections.defaultdict(int)
        for item in tqdm(data):
            # if item[item_name] not in collection_item:
            collection_item[item[item_name]] += 1
    return collection_item    

def random_sample(file_path, file_path_list, file_save_path):
    data_retrieve = read_json_or_jsonl(file_path)
    assert len(file_path_list) >= 1
    data_list = []
    for file_p in file_path_list:
        data = read_json_or_jsonl(file_p)
        data_list.append(data)
    data = data_list[0]
    save_data = []
    for i, item in enumerate(tqdm(data)):
        retrieved_list = []
        seen_ids = set()
        for data_j in data_list:
            for item_s in data_j[i]['retrieve']:
                if item_s['src_uid'] not in seen_ids:
                    seen_ids.add(item_s['src_uid'])
                    retrieved_list.append(item_s)
        filtered = [item for item in data_retrieve if item not in retrieved_list]
        # 去掉 A 中所有出现在 B 中的元素
        samples = random.sample(filtered, 10)
        save_sample = {key: item[key] for key in item if key != 'retrieve'}
        save_sample['retrieve'] = samples
        save_data.append(save_sample)
    write_json_or_jsonl(file_save_path, save_data)
    
def apps_train_retrieve_process(file_path, file_save_path):
    data = read_json_or_jsonl(file_path)
    data_save = []
    for item in data:
        if len(item['solutions']) == 0:
            continue
        item_save = {
            "src_uid": 'train_' + str(item['id']),
            "original_question": str(item['question']),
            "solution_code": item['solutions']
        }
        data_save.append(item_save)
    write_json_or_jsonl(file_save_path, data_save)

def apps_test_retrieve_process(file_path, file_save_path):
    data = read_json_or_jsonl(file_path)
    data_save = []
    for item in data:
        if item['difficulty'] != 'competition':
            continue
        item_save = {key: item[key] for key in item if key not in ['question', 'id', 'solutions', 'starter_code']}
        item_save['description'] = item['question']
        item_save['src_uid'] = 'test_i' + str(item['id'])
        data_save.append(item_save)
    write_json_or_jsonl(file_save_path, data_save)

# file_input = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_logic_for_review_top_k"
# split_jsonl_file(file_input, file_save_path, num_parts=20)
# stop = input("finish")
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/apps_reviewed2.jsonl"
# data = read_json_or_jsonl(file_path)
# item = data[0]
# for name in item:
#     print(name)
# input_output = json.loads(item['input_output'])
# print(len(input_output['inputs']), len(input_output['outputs']))
# stop = input()
# print(input_output['inputs'])
# stop = input(
# )
# print(input_output['outputs'])
# stop = input()

# file_path = "/home/zhangshiwen/APPS/test.jsonl"
# file_save_path = "/home/zhangshiwen/APPS/test_competition_nl.jsonl"
# apps_test_retrieve_process(file_path, file_save_path)
# stop = input("finish")
# file_path = "/home/zhangshiwen/APPS/train_clean.jsonl"
# file_save_path = "/home/zhangshiwen/APPS/train_clean_for_retrieve.json"
# apps_train_retrieve_process(file_path, file_save_path)
# stop = input("finish")

# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/original_questions_train_with1code.json"
# file_list = [
#     "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_bm25_with_notes_393_reviewed2.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_codebert_with_notes_393_reviewed2.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_dpr_with_notes_393_reviewed2.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/dataset/random_retrieve_samples.jsonl"
# ]
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/random_retrieve_samples_2.jsonl"
# random_sample(file_path, file_list, file_save_path)
# stop = input("finish")


# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl.jsonl"
# data = read_json_or_jsonl(file_path)
# item = data[1]
# # for item in data:
# #     print(item['solutions'])
# #     print(len(item['solutions']))
# #     stop = input()
#     # assert len(item['solutions']) > 0
# for name in item:
#     print(name)
#     stop = input()
#     print(item[name])
# stop = input()
# file_path = "/home/zhangshiwen/APPS/test.jsonl"
# collection_item = show_json_item_collection(file_path, item_name='difficulty')
# print(collection_item)
# stop = input()



# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# data = read_json_or_jsonl(file_path)
# item = data[0]
# for name in item['retrieve']:
#     print(name['solution_code'])
#     stop = input()

# _, src_list = count_diff_num("/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2_new.jsonl")

# file_path = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_re_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# compute_pass_list(file_path, src_list)

# file_list_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2_new.jsonl"
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_re_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# compute_part_solution(file_path, file_list_path)

# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed1.jsonl"
# file_path_list = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# file_save = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_dpr_with_notes_393_reviewed2_diff_2000_faw.jsonl"
# select_flaw_retrieve(file_path, file_path_list, file_save)

# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_codebert_with_notes_393_reviewed2.jsonl"
# save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_codebert_with_notes_393_reviewed2_diff_0.jsonl"
# extract_difficult(file_path, save_path)



# file_path_dpr = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_dpr_with_notes_393_reviewed2.jsonl"
# file_path_bm25 = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_bm25_with_notes_393_reviewed2.jsonl"
# file_path_true = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# all_num_dpr, cover_dpr, result_dpr, diff_count_dpr = coverage(file_path_dpr, file_path_true)
# all_num_bm25, cover_bm25, result_bm25, diff_count_bm25 = coverage(file_path_bm25, file_path_true)

# print("\t覆盖数量\t覆盖率")
# print("bm25:", '\t', cover_bm25, '\t', result_bm25)
# print("dpr:", '\t', cover_dpr, '\t', result_dpr)
# print(diff_count_dpr)
# print(diff_count_bm25)

# print(all_num, cover, result)
    
# file_path_logic = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# file_path_rag = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_bm25_with_notes_393_reviewed2.jsonl"
# file_save = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_merge_with_notes_393_reviewed2.jsonl"
# merge_diff_solution(file_path_logic, file_path_rag, file_save)    

# longth_dict = detection_longth(file_path_logic)
# print(longth_dict)


# diff_pass = show_difficulty_pass("/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_4o_code_review2/reproduce_python/PyPy3_results.jsonl")
# for item in sorted(diff_pass.keys()):
#     print(f"{item}, {diff_pass[item]}")

# file_path2 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_re_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_dpr_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_bm25_4o_code_review2/reproduce_python/PyPy3_results.jsonl"

# save_difficulty_pass_3(file_path2, file_path3, file_path4)
# gpt4o
# file_path1 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path2 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_re_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_dpr_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_bm25_4o_code_review2/reproduce_python/PyPy3_results.jsonl"
# file_path5 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_unixcoder_4o_review2_t/reproduce_python/PyPy3_results.jsonl"
# file_path6 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_codebert_4o/reproduce_python/PyPy3_results.jsonl"
# file_path7 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_random_samples_4o_2/reproduce_python/PyPy3_results.jsonl"
# file_path8 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_tf_4o/reproduce_python/PyPy3_results.jsonl"
# file_path9 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_idf_4o/reproduce_python/PyPy3_results.jsonl"
file_path10 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_rere_reacc_4o/reproduce_python/PyPy3_results.jsonl"


#gpt4o top1
# file_path1 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/same_srcuid_filtered_gpt35/PyPy3_results_no_rag.jsonl"
# file_path2 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/same_srcuid_filtered_gpt35/PyPy3_results_logic.jsonl"
# file_path3 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/same_srcuid_filtered_gpt35/PyPy3_results_dpr.jsonl"
# file_path4 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/same_srcuid_filtered_gpt35/PyPy3_results_bm25.jsonl"
# file_path6 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/same_srcuid_filtered_gpt35/PyPy3_results_codebert.jsonl"


# gpt3.5
# file_path1 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_gpt35t/reproduce_python/PyPy3_results.jsonl"
# file_path2 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_re_top2_gpt35_logic/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_re_top2_gpt35_dpr/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_re_top2_gpt35_bm25/reproduce_python/PyPy3_results.jsonl"
# file_path5 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_unixcoder_gpt35_t/reproduce_python/PyPy3_results.jsonl"
# file_path6 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_re_top2_gpt35_codebert/reproduce_python/PyPy3_results.jsonl"
# file_path7 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_random_samples_35/reproduce_python/PyPy3_results.jsonl"
# file_path8 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_tf_gpt35/reproduce_python/PyPy3_results.jsonl"
# file_path9 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_idf_gpt35/reproduce_python/PyPy3_results.jsonl"
# file_path10 = "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_re_reacc_gpt35/reproduce_python/PyPy3_results.jsonl"


# # codellama
# file_path1 = "/home/zhangshiwen/model_test/CodeLlama_test_no_rag/reproduce_python/PyPy3_results.jsonl"
# file_path2 = "/home/zhangshiwen/model_test/CodeLlama_test_logic/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/model_test/CodeLlama_test_dpr/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/model_test/CodeLlama_test_bm25/reproduce_python/PyPy3_results.jsonl"
# file_path6 = "/home/zhangshiwen/model_test/CodeLlama_test_codebert/reproduce_python/PyPy3_results.jsonl"

# deepseek-coder
# file_path1 = "/home/zhangshiwen/model_test/deepseek_coder_test_no_rag/reproduce_python/PyPy3_results.jsonl"
# file_path2 = "/home/zhangshiwen/model_test/deepseek_coder_test_logic/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/model_test/deepseek_coder_test_dpr/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/model_test/deepseek_coder_test_bm25/reproduce_python/PyPy3_results.jsonl"
# file_path6 = "/home/zhangshiwen/model_test/deepseek_coder_test_codebert/reproduce_python/PyPy3_results.jsonl"

# starcoder
# file_path1 = "/home/zhangshiwen/model_test/starcoder_test_no_rag/reproduce_python/PyPy3_results.jsonl"
# file_path2 = "/home/zhangshiwen/model_test/starcoder_test_logic/reproduce_python/PyPy3_results.jsonl"
# file_path3 = "/home/zhangshiwen/model_test/starcoder_test_dpr/reproduce_python/PyPy3_results.jsonl"
# file_path4 = "/home/zhangshiwen/model_test/starcoder_test_bm25/reproduce_python/PyPy3_results.jsonl"
# file_path6 = "/home/zhangshiwen/model_test/starcoder_test_codebert/reproduce_python/PyPy3_results.jsonl"


# all_item_rat, correct_rat , pass_rat = count_new_pass3(file_path1)
# all_item_logic, correct_logic , pass_rat_logic = count_new_pass3(file_path2)
# all_item_dpr, correct_dpr , pass_rat_dpr = count_new_pass3(file_path3)
# all_item_bm25, correct_bm25 , pass_rat_bm25 = count_new_pass3(file_path4)
# # all_item_unicoder, correct_unixcoder , pass_rat_unixcoder = count_new_pass3(file_path5)
# all_item_codebert, correct_codebert  , pass_rat_codebert = count_new_pass3(file_path6)
# all_item_random, correct_random  , pass_rat_random = count_new_pass3(file_path7)
# all_item_tf, correct_tf  , pass_rat_tf = count_new_pass3(file_path8)
# all_item_idf, correct_idf  , pass_rat_idf = count_new_pass3(file_path9)
all_item_reacc, correct_reacc  , pass_rat_reacc = count_new_pass3(file_path10)



# print('gpt-35-t')
# print("无检索：" + "".join(f"\t{sum(correct_rat[1:]) / sum(all_item_rat[1:]) * 100:.2f}"))
# print("逻辑检索：" + "".join(f"\t{sum(correct_logic[1:]) / sum(all_item_logic[1:]) * 100:.2f}"))
# print("普通dpr：" + "".join(f"\t{sum(correct_dpr[1:]) / sum(all_item_dpr[1:]) * 100:.2f}"))
# print("bm25：\t" + "".join(f"\t{sum(correct_bm25[1:]) / sum(all_item_bm25[1:]) * 100:.2f}"))
# # print("unixcoder：" + "".join(f"\t{sum(correct_unixcoder[1:]) / sum(all_item_unixcoder[1:]) * 100:.2f}"))
# print("codebert：" + "".join(f"\t{sum(correct_codebert[1:]) / sum(all_item_codebert[1:]) * 100:.2f}"))
# print("random：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_random]))



# print("gpt-4o")
# print(f"\t\t <=1400\t\t<=2000\t\t>2000")
# print("无检索：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat]))
# print("逻辑检索：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_logic]))
# print("普通dpr：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_dpr]))
# print("bm25：\t" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_bm25]))
# print("unixcoder：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_unixcoder]))
# print("codebert：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_codebert]))
# print("random：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_random]))
# print("tf：   " + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_tf]))
# print("idf：  " + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_idf]))
print("reacc：  " + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_reacc]))




# print("codellama")
# print(f"\t\t <=1400\t\t<=2000\t\t>2000")
# print("无检索：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat]))
# print("逻辑检索：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_logic]))
# print("普通dpr：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_dpr]))
# print("bm25：\t" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_bm25]))
# print("codebert：" + "".join([f"\t{rate * 100:.2f}%\t" for rate in pass_rat_codebert]))



# print(all_item)

# plot_comparison(pass_rat, pass_rat_logic, pass_rat_dpr, pass_rat_bm25)
# print(f"bm25：\t{pass_rat_bm25 * 100:.2f}")
# print(f"无检索：\t{62.57}")

# save_difficulty_pass(file_path1, file_path4, "test_4o_bm25.png")
# _, _ , pass_rat_re = count_new_pass1(file_path1, file_path2)
# _, _ , pass_rat_dpr = count_new_pass1(file_path1, file_path3)
# _, _ , pass_rat_bm25 = count_new_pass1(file_path1, file_path4)
# print(f"\t\t pass@1")
# print(f"逻辑检索 + bm25：\t{pass_rat_re * 100:.2f}")
# print(f"普通dpr：\t{pass_rat_dpr * 100:.2f}")
# print(f"bm25-1 + bm25-2：\t{pass_rat_bm25 * 100:.2f}")
# print(f"无检索：\t{62.57}")

# diff_pass = show_difficulty_pass(file_path1)
# print(diff_pass)
# for item in sorted(diff_pass.keys()):
#     print(f"{item}, {diff_pass[item]}")

# file_path_list = [
#     "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_deepseek/reproduce_python/PyPy3_results.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_deepseek_code_review/reproduce_python/PyPy3_results.jsonl",
#     "/home/zhangshiwen/xCodeEval/evaluation/program_synthesis/my_outputs_backup_deepseek_code_review_2/reproduce_python/PyPy3_results.jsonl"
# ]
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2.jsonl"
# save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2_true_solution.jsonl"
# change_solution_code_for_test(file_path_list, file_path, save_path)

# save_all_reviewed_code使用样例
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_dpr_with_notes_393_reviewed2_diff_1400.jsonl"
# file_list_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/nl_code_train_review_merge1.jsonl"
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_dpr_with_notes_393_reviewed2_diff_1400_review1.jsonl"
# save_all_reviewed_code(file_path, file_list_path, file_save_path)

# # merge_split使用样例
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_logic_for_review_top_k_reviewed"
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/topk_logic_1.jsonl"
# merge_split(file_path, file_save_path)

# split_jsonl_file使用样例
# file_input = "/home/zhangshiwen/xCodeEval/evaluation/dataset/apss_with_code_logic.jsonl"
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/apss_with_code_logic_split"
# split_jsonl_file(file_input, file_save_path, num_parts=20)

# save_one_reviewed_code使用样例
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_graphcodebert_new.jsonl"
# file_list_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/nl_code_train_review_merge1.jsonl"
# file_save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_graphcodebert_reviewed1.jsonl"
# save_one_reviewed_code(file_path, file_list_path, file_save_path)




# extract_sample_question_same使用样例
# file_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_bm25_with_notes_393_reviewed1.jsonl"
# file_list_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed2_new.jsonl"
# save_path = "/home/zhangshiwen/xCodeEval/evaluation/dataset/prog_syn_test_nl_retrive_with_code_bm25_with_notes_393_reviewed2_new.jsonl"
# extract_sample_question_same(file_path, file_list_path, save_path)

