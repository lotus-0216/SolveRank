# import openai
# import json
# import time
# from tqdm import tqdm
# import os

# openai.api_key = "sk-DFpJaQWzEeSBTcsC6kfi2pOEvJj9RYVBmrZ5BEt3ZiTBslBD"
# openai.api_base = "https://api.chatanywhere.tech/v1"

# # INPUT_FILE = "retrieved_top20_dpr_traindata.json"
# INPUT_FILE = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed1_split_new/data_part_5.jsonl"
# OUTPUT_FILE = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed1_split_new/data_part_5_reviewed.jsonl"
# MODEL = "deepseek-v3"

# add_prompt = """and if the solution to Problem B uses a specific algorithm, please determine whether this method is fundamentally applicable to Problem A as well; in other words, can the algorithm used for Problem B be reasonably applied to solve Problem A? Only reply YES if both conditions are met.
# """

# def gpt4o_logic_relevance(query, retrieved_question, solution_code, max_retries=3):
#     prompt = f"""Please determine whether the following two questions belong to the same category in terms of modeling logic and algorithmic abstraction. Focus only on their algorithmic modeling structure, core optimization objectives, and typical solution approaches. Ignore the specific real-world background or story.

# If you believe that the two problems share the same fundamental algorithmic abstraction {add_prompt}—for example, both can be reduced to or solved as one of the following problem types:

# - Shortest path, minimum/maximum cost path
# - Traveling salesman problem (TSP)
# - Network flow (max flow, min cut, matching, circulation)
# - Dynamic programming (DP) for sequences, subsets, intervals, or grids
# - Knapsack/packing, subset sum, resource allocation
# - Greedy algorithms (interval scheduling, activity selection, Huffman coding, matroids, etc.)
# - Sorting and searching (binary search, selection, insertion, merge, quicksort, etc.)
# - Priority queue/heap/stack/queue manipulation
# - Divide and conquer (merge sort, quick sort, etc.)
# - Permutations, combinations, arrangements, ranking
# - Counting or enumerative combinatorics
# - Substring/subarray/subsequence problems (longest increasing, palindromic, etc.)
# - Tree and graph traversal (DFS, BFS, topological sort, Euler/Hamiltonian path)
# - Minimum/maximum spanning tree (Kruskal, Prim)
# - Union-find/disjoint set operations
# - Interval cover/interval scheduling/interval union & intersection
# - Range query data structures (segment tree, Fenwick tree, sparse table, RMQ)
# - Bitmask enumeration, bitwise operations
# - Game theory (Nim, Grundy numbers, minimax)
# - Geometry (convex hull, line sweep, closest pair, etc.)
# - Probabilistic/expected value/Monte Carlo simulations
# - Linear programming, optimization, or simplex methods
# - Simulation and state space search (BFS/DFS over states)
# - Hashing, rolling hash, map/set lookups
# - Modular arithmetic/number theory (gcd, lcm, primes, Chinese remainder, etc.)
# - Matrix and linear algebra (Gaussian elimination, matrix exponentiation)
# - Regular expression/automata/state machines
# - Recursion and recursive search
# - Any other standard algorithmic paradigm widely used in programming contests and computer science

# If the problems are based on the same core abstraction (even with different story settings), answer “Yes”. Otherwise, answer “No”.

# Question A:
# {query}

# Question B:
# {retrieved_question}

# Solution to Problem B:
# {solution_code}

# Please answer only “Yes” or “No”."""
#     # print(prompt)
#     for attempt in range(max_retries):
#         try:
#             response = openai.ChatCompletion.create(
#                 model=MODEL,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=10,
#                 temperature=0.0
#             )
#             answer = response.choices[0].message['content'].strip()
#             if "Yes" in answer or answer.lower().startswith("yes"):
#                 return True
#             if "No" in answer or answer.lower().startswith("no"):
#                 return False
#         except Exception as e:
#             print(f"调用失败，第{attempt+1}次，错误：{e}")
#             time.sleep(3)
#     return False

# def get_existing_src_uids(output_file):
#     """读取已保存结果文件中的 src_uid，避免重复处理"""
#     if not os.path.exists(output_file):
#         return set()
#     processed = set()
#     with open(output_file, "r", encoding="utf-8") as fin:
#         for line in fin:
#             try:
#                 data = json.loads(line)
#                 if "src_uid" in data:
#                     processed.add(data["src_uid"])
#             except Exception:
#                 continue
#     return processed

# def main():
#     data = []
#     with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
#         for line in fin:
#             data.append(json.loads(line))

#     processed_src_uids = get_existing_src_uids(OUTPUT_FILE)
#     print(f"已处理 {len(processed_src_uids)} 个 src_uid，将跳过...")

#     with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
#         for entry in tqdm(data[:], desc="筛选逻辑相关问题"):
#             src_uid = entry.get("src_uid")
#             query = entry["description"]
#             top20 = entry["retrieve"]

#             # 跳过已处理过的src_uid
#             if src_uid in processed_src_uids:
#                 continue

#             # 构建 question -> src_uid 映射
#             # question_to_src_uid = {item["question"]: item.get("src_uid") for item in top20}

#             # logic_similar_questions = []
#             # found = False
#             # for item in top20:
#             #     for code in item["solution_code"]:
#             #         # stop = input(code)
#             #         if gpt4o_logic_relevance(query, item["original_question"], code):
#             #             item["solution_code"] = code
#             #             logic_similar_questions.append(item)
#             #             found = True
#             #             break
#             #     if found:
#             #         break
            
#             logic_similar_questions = []
#             for item in top20:
#                 # if item["label"] == 0:
#                 for code in item["solution_code"]:
#                     is_logic_relevant = gpt4o_logic_relevance(query, item["original_question"], code)
#                     if is_logic_relevant:
#                         break
#                 item["solution_code"] = code
#                 logic_similar_questions.append(item)
                
#             if logic_similar_questions:
#                 result_save = entry
#                 result_save['retrieve'] = logic_similar_questions
#                 # result = {
#                 #     "src_uid": src_uid,
#                 #     "query": query,
#                 #     "Logically_similar_questions": logic_similar_questions
#                 # }
#                 fout.write(json.dumps(result_save, ensure_ascii=False) + "\n")
#                 fout.flush()  # 保证每条都落盘

# if __name__ == "__main__":
#     main()
import openai 
import json
import time
import os
import tiktoken
from tqdm import tqdm

# 设置 ChatAnywhere API 兼容 DeepSeek
openai.api_key = "sk-DFpJaQWzEeSBTcsC6kfi2pOEvJj9RYVBmrZ5BEt3ZiTBslBD"
openai.api_base = "https://api.chatanywhere.tech/v1"
MODEL = "gpt-4o"

# 数据路径配置
INPUT_FILE = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed1_split_new/data_part_5.jsonl"
OUTPUT_FILE = "/home/zhangshiwen/xCodeEval/evaluation/dataset/retrieved_top20_dpr_traindata_logic_high_with_notes_393_reviewed1_split_new/data_part_5_reviewed.jsonl"

# 成本估算配置
USD_TO_CNY = 7.2
GPT4O_PRICES = {
    "prompt": 0.0175 / 1000,
    "completion": 0.07 / 1000
}
ENCODER = tiktoken.encoding_for_model("gpt-4")

# 累计统计
total_prompt_tokens = 0
total_completion_tokens = 0
total_cost_usd = 0.0
total_cost_cny = 0.0

add_prompt = """and if the solution to Problem B uses a specific algorithm, please determine whether this method is fundamentally applicable to Problem A as well; in other words, can the algorithm used for Problem B be reasonably applied to solve Problem A? Only reply YES if both conditions are met."""


def count_tokens(text):
    return len(ENCODER.encode(text))


def estimate_cost(prompt_tokens, completion_tokens):
    cost_usd = (
        prompt_tokens * GPT4O_PRICES["prompt"] +
        completion_tokens * GPT4O_PRICES["completion"]
    )
    cost_cny = cost_usd * USD_TO_CNY
    return round(cost_usd, 6), round(cost_cny, 4)


def gpt4o_logic_relevance(query, retrieved_question, solution_code, max_retries=3):
    global total_prompt_tokens, total_completion_tokens, total_cost_usd, total_cost_cny

    prompt = f"""Please determine whether the following two questions belong to the same category in terms of modeling logic and algorithmic abstraction. Focus only on their algorithmic modeling structure, core optimization objectives, and typical solution approaches. Ignore the specific real-world background or story.

If you believe that the two problems share the same fundamental algorithmic abstraction {add_prompt} — for example, both can be reduced to or solved as one of the following problem types:

If the problems are based on the same core abstraction (even with different story settings), answer “Yes”. Otherwise, answer “No”.

Question A:
{query}

Question B:
{retrieved_question}

Solution to Problem B:
{solution_code}

Please answer only “Yes” or “No”."""

    prompt_tokens = count_tokens(prompt)

    for attempt in range(max_retries):
        try:
            # 记录开始时间
            start_time = time.time()
            
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            # 记录结束时间
            end_time = time.time()
            generation_time = end_time - start_time  # 计算生成时间

            answer = response.choices[0].message['content'].strip()
            completion_tokens = count_tokens(answer)
            cost_usd, cost_cny = estimate_cost(prompt_tokens, completion_tokens)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_cost_usd += cost_usd
            total_cost_cny += cost_cny

            # ✅ 打印当前调用信息
            print(f"[TOKEN] Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {prompt_tokens + completion_tokens}")
            print(f"[COST]  USD: ${cost_usd} | RMB: ¥{cost_cny} | Result: {answer}")
            print(f"[TIME] Generation time: {generation_time:.2f} seconds")

            if "Yes" in answer or answer.lower().startswith("yes"):
                return True
            if "No" in answer or answer.lower().startswith("no"):
                return False
        except Exception as e:
            print(f"调用失败，第 {attempt+1} 次，错误：{e}")
            time.sleep(3)

    return False


def get_existing_src_uids(output_file):
    if not os.path.exists(output_file):
        return set()
    processed = set()
    with open(output_file, "r", encoding="utf-8") as fin:
        for line in fin:
            try:
                data = json.loads(line)
                if "src_uid" in data:
                    processed.add(data["src_uid"])
            except Exception:
                continue
    return processed


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        data = [json.loads(line) for line in fin]

    processed_src_uids = get_existing_src_uids(OUTPUT_FILE)
    print(f"已处理 {len(processed_src_uids)} 个 src_uid，将跳过...")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for entry in tqdm(data[:], desc="筛选逻辑相关问题"):
            src_uid = entry.get("src_uid")
            if src_uid in processed_src_uids:
                continue

            query = entry["description"]
            top20 = entry["retrieve"]
            logic_similar_questions = []

            for item in top20:
                for code in item.get("solution_code", []):
                    is_logic_relevant = gpt4o_logic_relevance(query, item["original_question"], code)
                    if is_logic_relevant:
                        break
                item["solution_code"] = code
                logic_similar_questions.append(item)

            if logic_similar_questions:
                entry["retrieve"] = logic_similar_questions
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fout.flush()


if __name__ == "__main__":
    main()

    # === 总结统计输出 ===
    print("\n✅ 总结 Token 和成本:")
    print(f"Prompt tokens 总数: {total_prompt_tokens}")
    print(f"Completion tokens 总数: {total_completion_tokens}")
    print(f"Token 总数: {total_prompt_tokens + total_completion_tokens}")
    print(f"总成本 (USD): ${round(total_cost_usd, 6)}")
    print(f"总成本 (RMB): ¥{round(total_cost_cny, 4)}")
