import os
import time
import tqdm
import json
import openai
import concurrent
from promptsource.templates import Template
from tiktoken import encoding_for_model

# 固定参数写在这里
N = 3
INPUT_JSONL = "../dataset/topk_logic.jsonl"  # 输入文件
OUTPUT_DIR = "my_outputs_re_top3_4o_logic_1"     # 输出目录
NUM_PROC = 1                                                      # 并行核数
NSAMPLE = 1                                                      # 每个prompt生成10个样本
TEMPERATURE = 0.2                                                 # sampling温度
IF_SAMPLE = False


# 设置OpenAI API
openai.api_key = "sk-Okf9TYIr0BvKEetyZ7j5vtQyWBmE7EYuMhf5ayS5HSnijONO"
openai.api_base = "https://api.chatanywhere.tech/v1"

def gen(prompt, temperature, nsample):
    cnt = 0
    while cnt < 999:
        try:
            model = "gpt-4o"
            # enc = encoding_for_model(model)
            # max_prompt_tokens = 16385
            # encoded_prompt = enc.encode(prompt)
            # if len(encoded_prompt) > max_prompt_tokens:
            #     truncated_prompt = enc.decode(encoded_prompt[:max_prompt_tokens])
            # else:
            #     truncated_prompt = enc.decode(encoded_prompt)
            c = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=1,
                n=nsample,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=1024
            )
            c["prompt"] = prompt
            return c
        except Exception as e:
            cnt += 1
            time.sleep(5)
            print(f"API Error: {e}")
    return None

# xcodeeval_prompt_template = {
#     "program_synthesis": [
#         "Write a program in {{lang_cluster}} to solve this programming problem:\nDescription: {{description}}\nInput Specification: {{input_spec}}\nOutput Specification: {{output_spec}}\n{% for input, output in zip(sample_inputs, sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nNotes: {{notes}}\nTake input from {{input_from}} and output to {{output_to}}\nProvide the {{lang_cluster}} code without any extra description or tokens. Target code: ||END-of-SRC|| ",
#     ]
# }
xcodeeval_prompt_template = {
    "program_synthesis": [
        "Write a program in {{lang_cluster}} to solve this programming problem:\nDescription: {{description}}\n\n{% if retrieved_context %}Relevant examples(The following examples are selected based on their similarity to the current problem in terms of algorithmic modeling logic and abstraction. They share comparable modeling structures, core optimization objectives, or typical solution strategies. You may ignore the specific application context or surface narrative — focus instead on the underlying algorithmic structure and reasoning process. Use these examples as guidance to help generate code that aligns with the intended problem-solving logic.):\n{{retrieved_context}}\n{% endif %}\nInput Specification: {{input_spec}}\nOutput Specification: {{output_spec}}\n{% for input, output in zip(sample_inputs, sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nNotes: {{notes}}\nTake input from {{input_from}} and output to {{output_to}}\nProvide the {{lang_cluster}} code without any extra description or tokens. Target code: ||END-of-SRC|| ",
    ]
}

# xcodeeval_prompt_template = {
#     "program_synthesis": [
#         "Write a program in {{lang_cluster}} to solve this programming problem.\n\n"
#         "Description:\n{{description}}\n\n"
        
#         "{% if retrieved_examples %}"
#         "Relevant Example Problems:\n"
#         "The following examples are selected based on their similarity in algorithmic structure, abstraction, or problem-solving strategy. "
#         "Focus on understanding and reusing the **underlying algorithmic ideas**, not the surface syntax or context. "
#         "Compare them and identify **shared patterns** that can help solve the current problem.\n\n"
        
#         "{% for item in retrieved_examples %}"
#         "---\n"
#         "**Example {{loop.index}}**\n"
#         "Problem Description:\n{{item.original_question}}\n\n"
#         "Solution ({{lang_cluster}}):\n{{item.solution_code}}\n"
#         "{% endfor %}"
#         "---\n\n"
#         "{% endif %}"
        
#         "Now solve the current problem based on the shared modeling logic above.\n\n"
#         "Input Specification:\n{{input_spec}}\n\n"
#         "Output Specification:\n{{output_spec}}\n\n"
        
#         "{% for input, output in zip(sample_inputs, sample_outputs) %}"
#         "Sample Input:\n{{input}}\n"
#         "Sample Output:\n{{output}}\n\n"
#         "{% endfor %}"
        
#         "Additional Notes:\n{{notes}}\n\n"
#         "Take input from {{input_from}} and output to {{output_to}}.\n\n"
# #         "Provide the {{lang_cluster}} code **only**, without any explanation, description, or extra formatting.\n\n"
# #         "Target code: ||END-of-SRC||"
#         "**Output format requirement:**\n"
#         "Your response must follow **this exact format**:\n"
#         "[algorithm]：\n<Write the inferred algorithm idea here>\n\n"
#         "[target code]：\n<Provide the {{lang_cluster}} code here without extra explanation>\n"
#         "End your response after the code.\n||END-of-SRC||"
#     ]
# }

# xcodeeval_prompt_template = {
#     "program_synthesis": [
#         "Write a program in {{lang_cluster}} to solve this programming problem:\nDescription: {{description}}\n\n{% if retrieved_context %}Relevant examples:\n{{retrieved_context}}\n{% endif %}\nInput Specification: {{input_spec}}\nOutput Specification: {{output_spec}}\n{% for input, output in zip(sample_inputs, sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nNotes: {{notes}}\nTake input from {{input_from}} and output to {{output_to}}\nProvide the {{lang_cluster}} code without any extra description or tokens. Target code: ||END-of-SRC|| ",
#     ]
# }
# xcodeeval_prompt_template = {
#     "program_synthesis": [
#         "Write a program in {{lang_cluster}} to solve this programming problem:\nDescription: {{description}}\n\n{% if retrieved_context %}Relevant examples:\n{{retrieved_context}}\n{% endif %}\nInput Specification: {{input_spec}}\nOutput Specification: {{output_spec}}\n{% for input, output in zip(sample_inputs, sample_outputs) %}\nSample Input:\n{{input}}\nSample Output:\n{{output}}\n{% endfor %}\nTake input from {{input_from}} and output to {{output_to}}\nProvide the {{lang_cluster}} code without any extra description or tokens. Target code: ||END-of-SRC|| ",
#     ]
# }

def process_prompt(dt, temperature, nsample, language, template, output_dir, index, dry_run=0):
    file_path = os.path.join(output_dir, f"{index}_{temperature}_{language}.json")
    if not os.path.exists(file_path):
        # os.makedirs(file_path)
        dt = dict(dt)  # 避免多进程出错
        dt["lang_cluster"] = language
        if isinstance(dt["sample_inputs"], str):
            dt["sample_inputs"] = json.loads(dt["sample_inputs"])
        if isinstance(dt["sample_outputs"], str):
            dt["sample_outputs"] = json.loads(dt["sample_outputs"])
        
        if isinstance(dt["retrieve"], str):
            dt["retrieve"] = json.loads(dt["retrieve"])

        # 提取前 n 个
        retrieved_context_list = []
        if IF_SAMPLE:
            for item in dt["retrieve"][N-1:N]:
            # for item in dt["retrieve"][:n]:
                retrieved_context_list.append(
                    f"Problem: {item['original_question']}\nSolution:\n{item['solution_code']}"
                )
        else:
            for item in dt["retrieve"][:N]:
            # for item in dt["retrieve"][:n]:
                retrieved_context_list.append(
                    f"Problem: {item['original_question']}\nSolution:\n{item['solution_code']}"
                )
        dt["retrieved_context"] = "\n---\n".join(retrieved_context_list)

        # n = 1
        # retrieved_context_list = []
        # for item in dt["retrieve"][:n]:
        #     retrieved_context_list.append({
        #         "original_question": item["original_question"],
        #         "solution_code": item["solution_code"]
        #     })
        # dt["retrieved_examples"] = retrieved_context_list
        
        lm_io = template.apply(dt)
        assert len(lm_io) == 2, f"{json.dumps(lm_io, indent=4)}"
        
        # print(lm_io)
        # stop = input()
        
        if dry_run:
            open(file_path, "w").write(json.dumps(lm_io[0], indent=4))
        else:
            out = gen(lm_io[0], temperature, nsample)
            export_data = {"oai_response": out, "source_data": dt}
            open(file_path, "w").write(json.dumps(export_data, indent=4))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    templates = [
        Template(f"prog_syn_{idx}", template, "xCodeEval", delimeter="||END-of-SRC||")
        for idx, template in enumerate(xcodeeval_prompt_template["program_synthesis"])
    ]
    template = templates[0]

    # 读取输入文件
    prog_synthesis_dataset = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            prog_synthesis_dataset.append(json.loads(line))

    temperature_list = [TEMPERATURE]

    # 只生成 Python 的代码
    language = "Python"

    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROC) as executor:
        futures = []
        for idx, dt in tqdm.tqdm(
            enumerate(prog_synthesis_dataset),
            total=len(prog_synthesis_dataset),
            desc=f"Preparing prompts for {language}",
        ):
            for temperature in temperature_list:
                future = executor.submit(
                    process_prompt,
                    dt,
                    temperature,
                    NSAMPLE,
                    language,
                    template,
                    OUTPUT_DIR,
                    idx,
                    dry_run=0
                )
                futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Calling OpenAI API for {language}",
        ):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
