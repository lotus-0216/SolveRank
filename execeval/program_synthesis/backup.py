import os
import json
import tqdm

# # 设置你的输出目录
input_dir = "my_outputs_rere_reacc_4o"
backup_dir = "my_outputs_backup_rere_reacc_4o"

# # 设置你的输出目录
# input_dir = "/home/zhangshiwen/model_test/CodeLlama_test_codebert/"
# backup_dir = "/home/zhangshiwen/model_test/CodeLlama_backup_test_codebert/"

# 如果需要，可以备份原始文件
os.makedirs(backup_dir, exist_ok=True)

def build_hidden_unit_tests(sample_inputs, sample_outputs):
    hidden_unit_tests = []
    for inp, out in zip(sample_inputs, sample_outputs):
        hidden_unit_tests.append({
            "input": inp,
            "output": [out] if isinstance(out, str) else out  # 保证 output 是 list
        })
    return hidden_unit_tests

def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "source_data" not in data:
        print(f"Warning: no source_data in {file_path}")
        return

    # 只有当没有 hidden_unit_tests 时才补充
    if "hidden_unit_tests" not in data["source_data"]:
        sample_inputs = data["source_data"].get("sample_inputs", [])
        sample_outputs = data["source_data"].get("sample_outputs", [])
        
        if not sample_inputs or not sample_outputs:
            print(f"Warning: sample_inputs or sample_outputs missing in {file_path}")
            return

        data["source_data"]["hidden_unit_tests"] = json.dumps(
            build_hidden_unit_tests(sample_inputs, sample_outputs),
            ensure_ascii=False
        )

        # 保存之前备份一下原始文件
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        with open(backup_path, "w", encoding="utf-8") as bf:
            json.dump(data, bf, ensure_ascii=False, indent=4)

        # 保存更新后的文件
        with open(file_path, "w", encoding="utf-8") as wf:
            json.dump(data, wf, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    for file_name in tqdm.tqdm(files, desc="Processing files"):
        process_file(os.path.join(input_dir, file_name))

    print("✅ 所有文件处理完成！hidden_unit_tests 补充成功。")
