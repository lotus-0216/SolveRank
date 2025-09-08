import torch
from transformers import BertTokenizer
from dpr.models.biencoder import BiEncoder
from dpr.models.hf_models import HFBertEncoder
import json
from tqdm import tqdm
import torch.nn.functional as F
import faiss

def encode_texts(model, tokenizer, texts, device, batch_size=64):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
            _, pooled, _ = model(
                input_ids=inputs['input_ids'],
                token_type_ids=inputs['token_type_ids'],
                attention_mask=inputs['attention_mask']
            )
            pooled = F.normalize(pooled, dim=-1)
            all_embeddings.append(pooled.cpu())
    return torch.cat(all_embeddings, dim=0)

def build_faiss_index(embeddings, use_gpu=True):
    dim = embeddings.shape[1]
    index_cpu = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings.numpy())
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    else:
        index = index_cpu
    index.add(embeddings.numpy())
    return index

def retrieve_top20():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_path = "/home/xxx/solverank/bert-base-uncased"
    # pretrained_path = "/home/xxx/solverank/codebert-base"
    model_path = "/home/xxx/solverank/dpr/output/infonce_dpr_rand20_bm5_new0508/infonce_epoch10.pt"
    merged_data_path = "/home/xxx/solverank/dpr/dataset/prog_syn_test_nl.jsonl"
    extracted_data_path = "/home/xxx/solverank/dpr/dataset/original_questions_train_with1code.json"

    tokenizer = HFBertEncoder.init_tokenizer(pretrained_path)
    encoder = HFBertEncoder.init_encoder(pretrained_path).to(device)
    model = BiEncoder(encoder, encoder).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # with open(merged_data_path, 'r', encoding='utf-8') as f1:
    #     merged_data = json.load(f1)
    
    merged_data = []
    with open(merged_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            merged_data.append(json.loads(line))

    with open(extracted_data_path, 'r', encoding='utf-8') as f2:
        extracted_data = json.load(f2)

    # 检索库集合：只用 original_questions.json 里的问题
    retrieval_set = [item["original_question"].strip() for item in extracted_data]
    # retrieval_set = [item for item in extracted_data]
    passage_lookup = retrieval_set
    passage_lookup_test = extracted_data
    passage_embs = encode_texts(model.question_model, tokenizer, retrieval_set, device)
    faiss_index = build_faiss_index(passage_embs, use_gpu=True)

    results = []
    for item in tqdm(merged_data, desc="Retrieving"):
        query = item["description"].strip()
        # query = item["original_question"].strip()
        src_uid = item.get("src_uid", None)
        query_emb = encode_texts(model.question_model, tokenizer, [query], device)
        query_np = query_emb.cpu().numpy()
        faiss.normalize_L2(query_np)
        D, I = faiss_index.search(query_np, 20)  # 取前20
        
        # stop = input(D)

        retrieved_top20 = [passage_lookup[idx] for idx in I[0]]
        retrieved_top20_test = [passage_lookup_test[idx] for idx in I[0]]
        # print(retrieved_top20)
        # stop = input()
        # stop = input(retrieved_top20_test)
        
        result_save = item
        result_save['retrieve'] = retrieved_top20_test
        
        # stop = input(result_save)

        results.append(result_save)

    with open('/home/xxx/solverank/dpr/dataset/prog_syn_test_nl_retrive_with_code_for_logic_select.jsonl', 'w', encoding='utf-8') as f:
        for item in results:
            json_line = json.dumps(item)
            f.write(json_line + '\n')

    # with open("retrieved_top20_test.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved {len(results)} queries with top-20 retrieved to retrieved_top20.json")

if __name__ == '__main__':
    retrieve_top20()
