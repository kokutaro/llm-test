import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "cyberagent/open-calm-3b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-calm-7b"
output_dir = "lora-calm-7b-results"

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

# トークンナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRAモデルの準備
model = PeftModel.from_pretrained(
    model, 
    peft_name, 
    device_map="auto"
)

# 評価モード
model.eval()

# プロンプトテンプレートの準備
def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:"""

# テキスト生成関数の定義
def generate(instruction,input=None,maxTokens=256):
    # 推論
    prompt = generate_prompt({'instruction':instruction,'input':input})
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(
        input_ids=input_ids, 
        max_new_tokens=maxTokens, 
        do_sample=True,
        temperature=0.7, 
        top_p=0.75, 
        top_k=40,         
        no_repeat_ngram_size=2,
    )
    outputs = outputs[0].tolist()

    # EOSトークンにヒットしたらデコード完了
    if tokenizer.eos_token_id in outputs:
        eos_index = outputs.index(tokenizer.eos_token_id)
        decoded = tokenizer.decode(outputs[:eos_index])

        # レスポンス内容のみ抽出
        sentinel = "### Response:"
        sentinelLoc = decoded.find(sentinel)
        if sentinelLoc >= 0:
            print(decoded[sentinelLoc+len(sentinel):])
        else:
            print('Warning: Expected prompt template to be emitted.  Ignoring output.')
    else:
        print('Warning: no <eos> detected ignoring output')

while True:
    prompt = input("Prompt: ")
    generate(prompt)
