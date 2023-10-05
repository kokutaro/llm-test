import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft.utils.config import TaskType
from peft.utils.other import prepare_model_for_int8_training
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig

CUTOFF_LEN = 256
model_name = "cyberagent/open-calm-7b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-calm-7b"
output_dir = "lora-calm-7b-results2"

data = load_dataset(dataset)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(prompt, tokenizer):
   result = tokenizer(
      prompt + "<|endoftext|>",
      truncation = True,
      max_length = CUTOFF_LEN,
      padding = False
   )
   return {
      "input_ids": result["input_ids"],
      "attention_mask": result["attention_mask"]
   }

def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
)

lora_config = LoraConfig(
    r= 8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
# モデルの前処理
model = prepare_model_for_int8_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

eval_steps = 200
save_steps = 200
logging_steps = 20

VAL_SET_SIZE = 2000

# 学習データと検証データの準備
train_val = data["train"].train_test_split( # type: ignore
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data, # type: ignore
    eval_dataset=val_data, # type: ignore
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",  # type: ignore
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 学習の実行
model.config.use_cache = False
trainer.train() 
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(peft_name)

# def load_model():
#   tokenizer = AutoTokenizer.from_pretrained(model_name)
#   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
#   return tokenizer,model

# def main():
#     print("Loading...")
#     tokenizer,model = load_model()
#     while True:
#       prompt = input("Prompt: ")
#       prompt = f"ユーザー:{prompt}<NL>システム:"
#       inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#       with torch.no_grad():
#           tokens = model.generate(
#               **inputs,
#               max_new_tokens=128,
#               min_new_tokens=128,
#               do_sample=True,
#               temperature=0.8,
#               pad_token_id=tokenizer.pad_token_id,
#           )
#       prompt = prompt.replace("<NL>", "\n")
#       output = tokenizer.decode(tokens[0], skip_special_tokens=True).replace("<NL>", "\n")
#       print(output[len(prompt):])


# if __name__ == "__main__":
#     main()
