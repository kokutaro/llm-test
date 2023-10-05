import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

def load_model():
  tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
  model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
  return tokenizer,model

def main():
    print("Loading...")
    tokenizer,model = load_model()
    prompt = ""
    while True:
      question = input("Prompt: ")
      if question == "clear":
         prompt = ""
         continue
      prompt = prompt + f"ユーザー:{question}<NL>システム:"
      token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
      with torch.no_grad():
          tokens = model.generate(
              token_ids,
              max_new_tokens=128,
              min_new_tokens=128,
              do_sample=True,
              temperature=0.8,
              pad_token_id=tokenizer.pad_token_id,
              bos_token_id=tokenizer.bos_token_id,
              eos_token_id=tokenizer.eos_token_id,
          )

      output = tokenizer.decode(tokens.tolist()[0])
      prompt = prompt + output + "<NL>"
      output = output.replace("<NL>", "\n").replace("</s>", "")
      print(output)

if __name__ == "__main__":
    main()
