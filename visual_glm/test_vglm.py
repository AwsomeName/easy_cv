from transformers import AutoTokenizer, AutoModel
import torch

# tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
# model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained("/home/lc/models/THUDM/visualglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/lc/models/THUDM/visualglm-6b", torch_dtype=torch.float16).cuda()
image_path = "../YOLO/zidane.jpg"
response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
print(response)
response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
print(response)
