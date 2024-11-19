import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from torch.cuda.amp import GradScaler, autocast
import numpy as np

class RecSys(nn.Module):
    def __init__(self, **args):
        super(RecSys, self).__init__()
        self.args = args
        self.input_dim, self.output_dim = args['input_dim'], args['output_dim']
        self.base_model = args["base_model"]

        # lora配置
        # 参照https://github.com/QwenLM/Qwen2.5/blob/main/examples/llama-factory/qwen2-7b-lora-sft.yaml
        peft_config = LoraConfig(task_type='FEATURE_EXTRACTION', target_modules=[
                                 "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], r=16, lora_alpha=16, lora_dropout=0.05)

        config = AutoConfig.from_pretrained(self.args['base_model'], output_hidden_states=True)
        # model和tokenizer设置
        print("model name:",self.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model,
                                                     config=config,
                                                     # load_in_8bit=True,
                                                     # torch_dtype=torch.float16,
                                                     local_files_only=True,
                                                     cache_dir='/root/autodl-tmp/')
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model,
                                                       # load_in_8bit=True,
                                                       # torch_dtype=torch.float16,
                                                       local_files_only=True,
                                                       cache_dir='/root/autodl-tmp/')
        print("model:",self.model)
        # 加载lora配置
        self.model = get_peft_model(self.model, peft_config)

        # 得到输入的id和mask
        instruct = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven the user’s purchase history, predict next possible item to be purchased.\n\n### Input:\n"
        self.instruct_ids, self.instruct_mask = self.tokenizer(instruct,
                                                               # 获取两个值吗？
                                                               truncation=True, padding=False, return_tensors='pt', add_special_tokens=False).values()
        response = "\n### Response:\n"
        self.response_ids, self.response_mask = self.tokenizer(response,
                                                               truncation=True, padding=False, return_tensors='pt', add_special_tokens=False).values()

        self.embed_tokens = self.model.get_input_embeddings()
        # 嵌入层的设置
        self.item_embed = nn.Embedding.from_pretrained(
            self.args["item_embed"], freeze=True)  # 将SASRec的嵌入层权重加载进来，并冻结
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.item_embed.to(self.device)
        
        self.item_proj = nn.Linear(
            self.input_dim, self.model.config.hidden_size)  # 不确定是否可以得到config
        self.item_proj.to(self.device)
        
        self.score = nn.Linear(
            self.model.config.hidden_size, self.output_dim, bias=False)
        self.score.to(self.device)
        

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        # instruct_embeds = self.model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        instruct_embeds = self.embed_tokens(
            self.instruct_ids.cuda()).expand(bs, -1, -1)
        # response_embeds = self.model.model.embed_tokens(self.response_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.embed_tokens(
            self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)
        
        # print(self.device)
        # print(inputs.device)
        # print(next(self.item_proj.parameters()).device)

        inputs = self.item_proj(self.item_embed(inputs))
        print("inputs:",inputs)
        print("instruct_embeds:",instruct_embeds)
        print("response_embeds:",response_embeds)
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat(
            [instruct_mask, inputs_mask, response_mask], dim=1)
        # assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]
        print("attention_mask.size()[0]:",attention_mask.size()[0])
        print("inputs.size()[0]:",inputs.size()[0])
        print("attention_mask.size()[1]:",attention_mask.size()[1])
        print("inputs.size()[1]:",inputs.size()[1])
        
        print("inputs has nan:",torch.isnan(inputs).any())
        # 判断张量中各个元素是否等于0
        contains_zero = torch.eq(inputs, 0)  # 返回布尔张量

        # 判断是否存在0
        exists_zero = torch.any(contains_zero)  # 返回布尔值
        print("exists_zero",exists_zero)

        # print("inputs.dtype:",inputs.dtype)
        # print("attention_mask.dtype:",attention_mask.dtype)
        # print("inputs.float().dtype:",inputs.float().dtype)
        with autocast():  # 使用自动混合精度
            outputs = self.model(inputs_embeds=inputs,
                             attention_mask=attention_mask, return_dict=True)
        
        print("outputs.hidden_states:",dir(outputs))
        # hs=torch.tensor(outputs.hidden_states)
        # print("outputs.hidden_states.shape:",dir(outputs.hidden_states))
        pooled_output = outputs.hidden_states[-1]
        print("pooled_output",pooled_output)
        print("pooled_output.shape",pooled_output.shape)
        pooled_output = pooled_output[:,-1]
        # print("pooled_output",pooled_output)
        print("pooled_output.shape",pooled_output.shape)
        pooled_logits = self.score(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)
    
    def predict2(self, inputs, inputs_mask):
        inputs=["你好","有什么可以帮助你的吗","中午吃什么","你喜欢我吗"]
        X = self.tokenizer(inputs,padding=True,truncation=True,return_tensors="pt")
        print("X:\n",X)
        
       
        with autocast():  # 使用自动混合精度
            outputs = self.model.generate(**X)
        print("outputs:",outputs)
        for i in range(outputs.shape[0]):
            generated_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            print("generated_text:",generated_text)
        # print("outputs.hidden_states:",outputs.hidden_states)
        # pooled_output = outputs.hidden_states[-1]
        # print("pooled_output",pooled_output)
        # pooled_output = pooled_output[:,-1]
        # # print("pooled_output",pooled_output)
        # print("pooled_output.shape",pooled_output.shape)
        # pooled_logits = self.score(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask):
        outputs, pooled_logits = self.predict2(inputs, inputs_mask)

        # loss = None
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(pooled_logits, labels.view(-1))

        # return SequenceClassifierOutputWithPast(
        #     loss=loss,
        #     logits=pooled_logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # )

        return pooled_logits
