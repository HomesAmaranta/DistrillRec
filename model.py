import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutputWithPast


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

        # model和tokenizer设置
        model = AutoModelForCausalLM.from_pretrained(self.base_model,
                                                     load_in_8bit=True,
                                                     torch_dtype=torch.float16,
                                                     local_files_only=True,
                                                     cache_dir='/root/autodl-tmp/')
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model,
                                                       load_in_8bit=True,
                                                       torch_dtype=torch.float16,
                                                       local_files_only=True,
                                                       cache_dir='/root/autodl-tmp/')

        # 加载lora配置
        self.model = get_peft_model(model, peft_config)

        # 得到输入的id和mask
        instruct = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nGiven the user’s purchase history, predict next possible item to be purchased.\n\n### Input:\n"
        self.instruct_ids, self.instruct_mask = self.tokenizer(instruct,
                                                               # 获取两个值吗？
                                                               truncation=True, padding=False, return_tensors='pt', add_special_tokens=False).values()
        response = "\n### Response:\n"
        self.response_ids, self.response_mask = self.tokenizer(response,
                                                               truncation=True, padding=False, return_tensors='pt', add_special_tokens=False).values()

        # 嵌入层的设置
        self.item_embed = nn.Embedding.from_pretrained(
            self.args["item_embed"], freeze=True)  # 将SASRec的嵌入层权重加载进来，并冻结
        self.item_proj = nn.Linear(
            self.input_dim, self.model.config.hidden_size)  # 不确定是否可以得到config
        self.score = nn.Linear(
            self.model.config.hidden_size, self.output_dim, bias=False)

    def predict(self, inputs, inputs_mask):
        bs = inputs.shape[0]
        instruct_embeds = self.model.model.embed_tokens(self.instruct_ids.cuda()).expand(bs, -1, -1)
        response_embeds = self.model.model.embed_tokens(
            self.response_ids.cuda()).expand(bs, -1, -1)
        instruct_mask = self.instruct_mask.cuda().expand(bs, -1)
        response_mask = self.response_mask.cuda().expand(bs, -1)

        inputs = self.item_proj(self.item_embed(inputs))
        inputs = torch.cat([instruct_embeds, inputs, response_embeds], dim=1)
        attention_mask = torch.cat(
            [instruct_mask, inputs_mask, response_mask], dim=1)
        # assert attention_mask.size()[0] == inputs.size()[0] and attention_mask.size()[1] == inputs.size()[1]

        outputs = self.model(inputs_embeds=inputs,
                             attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, -1]
        pooled_logits = self.score(pooled_output)

        return outputs, pooled_logits.view(-1, self.output_dim)

    def forward(self, inputs, inputs_mask):
        outputs, pooled_logits = self.predict(inputs, inputs_mask)

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
