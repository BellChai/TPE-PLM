import torch
import torch.nn as nn

from transformers import  GPT2Config, GPT2Model, GPT2Tokenizer
import transformers

from layers.GCN_blocks import *
from models.Plugin import Plugin
from models.Fliter import Filter
from models.Mode_att import Mode_att

transformers.logging.set_verbosity_error()


class PredictModel(nn.Module):

    def init(self):
        # 初始化模块权重
        for name, param in self.llm_model.named_parameters():
            if 'weight' in name:
                if 'ln' in name:
                    param.data.fill_(1.0)
                else:
                    param.data.normal_(mean=0.0, std=0.02)
            elif 'bias' in name:
                param.data.zero_()


    def load_model(self, configs):
        if configs.llm_model == 'GPT2' or configs.llm_model == 'gpt2':
            self.gpt2_config = GPT2Config.from_pretrained('./openai-community/gpt2', local_files_only=True)

            self.gpt2_config.num_hidden_layers = configs.num_hidden_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    './openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    './openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise NotImplementedError(f"Model {configs.llm_model} is not supported.")


    #########################  True Model  #########################
    ################################################################


    def __init__(self, configs):
        super().__init__()

        self.load_model(configs)

        ####################### model freeze    #####################
        #############################################################
        finetune_list = [
            'embed', 'norm', 'scale', # deepseek
            'Norm',         # bert
            'ln_', 'wpe', # 'wte',   # gpt2
            'lora',
            ]
        for name, param in self.llm_model.named_parameters():
            grad_flat = False
            for ft in finetune_list:
                if ft in name:
                    grad_flat = True
                    break
            param.requires_grad = grad_flat

        # net except LLM
        self.llm_mode = configs.llm_mode
        self.config = configs
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_llm = configs.llm_dim
        self.node_num = configs.node_num

        if self.llm_mode == 'mlp':
            self.mlp_backbone = mlp(self.d_llm, self.d_llm, self.d_llm//2)
        elif self.llm_mode == 'gcn':
            self.supports = configs.supports
            self.gcn_layer = GCNLayer(self.d_llm, self.d_llm)
        elif self.llm_mode == 'avwgcn':
            node_dim = 32
            self.node_emb = nn.Parameter(torch.randn(self.node_num, node_dim))
            self.avwgcn_layer = AVWGCN(in_channels=self.d_llm, out_channels=self.d_llm, node_dim=node_dim)
        elif self.llm_mode == 'None':
            pass
        


        
        ####################### tokenizer    #####################
        #############################################################

        self.time_dim = 2 # 5

        self.filter_dim = self.seq_len
        self.enc_dim = self.seq_len * 2 * self.time_dim + self.seq_len       

        self.t2v1 = Time2Vec("cos", hiddem_dim=10)
        self.t2v2 = Time2Vec("cos", hiddem_dim=10)
        self.plugin = Plugin(args=configs, channel=self.node_num)
        self.mlp_enc = mlp(c_in=self.enc_dim, c_out=self.d_llm, c_hid=2*self.d_llm)
        self.mlp_dec = mlp(c_in=self.d_llm, c_out=self.enc_dim, c_hid=self.d_llm//2)
        self.ln_res = linear(c_in=self.enc_dim, c_out=self.pred_len)
        self.mlp_out = mlp(c_in=self.pred_len, c_out=self.pred_len, c_hid=self.pred_len*4)
        # self.filter_layer = Filter(configs, self.filter_dim)
        self.x_mean = nn.Parameter(torch.tensor([0.]))
        self.x_std = nn.Parameter(torch.tensor([1.]))
        if self.config.mode_att_mode!=0: self.mode_att = Mode_att(configs)

        self.enc_mlp = mlp(c_in=self.seq_len, c_out=self.seq_len, c_hid=self.seq_len*4)
        self.mlp_fold = mlp(c_in=self.node_num, c_out=self.node_num//2, c_hid=self.node_num*2)
        self.mlp_unfold = mlp(c_in=self.node_num//2, c_out=self.node_num, c_hid=self.node_num*2)


    def forward(self, x_enc, x_mark_enc, x_mark_dec):   # x_enc: B T N F,         x_mark_enc: B T 5 

        x_enc_copy, x_mark_enc_copy, x_mark_dec_copy = x_enc.clone(), x_mark_enc.clone(), x_mark_dec.clone()
        x_enc_copy2, x_mark_enc_copy2 = x_enc.clone(), x_mark_enc.clone()

        ####################### enc    ####################
        #############################################################
        enc = x_enc.permute(0,2,1) # B N T

        enc = (enc - self.config.train_mean) / self.config.train_std
        enc = self.enc_mlp(enc) # B N T

        # if self.config.filter_mode!=0: 
        #     enc, enc_filter = self.filter_layer(enc)
        #     # enc, enc_filter = self.filter_layer2(enc)
        B, N, D = enc.size()

        # enc = self.plugin_start(enc)

        for i in range(self.time_dim):
            if i==0: x_t2v = self.t2v1(x_mark_enc[:, :, i:i+1])             # B, T, 2
            else: x_t2v = self.t2v2(x_mark_enc[:, :, i:i+1])             # B, T, 2
            x_t2v = x_t2v.reshape(B, 1, self.seq_len*2).expand(-1, N, -1)     # B, N, 2T
            enc = torch.cat([enc, x_t2v], dim=-1)       # [B, N, D+time_dim*2*T]
        llm_enc = self.mlp_enc(enc)                                # # [B, N, d_llm]
        # llm_enc = llm_enc.reshape(B, N//2, -1)
        # llm_enc = self.mlp_fold(llm_enc.permute(0,2,1)).permute(0,2,1)

        ####################### LLM    ####################
        #############################################################
        if self.llm_mode == 'llm':
            # llm_enc = self.rearrange_arr(llm_enc, self.re_indices)
            out = self.llm_model(inputs_embeds=llm_enc)
            llm_dec = out.hidden_states[self.config.num_hidden_layers]
            # out1 = self.llm_model(inputs_embeds=llm_enc[:, :N//2, :]).hidden_states[self.config.num_hidden_layers]
            # out2 = self.llm_model(inputs_embeds=llm_enc[:, N//2:, :]).hidden_states[self.config.num_hidden_layers]
            # llm_dec = torch.cat([out1, out2], dim=1)
        elif self.llm_mode== 'gcn':
            adj_matrix = self.supports
            dec_out = self.gcn_layer(llm_enc, adj_matrix)
        elif self.llm_mode == 'avwgcn':
            dec_out = self.avwgcn_layer(llm_enc, self.node_emb)
        elif self.llm_mode == 'mlp':
            dec_out = self.mlp_backbone(llm_enc)
        elif self.llm_mode == 'none':
            dec_out = llm_enc
        elif self.llm_mode == 'transformer':
            # 创建因果掩码
            mask = torch.tril(torch.ones(self.node_num, self.node_num)).bool().to(llm_enc.device)
            # 前向传播
            llm_dec = self.llm_model(llm_enc, src_mask=mask)

            # dec_out = self.transformer_backbone(llm_enc)
            # dec_out=torch.where(torch.isnan(dec_out),torch.full_like(dec_out,0.),dec_out)

        # llm_dec = self.mlp_unfold(llm_dec.permute(0,2,1)).permute(0,2,1)
        ####################### dec    ####################
        #############################################################

        # dec_out = self.plugin4(llm_enc, x_mark_enc_copy[:, :, :5], llm_dec, x_mark_dec_copy[:, -self.pred_len:, :5])

        dec_out = self.mlp_dec(llm_dec)         # [B, N, enc_dim]
        dec_out = self.ln_res(dec_out + enc)    # [B, N, pred_len]

        dec_out = dec_out.permute(0, 2, 1)      # [B, pred_len, N]
        dec_out_clone = dec_out.clone()
        # 
        if self.config.plugin_mode!=0:
            dec_out = self.plugin(x_enc_copy, x_mark_enc_copy[:, :, :5], dec_out, x_mark_dec_copy[:, -self.pred_len:, :5])

        if self.config.mode_att_mode!=0:
            dec_att_out = self.mode_att(x_enc_copy2.permute(0,2,1), x_mark_enc_copy2[:,0,0:2], dec_out.permute(0,2,1)).permute(0,2,1)         # B N D
            dec_ori_out = dec_out
        else:
            dec_att_out = dec_out
            dec_ori_out = 0

        return dec_att_out, (dec_ori_out, dec_out_clone)


