# %%
import einops
from fancy_einsum import einsum
from einops import rearrange, repeat
from dataclasses import dataclass
from transformer_lens import HookedTransformer
import torch
import torch.nn as nn
import numpy as np
import math
import tqdm.auto as tqdm

# reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

# %%

def gelu_new(
    input
):
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return (
        0.5
        * input
        * (
            1.0
            + torch.tanh(
                np.sqrt(2.0 / np.pi) * (input + 0.044715 * torch.pow(input, 3.0))
            )
        )
    )

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()

class LayerNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual, parallel=True):
        # residual: [batch, position, d_model]
        if self.cfg.debug: print("Residual:", residual.shape)
        if parallel:
            pattern = "batch n_heads position d_model -> batch n_heads position 1"
        else:
            pattern = "batch position d_model -> batch position 1"
        residual = residual - einops.reduce(residual, pattern, "mean")
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), pattern, "mean") + self.cfg.layer_norm_eps).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        if self.cfg.debug: print("Normalized:", residual.shape)
        return normalized

class Embed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        embed = self.W_E[tokens, :] # [batch, position, d_model]
        if self.cfg.debug: print("Embeddings:", embed.shape)
        return embed

class PosEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens):
        # tokens: [batch, position]
        if self.cfg.debug: print("Tokens:", tokens.shape)
        pos_embed = self.W_pos[:tokens.size(1), :] # [position, d_model]
        pos_embed = einops.repeat(pos_embed, "position d_model -> batch position d_model", batch=tokens.size(0))
        if self.cfg.debug: print("pos_embed:", pos_embed.shape)
        return pos_embed

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))

        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))

        self.register_buffer("IGNORE", torch.tensor(-1e5, dtype=torch.float32, device="cuda"))

    def apply_attention(self, normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_pre:", normalized_resid_pre.shape)

        q = einsum("batch n_prev_head query_pos d_model, n_heads d_model d_head -> batch n_prev_head query_pos n_heads d_head", normalized_resid_pre, self.W_Q) + self.b_Q
        k = einsum("batch n_prev_head key_pos d_model, n_heads d_model d_head -> batch n_prev_head key_pos n_heads d_head", normalized_resid_pre, self.W_K) + self.b_K

        attn_scores = einsum("batch n_prev_head query_pos n_heads d_head, batch n_prev_head key_pos n_heads d_head -> batch n_prev_head n_heads query_pos key_pos", q, k)
        attn_scores = attn_scores / math.sqrt(self.cfg.d_head)
        attn_scores = self.apply_causal_mask(attn_scores)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        v = einsum("batch n_prev_head key_pos d_model, n_heads d_model d_head -> batch n_prev_head key_pos n_heads d_head", normalized_resid_pre, self.W_V) + self.b_V

        z = einsum("batch n_prev_head n_heads query_pos key_pos, batch n_prev_head key_pos n_heads d_head -> batch n_prev_head query_pos n_heads d_head", pattern, v)

        attn_out = einsum("batch n_prev_head query_pos n_heads d_head, n_heads d_head d_model -> batch n_prev_head query_pos n_heads d_model", z, self.W_O)
        return attn_out

    def forward(self, normalized_resid_pre, alt_normalized_resid_pre):
        # normalized_resid_pre: [batch, position, d_model]
        attention_input = torch.cat([normalized_resid_pre, torch.unsqueeze(alt_normalized_resid_pre, dim=1)], dim=1)

        attention_output = self.apply_attention(attention_input)
        old_ablated_attention = torch.sum(attention_output, dim=-2)

        orig_attention_output = attention_output[:, [1]]
        alt_attention_output = attention_output[:,[0]]

        n_heads = self.cfg.n_heads
        ablate_mtrx = torch.ones((n_heads, n_heads)) - torch.eye(n_heads)
        # rearrange: batch query_pos n_heads d_model -> batch n_heads query_pos d_model
        new_ablated_attention = einsum(
            "batch query_pos n_heads d_model, keep_heads n_heads -> batch keep_heads query_pos d_model", orig_attention_output, ablate_mtrx
        ) + rearrange(
            alt_attention_output, "bqnd -> bnqd"
        )

        return torch.cat([old_ablated_attention, new_ablated_attention], dim=1) + self.b_O

    def apply_causal_mask(self, attn_scores):
        # attn_scores: [batch, n_heads, query_pos, key_pos]
        mask = torch.triu(torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device), diagonal=1).bool()
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        nn.init.normal_(self.W_out, std=self.cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))

    def forward(self, normalized_resid_mid):
        # normalized_resid_mid: [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_mid:", normalized_resid_mid.shape)
        pre = einsum("batch n_heads position d_model, d_model d_mlp -> batch n_heads position d_mlp", normalized_resid_mid, self.W_in) + self.b_in
        post = gelu_new(pre)
        mlp_out = einsum("batch n_heads position d_mlp, d_mlp d_model -> batch n_heads position d_model", post, self.W_out) + self.b_out
        return mlp_out

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, resid_pre):
        # resid_pre [batch, n_heads, position, d_model]
        normalized_resid_pre = self.ln1(resid_pre)
        attn_out = self.attn(normalized_resid_pre)

        resid_mid = torch.cat([resid_pre, repeat(resid_pre[:,1], "bsd -> bnsd", n=self.cfg.n_heads)]) + attn_out
        
        normalized_resid_mid = self.ln2(resid_mid)
        mlp_out = self.mlp(normalized_resid_mid)
        resid_post = resid_mid + mlp_out
        return resid_post

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(self, normalized_resid_final):
        # normalized_resid_final [batch, position, d_model]
        if self.cfg.debug: print("Normalized_resid_final:", normalized_resid_final.shape)
        logits = einsum("batch n_heads position d_model, d_model d_vocab -> batch n_heads position d_vocab", normalized_resid_final, self.W_U) + self.b_U
        return logits

class DemoTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens, alt_tokens):
        # tokens [batch, position]
        embed = self.embed(tokens)
        pos_embed = self.pos_embed(tokens)
        residual = embed + pos_embed

        alt_embed = self.embed(alt_tokens)
        alt_pos_embed = self.pos_embed(alt_tokens)
        alt_residual = alt_embed + alt_pos_embed

        # in dimension 1, the alt residual is the first elt, followed by the original residual, which is the second elt
        residual = torch.stack([alt_residual, residual], dim=1)

        # output of each block:
        # embeddings + alt embeddings
        # no ablation. apply block to each previous ablated examples. and add new ablated examples.
        # output:
        # no ablation + previous ablations + new ablations, and alt with no ablation
        for block in self.blocks:
            residual = block(residual)
        # shape: no ablation + all ablations + alt no ablation.
        normalized_resid_final = self.ln_final(residual)

        logits = self.unembed(normalized_resid_final)
        # logits have shape [batch, position, logits]
        return logits

# demo_gpt2 = DemoTransformer(Config(debug=False))
# demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)
# demo_gpt2.cuda()

# """Take a test string - the intro paragraph of today's featured Wikipedia article. Let's calculate the loss!"""

# test_string = """Mini scule is a species of microhylid frog endemic to Madagascar that was described in 2019. The scientific name of the species refers to its size, being a pun on the word minuscule. It is very small, measuring only 8.4 to 10.8 mm (0.33 to 0.43 in) in snout–vent length. It has bronze underparts with a brown groin and back of the thigh, cream upperparts with brown flecking, a dark brown side of the head, and a red iris. On the hind feet, the first toe is absent and the second and fifth toes are strongly reduced. The frog is known only from the Sainte Luce Reserve, where it inhabits areas with deep leaf litter near semi-permanent water bodies. Specimens of frogs from Mandena, the Vohimena mountains, the southern Anosy Mountains, and Tsitongambarika may also be of this species. Along with Mini mum and Mini ature, the other two species in its genus, it received media attention when first described due to the wordplay in its scientific name. (Full article...)"""

# test_tokens = reference_gpt2.to_tokens(test_string).cuda()
# demo_logits = demo_gpt2(test_tokens)

# def lm_cross_entropy_loss(logits, tokens):
#     # Measure next token loss
#     # Logits have shape [batch, position, d_vocab]
#     # Tokens have shape [batch, position]
#     log_probs = logits.log_softmax(dim=-1)
#     pred_log_probs = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
#     return -pred_log_probs.mean()
# loss = lm_cross_entropy_loss(demo_logits, test_tokens)
# print(loss)
# print("Loss as average prob", (-loss).exp())
# print("Loss as 'uniform over this many variables'", (loss).exp())
# print("Uniform loss over the vocab", math.log(demo_gpt2.cfg.d_vocab))

# """We can also greedily generate text:"""

# test_string = "Breaking News: President Trump has been impeached by the House of Representatives for abuse of power and obstruction of Congress. The vote was 230 to 197, with 10 Republicans joining all Democrats in voting to impeach. The president is now only the third in American history to be impeached, and the first to be impeached twice. The House will now send the articles of impeachment to the Senate, where a trial will be held to determine whether to remove the president from office. The Senate is expected to begin the trial on"
# for i in tqdm.tqdm(range(100)):
#     test_tokens = reference_gpt2.to_tokens(test_string).cuda()
#     demo_logits = demo_gpt2(test_tokens)
#     test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())
# print(test_string)

# """# Training a Model!

# This is a lightweight demonstration of how you can actually train your own GPT-2 with this code! Here we train a tiny model on a tiny dataset, but it's fundamentally the same code for training a larger/more real model (though you'll need beefier GPUs and data parallelism to do it remotely efficiently, and fancier parallelism for much bigger ones).

# For our purposes, we'll train 2L 4 heads per layer model, with context length 256, for 1000 steps of batch size 8, just to show what it looks like (and so the notebook doesn't melt your colab lol).
# """

# # Commented out IPython magic to ensure Python compatibility.
# if IN_COLAB:
# #     %pip install datasets
# #     %pip install transformers
# import datasets
# import transformers
# import plotly.express as px

# """## Config"""

# batch_size = 8
# num_epochs = 1
# max_steps = 1000
# log_every = 10
# lr = 1e-3
# weight_decay = 1e-2
# model_cfg = Config(debug=False, d_model=256, n_heads=4, d_head=64, d_mlp=1024, n_layers=2, n_ctx=256, d_vocab=reference_gpt2.cfg.d_vocab)

# """
# ## Create Data

# We load in a tiny dataset I made, with the first 10K entries in the Pile (inspired by Stas' version for OpenWebText!)
# """

# dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train")
# print(dataset)
# print(dataset[0]['text'][:100])
# tokens_dataset = tokenize_and_concatenate(dataset, reference_gpt2.tokenizer, streaming=False, max_length=model_cfg.n_ctx, column_name="text", add_bos_token=True, num_proc=4)
# data_loader = torch.utils.data.DataLoader(tokens_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# """## Create Model

# """

# model = DemoTransformer(model_cfg)
# model.cuda()

# """## Create Optimizer
# We use AdamW - it's a pretty standard optimizer.
# """

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# """## Run Training Loop

# """

# losses = []
# print("Number of batches:", len(data_loader))
# for epoch in range(num_epochs):
#     for c, batch in tqdm.tqdm(enumerate(data_loader)):
#         tokens = batch['tokens'].cuda()
#         logits = model(tokens)
#         loss = lm_cross_entropy_loss(logits, tokens)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#         losses.append(loss.item())
#         if c % log_every == 0:
#             print(f"Step: {c}, Loss: {loss.item():.4f}")
#         if c > max_steps:
#             break

# """We can now plot a loss curve!"""

# px.line(y=losses, x=np.arange(len(losses))*(model_cfg.n_ctx * batch_size), labels={"y":"Loss", "x":"Tokens"}, title="Training curve for my tiny demo model!")