import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from . import SASRecP, BERT4Rec
import random
import numpy as np
import math
import torch
import torch.nn.functional as F

num_train_timesteps = 1000



class Scheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.num_train_timesteps = num_train_timesteps
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_bar_sqrt = alphas_cumprod.sqrt()

        # Store old values.
        alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
        alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

        # Shift so the last timestep is zero.
        alphas_bar_sqrt -= alphas_bar_sqrt_T

        # Scale so the first timestep is back to the old value.
        alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

        # Convert alphas_bar_sqrt to betas
        alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
        alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
        alphas = torch.cat([alphas_bar[0:1], alphas])
        betas = 1 - alphas
        
        self.betas = betas

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps=None
    ):
        bsz = original_samples.shape[0]
        if timesteps is None:
            timesteps = torch.randint(
                low=0, 
                high=self.num_train_timesteps, 
                size=(bsz,),
                device=original_samples.device
            )
        
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples, timesteps


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimDiff(SequentialRecommender):

    def __init__(self, config, dataset):
        super(SimDiff, self).__init__(config, dataset)

        # load parameters info
        self.initializer_range = config['initializer_range']

        self.mask_ratio = config['mask_ratio']
        self.generate_method = config['generate_method']

        self.mask_token = self.n_items
        self.hidden_size = config['hidden_size']  # same as embedding_size

        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.con_loss_fct = nn.CrossEntropyLoss()
        self.con_sim = Similarity(temp=0.05)

        self.mask_strategy = config['mask_strategy']

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_vocab_chunk_size = int(config['loss_vocab_chunk_size']) if 'loss_vocab_chunk_size' in config else 256
        self.n_embedding = config['n_embedding']
        self.n_sampling = config['n_sampling']
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1

        self.encoder = SASRecP(config, dataset, self.item_embedding).to(config['device'])

        self.generator = BERT4Rec(config, dataset, self.item_embedding).to(config['device'])
        self.generator.trm_encoder = self.encoder.trm_encoder
        
        self.encode_loss_weight = config['encoder_loss_weight']
        self.con_loss_weight = config['contrastive_loss_weight']
        self.generate_loss_weight = config['generate_loss_weight']
        
        self.scheduler = Scheduler(num_train_timesteps)

        self.time_embed = nn.Embedding(num_train_timesteps, config['hidden_size'])
        
        self.apply(self._init_weights)
        self.encoder.apply(self._init_weights)

        self.recall, self.recall_n = 0, 0

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_parameters(self):
        self.apply(self._init_weights)
        self.encoder.apply(self._init_weights)
        self.generator.apply(self._init_weights)
        if hasattr(self, 'time_embed'):
            self.time_embed.apply(self._init_weights)
        if hasattr(self.item_embedding, 'reset_parameters'):
            self.item_embedding.reset_parameters()


    def forward(self, item_seq, item_seq_len):
        emb_weight = self.item_embedding.weight[:self.n_items].T
        
        masked_indices = None
        
        seq_emb = self.item_embedding(item_seq)
        
        batch_size, seq_len, _ = seq_emb.shape
        top_n_indices_list = []

        # Memory optimization: compute each sequence step independently to avoid
        # allocating a huge [batch_size, seq_len, n_items] similarity tensor.
        batch_arange = torch.arange(batch_size, device=item_seq.device)
        for i in range(seq_len):
            current_step_emb = seq_emb[:, i, :]  # [batch_size, hidden_size]
            sims_i = torch.matmul(current_step_emb, emb_weight)  # [batch_size, n_items]

            # Mask self-item to avoid selecting itself as the nearest neighbor.
            sims_i[batch_arange, item_seq[:, i]] = torch.finfo(emb_weight.dtype).min

            _, top_n_i = torch.topk(sims_i, k=self.n_embedding, dim=-1)
            top_n_indices_list.append(top_n_i)

        top_n_indices = torch.stack(top_n_indices_list, dim=1)  # [batch_size, seq_len, n_embedding]

        top_n_embeds = self.item_embedding(top_n_indices)  # [batch_size, seq_len, n, hidden_dim]

        noise = top_n_embeds.mean(dim=2)  # [batch_size, seq_len, hidden_dim]

        replaced_items, timesteps = self.scheduler.add_noise(seq_emb, noise)
        noise_embeds = replaced_items.view(batch_size, seq_len, -1)
        
        timesteps_full = timesteps.view(batch_size, 1)
        timestep_embeddings = self.time_embed(timesteps_full)
        logits, generate_loss, seq_output = self.generator.predictSeq_diffusion(item_seq, noise_embeds + timestep_embeddings, None)
        
        
        # Avoid allocating an extra full-vocab softmax tensor.
        max_logits, topk = torch.topk(logits, k=self.n_sampling, dim=-1)
        max_probs = max_logits[..., 0]

        # Do not allow padding positions to be selected for replacement.
        valid_positions = item_seq != 0
        max_probs = max_probs.masked_fill(~valid_positions, torch.finfo(max_probs.dtype).min)

        # Dynamically set mask length based on each sequence's real length.
        valid_lengths = valid_positions.sum(dim=1)
        dynamic_mask_lengths = (valid_lengths.float() * self.mask_ratio).long()
        dynamic_mask_lengths = torch.where(
            (valid_lengths > 0) & (dynamic_mask_lengths == 0),
            torch.ones_like(dynamic_mask_lengths),
            dynamic_mask_lengths,
        )
        dynamic_mask_lengths = torch.minimum(dynamic_mask_lengths, valid_lengths)

        masked_indices = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=logits.device)
        max_dynamic_mask_length = int(dynamic_mask_lengths.max().item())
        if max_dynamic_mask_length > 0:
            _, topk_indices = torch.topk(max_probs, k=max_dynamic_mask_length, dim=1)
            rank = torch.arange(max_dynamic_mask_length, device=logits.device).unsqueeze(0)
            take_mask = rank < dynamic_mask_lengths.unsqueeze(1)
            masked_indices.scatter_(1, topk_indices, take_mask)
        

        pos_index = topk[..., 0]   # shape: [batch, seq_len]
        neg_index = topk[..., -1]   # shape: [batch, seq_len]
        
        pos_seqs = item_seq.clone()
        pos_seqs[masked_indices] = pos_index[masked_indices]
        
        neg_seqs = item_seq.clone()
        neg_seqs[masked_indices] = neg_index[masked_indices]
        
        
        
        encode_output = self.encoder.forward(item_seq, item_seq_len)
        
        return encode_output, generate_loss, pos_seqs, neg_seqs


    def calculate_con_loss(self, seq_output, seq_output_1):
        logits_0 = seq_output[:, -1:, :].mean(dim=1)
        logits_1 = seq_output_1[:, -1:, :].mean(dim=1)

        # Use normalized matmul to avoid a [B, 2B, D] broadcasted intermediate.
        logits_0 = F.normalize(logits_0, dim=-1)
        logits_1 = F.normalize(logits_1, dim=-1)
        cos_sim = torch.matmul(logits_0, logits_1.transpose(0, 1)) / self.con_sim.temp

        labels = torch.arange(logits_0.size(0)).long().to(self.device)
        con_loss = self.con_loss_fct(cos_sim, labels)

        return con_loss

    def _chunked_ce_loss(self, seq_emb, target_ids):
        # Compute CE exactly via -(x_y - logsumexp(x)) without materializing full [N, n_items] logits.
        valid_mask = target_ids != 0
        if not torch.any(valid_mask):
            return seq_emb.new_zeros(())

        seq_valid = seq_emb[valid_mask]
        target_valid = target_ids[valid_mask]
        item_table = self.item_embedding.weight[:-1, :]

        pos_emb = item_table[target_valid]
        pos_logits = (seq_valid * pos_emb).sum(dim=-1)

        running_lse = None
        total_items = item_table.size(0)
        for start in range(0, total_items, self.loss_vocab_chunk_size):
            end = min(start + self.loss_vocab_chunk_size, total_items)
            chunk_logits = torch.matmul(seq_valid, item_table[start:end].transpose(0, 1))
            chunk_lse = torch.logsumexp(chunk_logits, dim=1)
            running_lse = chunk_lse if running_lse is None else torch.logaddexp(running_lse, chunk_lse)

        return (running_lse - pos_logits).mean()


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, generate_loss, pos_seqs, neg_seqs = self.forward(item_seq, item_seq_len)

        pos_items = interaction[self.POS_ITEM_ID]

        item_label = item_seq[:, 1:]
        pad = pos_items.unsqueeze(-1)
        item_labeln = torch.cat((item_label, pad), dim=-1).long().to(self.device)
        seq_emb = seq_output.view(-1, self.hidden_size)  # [batch*seq_len hidden_size]
        pos_ids_l = torch.squeeze(item_labeln.view(-1))
        encode_loss = self._chunked_ce_loss(seq_emb, pos_ids_l)

        con_loss = torch.tensor(0)
        if self.con_loss_weight != 0:
            # Used to calculate contrastive loss
            seq_output_1 = self.encoder.forward(pos_seqs, item_seq_len)
            seq_output_2 = self.encoder.forward(neg_seqs, item_seq_len)
            
            con_loss = self.calculate_con_loss(seq_output, torch.cat([seq_output_1, seq_output_2], dim=0))
        return self.encode_loss_weight * encode_loss, self.con_loss_weight * con_loss, self.generate_loss_weight * generate_loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]

        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        # print(self.ITEM_SEQ)
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output_raw = self.encoder.forward(item_seq, item_seq_len)
        seq_output = seq_output_raw[:, -1, :].squeeze(1)

        test_items_emb = self.item_embedding.weight[:-1, :]
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]

        return scores



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
