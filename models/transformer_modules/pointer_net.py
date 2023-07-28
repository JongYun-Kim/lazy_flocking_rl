import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGenerator(nn.Module):
    def __init__(self):
        super(PointerGenerator, self).__init__()

    def forward(self, decoder_output, encoder_output):
        # decoder_output : (batch_size, seq_len, d_model)
        # encoder_output : (batch_size, seq_len, d_model)
        attention_scores = torch.bmm(decoder_output, encoder_output.transpose(1, 2))  # (batch_size, seq_len, seq_len)
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        return attention_probs


class PointerProbGenerator(nn.Module):

    def __init__(self, d_model, q_fc, k_fc, clip_value=10, dr_rate=0):
        super(PointerProbGenerator, self).__init__()
        self.d_model = d_model  # The dimension of the internal layers
        self.clip_value = clip_value  # Value used for clipping the attention scores

        # Linear layers for transforming the input query and key to the internal dimension
        self.q_fc = copy.deepcopy(q_fc)  # (d_embed_query, d_model)
        self.k_fc = copy.deepcopy(k_fc)  # (d_embed_key,   d_model)

        self.dropout = nn.Dropout(p=dr_rate)  # Dropout layer

    def calculate_attention(self, query, key, mask):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor
        batch_size = query.size(0)  # Get the batch size; TODO remove it

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = self.clip_value * torch.tanh(attention_score)  # Apply clipping to the attention scores
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)  # Apply the mask to the attention scores
        # attention_prob = F.softmax(attention_score, dim=-1)  # Apply softmax to get the attention probabilities
        # attention_prob = self.dropout(attention_prob)  # Apply dropout
        # attention_prob: (n_batch, seq_len_query, seq_len_key) - The attention probabilities
        # attention_score: (n_batch, seq_len_query, seq_len_key) - The attention scores
        return attention_score  # , attention_prob

    def forward(self, *args, query, key, mask=None):
        # query:      (n_batch, seq_len_query, d_embed_query) - Batch of query vectors
        # key:        (n_batch, seq_len_key,   d_embed_key) - Batch of key vectors
        # mask:       (n_batch, seq_len_query, seq_len_key) - Mask tensor

        n_batch = query.size(0)  # Get the batch size

        # Apply the linear transformations to the query and key
        query = self.q_fc(query)  # (n_batch, seq_len_query, d_model)
        key = self.k_fc(key)  # (n_batch, seq_len_key,   d_model)

        attention_score = self.calculate_attention(query, key, mask)  # (n_batch, seq_len_query, seq_len_key)

        return attention_score  # (n_batch, seq_len_query, seq_len_key) - The attention probabilities


class PointerPlaceholder(nn.Module):
    def __init__(self):
        super(PointerPlaceholder, self).__init__()

    def forward(self, *args, query, key, mask=None):
        # Assuming that the desired output shape is the same as that of the original module
        batch_size = query.size(0)
        seq_len_query = query.size(1)
        seq_len_key = key.size(1)

        # Create a tensor of zeros with the same shape as the original output
        out = torch.zeros(batch_size, seq_len_query, seq_len_key)

        # This placeholder function assumes that your model will work correctly with this output.
        # If your model needs different default values or behaviors, you'll have to adjust this.
        return out


class GaussianActionDistGenerator(nn.Module):

    def __init__(self, d_model, q_fc, k_fc, clip_value_mean=1, clip_value_std=10, dr_rate=0):
        super(GaussianActionDistGenerator, self).__init__()
        self.d_model = d_model  # The dimension of the internal layers
        self.clip_value_mean = clip_value_mean  # Value used for clipping the attention scores
        self.clip_value_std = clip_value_std  # Value used for clipping the attention scores

        # Linear layers for transforming the input query and key to the internal dimension
        # LL for mean
        self.q_fc_mean = copy.deepcopy(q_fc)  # (d_embed_query, d_model)
        self.k_fc_mean = copy.deepcopy(k_fc)  # (d_embed_key,   d_model)
        # LL for std
        self.q_fc_std = copy.deepcopy(q_fc)  # (d_embed_query, d_model)
        self.k_fc_std = copy.deepcopy(k_fc)  # (d_embed_key,   d_model)

        self.dropout = nn.Dropout(p=dr_rate)  # Dropout layer

    def calculate_attention(self, query, key, mask, mode='mean'):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores

        if mode == 'mean':
            # Apply clipping to the attention scores using the clip_value; [0, clip_value]
            # attention_score = self.clip_value_mean * torch.sigmoid(attention_score)
            attention_score = self.clip_value_mean * (torch.tanh(attention_score) + 1) / 2
            neutral_action = 0
            if mask is not None:
                # Apply the mask to the attention scores
                attention_score = attention_score.masked_fill(mask == 0, neutral_action)
        elif mode == 'std':
            # Apply clipping to the attention scores using the clip_value; [-clip_value, -u]
            u = 2  # 0
            # attention_score = self.clip_value_std * (torch.sigmoid(attention_score) - 1)
            attention_score = (self.clip_value_std - u) * ((torch.tanh(attention_score)-1)/2) - u
            small_log_std = -10
            if mask is not None:
                # Apply the mask to the attention scores
                attention_score = attention_score.masked_fill(mask == 0, small_log_std)
        else:
            raise ValueError('mode must be either mean or std')

        return attention_score

    def forward(self, *args, query, key, mask=None):
        # query:      (n_batch, seq_len_query, d_embed_query) - Batch of query vectors
        # key:        (n_batch, seq_len_key,   d_embed_key) - Batch of key vectors
        # mask:       (n_batch, seq_len_query, seq_len_key) - Mask tensor

        # Apply the linear transformations to the queries and keys
        query_mean = self.q_fc_mean(query)  # (n_batch, seq_len_query, d_model)
        key_mean = self.k_fc_mean(key)      # (n_batch, seq_len_key,   d_model)
        query_std = self.q_fc_std(query)    # (n_batch, seq_len_query, d_model)
        key_std = self.k_fc_std(key)        # (n_batch, seq_len_key,   d_model)

        # Calculate the attention scores
        # attention_score_mean: (n_batch, seq_len_query, seq_len_key) - The attention scores for the mean
        # attention_score_std:  (n_batch, seq_len_query, seq_len_key) - The attention scores for the std
        attention_score_mean = self.calculate_attention(query_mean, key_mean, mask, mode='mean')
        attention_score_std = self.calculate_attention(query_std, key_std, mask, mode='std')

        return attention_score_mean, attention_score_std  # (n_batch, seq_len_query, seq_len_key)s


class GaussianActionDistPlaceholder(nn.Module):
    def __init__(self):
        super(GaussianActionDistPlaceholder, self).__init__()

    def forward(self, *args, query, key, mask=None):
        # Assuming that the desired output shape is the same as that of the original module
        batch_size = query.size(0)
        seq_len_query = query.size(1)
        seq_len_key = key.size(1)

        # Create a tensor of zeros with the same shape as the original output
        out_mean = torch.zeros(batch_size, seq_len_query, seq_len_key)
        out_std = torch.zeros(batch_size, seq_len_query, seq_len_key)

        # This placeholder function assumes that your model will work correctly with this output.
        # If your model needs different default values or behaviors, you'll have to adjust this.
        return out_mean, out_std


class FakeMeanGenerator(GaussianActionDistGenerator):
    def forward(self, *args, query, key, mask=None):
        # query:      (n_batch, seq_len_query, d_embed_query) - Batch of query vectors
        # key:        (n_batch, seq_len_key,   d_embed_key) - Batch of key vectors
        # mask:       (n_batch, seq_len_query, seq_len_key) - Mask tensor

        # Apply the linear transformations to the queries and keys
        query_mean = self.q_fc_mean(query)  # (n_batch, seq_len_query, d_model)
        key_mean = self.k_fc_mean(key)      # (n_batch, seq_len_key,   d_model)

        # Calculate the attention scores
        # attention_score_mean: (n_batch, seq_len_query, seq_len_key) - The attention scores for the mean
        attention_score_mean = self.calculate_attention(query_mean, key_mean, mask, mode='mean')
        # Get the std as a constant tensor: small log_std (meaning std close to 0)
        attention_score_log_std_constant = torch.ones_like(attention_score_mean) * (-self.clip_value_std)
        # if mask is not None:
        #     attention_score_log_std_constant = attention_score_log_std_constant.masked_fill(mask == 0, 0)

        return attention_score_mean, attention_score_log_std_constant  # (n_batch, seq_len_query, seq_len_key)s


class MeanGenerator(PointerProbGenerator):

    def calculate_attention(self, query, key, mask):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores
        # Apply clipping to the attention scores using the clip_value; [0, clip_value]
        # attention_score = self.clip_value_mean * torch.sigmoid(attention_score)
        attention_score = self.clip_value * (torch.tanh(attention_score) + 1) / 2
        neutral_action = 0
        if mask is not None:
            # Apply the mask to the attention scores
            attention_score = attention_score.masked_fill(mask == 0, neutral_action)

        return attention_score


class StdGenerator(PointerProbGenerator):

    def calculate_attention(self, query, key, mask):
        # query:  (n_batch, seq_len_query, d_model) - Batch of query vectors
        # key:    (n_batch, seq_len_key,   d_model) - Batch of key vectors
        # mask:   (n_batch, seq_len_query, seq_len_key) - Mask tensor

        d_k = key.shape[-1]  # Get the last dimension of the key
        attention_score = torch.matmul(query, key.transpose(-2, -1))  # Calculate the dot product: (Q x K^T)
        attention_score = attention_score / math.sqrt(d_k)  # Scale the attention scores
        # Apply clipping to the attention scores using the clip_value; [-clip_value, 0]
        # attention_score = self.clip_value * (torch.sigmoid(attention_score) - 1)
        attention_score = self.clip_value * (torch.tanh(attention_score) - 1) / 2
        small_log_std = -10
        if mask is not None:
            # Apply the mask to the attention scores
            attention_score = attention_score.masked_fill(mask == 0, small_log_std)

        return attention_score


