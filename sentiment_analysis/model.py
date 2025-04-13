import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Thêm mã hóa vị trí vào nhúng mã thông báo để giới thiệu khái niệm về thứ tự từ.
    """
    def __init__(self, embedding_dim, max_seq_length=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, embedding_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
   Cơ chế multi-head attention cho phép mô hình cùng chú ý đến thông tin
từ các không gian biểu diễn khác nhau.
    """
    def __init__(self, embedding_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            mask: Optional mask for padding tokens
        """
        x = self.layer_norm(x)
        
        batch_size, seq_len, _ = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
        context = context.reshape(batch_size, seq_len, self.embedding_dim)
        
        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network, applied to each position separately and identically.
    Bao gồm hai phép biến đổi tuyến tính với hàm kích hoạt ReLU hoặc GELU ở giữa.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation='gelu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = self.layer_norm(x)
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Lớp mã hóa transformer bao gồm multi-head attention và feed-forward network.
    """
    def __init__(self, embedding_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForward(embedding_dim, ff_dim, dropout, activation='gelu')
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class FocalLoss(nn.Module):
    """
    Focus Loss để giải quyết tình trạng mất cân bằng lớp hiệu quả hơn so với mất CE có trọng số.
Tập trung nhiều hơn vào các ví dụ khó phân loại.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # trọng số cho các lớp
        self.gamma = gamma  # hệ số điều chỉnh
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        inputs: [B, C] where B is batch size and C is number of classes
        targets: [B] where each value is class index (0 to C-1)
        """
        # Áp dụng log_softmax để có độ ổn định về số
        log_probs = F.log_softmax(inputs, dim=-1)
        
        batch_size = targets.size(0)
        log_p = log_probs.gather(1, targets.unsqueeze(1))
        log_p = log_p.view(-1)
        
        probs = log_p.exp()
        
        focal_term = (1 - probs).pow(self.gamma)
        loss = -focal_term * log_p
        
        if self.alpha is not None:
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class SentimentTransformer(nn.Module):
    """
    Mô hình chuyển đổi hoàn chỉnh để phân loại tình cảm với 4 nhãn:
    Negative (0), Positive (1), Neutral (2), Irrelevant (3)
    """
    def __init__(self, vocab_size, embedding_dim=512, num_heads=8, num_layers=6, 
                 ff_dim=2048, max_seq_length=512, dropout=0.1, num_classes=4):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        
        # Chồng lớp mã hóa transformer
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len], containing token ids
            mask: Optional mask for padding tokens
        Returns:
            logits for sentiment classification
        """
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Classification
        logits = self.classifier(x)  # [batch_size, num_classes]
        
        return logits
    
    def classify_text(self, token_ids, mask=None):
        """
        Phân loại văn bản thành các loại 
        
        Args:
            token_ids: Tensor của token ids [batch_size, seq_len]
            mask: Optional attention mask
        
        Returns:
            Predicted class indices and labels
        """
        labels = {0: 'Negative', 1: 'Positive', 2: 'Neutral', 3: 'Irrelevant'}
        
        with torch.no_grad():
            logits = self(token_ids, mask)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
        return predictions, [labels[p.item()] for p in predictions]


# TextClassifier được cải tiến với khả năng điều chỉnh tốt hơn cho các tập dữ liệu mất cân bằng
class TextClassifier(nn.Module):
    """
    TextClassifier sử dụng kiến ​​trúc transformer để phân tích tình cảm
với 4 nhãn: Negative (0), Positive (1), Neutral (2), Irrelevant (3)
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=4,
                 d_ff=1024, max_seq_len=128, num_classes=4, dropout=0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.embed_dropout = nn.Dropout(dropout)
        
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout + 0.1),  # Increased dropout in classifier
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize all weights in the model"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass of the model
        
        Args:
            input_ids: Tensor of token ids [batch_size, seq_len]
            attention_mask: Optional attention mask for padding tokens
            
        Returns:
            Logits for classification [batch_size, num_classes]
        """
        batch_size, seq_len = input_ids.size()
        
        if attention_mask is not None:
            # Chuyển 2D mask [batch_size, seq_len] sang 4D mask [batch_size, 1, seq_len, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, -1, seq_len, -1)
            mask = mask.to(dtype=torch.bool)
        else:
            mask = None
            
        x = self.embedding(input_ids)  # [batch_size, seq_len, d_model]
        x = self.embed_dropout(x)  # Apply dropout to embeddings
        x = self.pos_encoding(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if attention_mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=attention_mask.device)
            attention_mask = torch.cat((cls_mask, attention_mask), dim=1)
            # Recreate the 4D mask
            seq_len = seq_len + 1  # +1 for CLS token
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, -1, seq_len, -1)
            mask = mask.to(dtype=torch.bool)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Classification based on [CLS] token
        cls_output = x[:, 0]  # Get CLS token output
        
        # Classification layer
        logits = self.classifier(cls_output)
        
        return logits
