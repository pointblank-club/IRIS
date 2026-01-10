
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import torch.nn.functional as F
from torch.utils.data import Dataset


def parse_features(feature_dict):
    """Converts a dictionary of features into a dictionary of floats.""" 
    return {k: float(v) for k, v in feature_dict.items()}

class PassSequenceDataset(Dataset):
    """
    Dataset for the IRis task.
    Each sample consists of program features and the best-performing pass sequence.
    """
    def __init__(self, data, target_metric, max_seq_len):
        super().__init__()
        self.target_metric = target_metric
        self.max_seq_len = max_seq_len

        self.samples = []
        self.pass_vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
        self.feature_keys = []

        self._process_data(data)
        self._build_vocab()
        self._tokenize_sequences()

        self.feature_scaler = StandardScaler()
        self._normalize_features()

    def _process_data(self, data):
        """Groups data by program and finds the best sequence for each.""" 
        if not data:
            return

        programs = {}
        for entry in data:
            prog_name = entry['program']
            if prog_name not in programs:
                programs[prog_name] = []
            programs[prog_name].append(entry)


        for prog_name, entries in programs.items():
            best_entry = min(entries, key=lambda x: x[self.target_metric])
            
            features = parse_features(best_entry['program_features'])
            if not self.feature_keys:
                self.feature_keys = sorted(features.keys())

            ordered_features = [features.get(k, 0.0) for k in self.feature_keys]

            self.samples.append({
                'features': np.array(ordered_features, dtype=np.float32),
                'sequence': best_entry['pass_sequence']
            })

    def _build_vocab(self):
        """Builds a vocabulary of all unique compiler passes.""" 
        pass_idx = len(self.pass_vocab)
        for sample in self.samples:
            for p in sample['sequence']:
                if p not in self.pass_vocab:
                    self.pass_vocab[p] = pass_idx
                    pass_idx += 1
        self.vocab_size = len(self.pass_vocab)

    def _tokenize_sequences(self):
        """Converts pass sequences to token IDs with padding/truncation and SOS/EOS.""" 
        for sample in self.samples:
            seq = sample['sequence']
            token_ids = [self.pass_vocab['<sos>']]
            token_ids.extend([self.pass_vocab.get(p, self.pass_vocab['<unk>']) for p in seq])
            token_ids.append(self.pass_vocab['<eos>'])


            if len(token_ids) < self.max_seq_len:
                token_ids.extend([self.pass_vocab['<pad>']] * (self.max_seq_len - len(token_ids)))
            else:
                token_ids = token_ids[:self.max_seq_len]
                token_ids[-1] = self.pass_vocab['<eos>'] 

            sample['sequence_tokens'] = torch.tensor(token_ids, dtype=torch.long)

    def _normalize_features(self):
        """Fits and transforms program features using StandardScaler.""" 
        if not self.samples:
            self.num_features = 0
            return
        
        all_features = np.array([s['features'] for s in self.samples])
        self.num_features = all_features.shape[1]
        
        if all_features.size > 0:
            scaled_features = self.feature_scaler.fit_transform(all_features)
            for i, sample in enumerate(self.samples):
                sample['features_scaled'] = torch.tensor(scaled_features[i], dtype=torch.float32)
        else: 
             for i, sample in enumerate(self.samples):
                sample['features_scaled'] = torch.zeros(self.num_features, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            sample['features_scaled'],
            sample['sequence_tokens']
        )


class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence.""" 
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PassFormer(nn.Module):
    """
    Transformer-based seq2seq model for IRis.
    Encoder: MLP processing program features.
    Decoder: Transformer generating a pass sequence.
    """
    def __init__(self, vocab_size, num_features, d_model, nhead, num_decoder_layers,
                 dim_feedforward, feature_mlp_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.pass_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        mlp_layers = []
        input_dim = num_features
        for layer_dim in feature_mlp_layers:
            mlp_layers.append(nn.Linear(input_dim, layer_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_dim
        self.feature_mlp = nn.Sequential(*mlp_layers)


        self.feature_projection = nn.Linear(input_dim, d_model)


        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)


        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, features, target_sequence):

        feature_representation = self.feature_mlp(features)
        memory = self.feature_projection(feature_representation).unsqueeze(1) 


        target_emb = self.pass_embedding(target_sequence) * math.sqrt(self.d_model)
        target_emb = self.pos_encoder(target_emb)


        target_mask = nn.Transformer.generate_square_subsequent_mask(target_sequence.size(1)).to(features.device)
        

        target_padding_mask = (target_sequence == 0) 

        decoder_output = self.transformer_decoder(
            tgt=target_emb,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_padding_mask
        )

        return self.fc_out(decoder_output)



def beam_search_decode(model, raw_features_dict, feature_keys, feature_scaler, pass_vocab, device, beam_width=5, max_len=50):
    """Generates a pass sequence using beam search."""
    model.eval()

    # Convert raw_features_dict to a numpy array in the correct order
    feature_vector = np.array([raw_features_dict.get(k, 0.0) for k in feature_keys], dtype=np.float32)

    # Scale the features
    features_scaled = feature_scaler.transform(feature_vector.reshape(1, -1))
    features = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    start_token = pass_vocab['<sos>']
    end_token = pass_vocab['<eos>']

    beams = [([start_token], 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[-1] == end_token:
                new_beams.append((seq, score))
                continue

            input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            with torch.no_grad():
                output_logits = model(features, input_tensor)
            
    
            log_probs = F.log_softmax(output_logits[0, -1, :], dim=0)
            top_k_log_probs, top_k_indices = torch.topk(log_probs, beam_width)

            for i in range(beam_width):
                new_seq = seq + [top_k_indices[i].item()]
                new_score = score + top_k_log_probs[i].item()
                new_beams.append((new_seq, new_score))


        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]


    best_seq, _ = beams[0]
    if best_seq[-1] != end_token:
        best_seq.append(end_token)


    id_to_pass = {v: k for k, v in pass_vocab.items()}
    generated_passes = [id_to_pass.get(tok, '<unk>') for tok in best_seq[1:-1]] 
    
    return generated_passes

def load_model(model_path, device):
    """Loads a trained model and its supplementary data."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    model = PassFormer(
        vocab_size=len(checkpoint['vocab']),
        num_features=len(checkpoint['feature_keys']),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        feature_mlp_layers=config['feature_mlp_layers'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['vocab'], checkpoint['feature_keys'], checkpoint['feature_scaler']
