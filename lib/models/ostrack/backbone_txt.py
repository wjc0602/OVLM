from torch import nn
from transformers import RobertaModel


class FeatureResizer(nn.Module):
    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class BackboneTxt(nn.Module):
    def __init__(self, text_encoder_type="roberta-base", freeze_text_encoder=True):
        super(BackboneTxt, self).__init__()
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        self.txt_proj = FeatureResizer(
            input_feat_size=self.text_encoder.config.hidden_size,
            output_feat_size=768,
            dropout=0.1,
        )

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    def forward(self, text_ids, text_masks):
        text_embeds = self.text_encoder.embeddings(input_ids=text_ids)
        device = text_embeds.device
        input_shape = text_masks.size()
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text_masks, input_shape, device)

        for layer in self.text_encoder.encoder.layer:
            text_embeds = layer(text_embeds, extend_text_masks)[0]
        text_embeds = self.txt_proj(text_embeds)
        return text_embeds, extend_text_masks


def build_backbone_txt(cfg, hidden_dim):
    model = BackboneTxt()
    return model
