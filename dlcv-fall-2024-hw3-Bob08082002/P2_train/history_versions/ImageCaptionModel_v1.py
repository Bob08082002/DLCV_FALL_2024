from torch import nn
import torch
import timm
from decoder_v1 import Config, Decoder
import torch.nn.functional as F



class ImageCaptionModel(nn.Module):
    """ ImageCaptionModel only include Text Decoder(Language Model) and Visual Encoder(ViT).
        Doesnt inculde tokenizer."""
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()

        # Text Decoder (Language Model)
        self.decoder_cfg = Config(text_decoder)  # create the Decoder config
        self.decoder = Decoder(self.decoder_cfg)

        # Visual Encoder (ViT)
        self.encoder = timm.create_model(vision_encoder, pretrained=True)
        self.projection_layer = nn.Linear(1024, self.decoder_cfg.n_embd) #self.decoder_cfg.n_embd = 768, project to same dim for concat 


    def forward(self, token_id, images):
        """ image.shape = BS*3*224*224, token_id.shape = BS*K, note that token_id is padded output from tokenizer"""
        # img_feature.shape = torch.Size([BS, 257, 1024]), where 257 = Patch^2 + 1, 1 is for class embedding
        img_feature = self.encoder.forward_features(images)
        # img_embed.shape = torch.Size([BS, 257, 768])
        img_embed = self.projection_layer(img_feature)


        # token_id.shape = (BS, K)
        # img_embed.shape = (BS, 257, 768)
        #print(token_id[0].dtype) #tensor([50256,    32,  9290,   286,   617,  2330,  8122,   351,  1223,   319,  262,  1735, 50256, 50256]) #torch.int64
        #print(img_embed[0].dtype) #torch.float32
        output_prob = self.decoder(token_id, img_embed) # output_prob.shape = torch.Size([BS, K, 50257])

        return output_prob
    
    def autoreg_infer(self, images, step = 30):
        # Implement autoregressive inference here
        img_feature = self.encoder.forward_features(images)
        img_embed = self.projection_layer(img_feature)

        text_id = torch.ones((img_embed.size(0), 1), dtype=torch.long, device=images.device) * 50256

        for i in range(step):
            text_pred = self.decoder(text_id, img_embed)
            text_pred_id = torch.argmax(text_pred, dim=-1)
            text_id = torch.cat((text_id, text_pred_id[:,i].unsqueeze(1)), dim=-1)
            
        return text_id
    def autoreg_infer_beam(self, image, beam, step = 100):
        # Implement autogressive inference with beam search here
        image_feat = self.encoder.forward_features(image)
        image_emb = self.projection_layer(image_feat) # batch x (patches + 1) x 768 (patches = 196, 256...)
        batch_cnt = image_emb.size(0)
        text_id = torch.ones((batch_cnt, 1, 1), dtype=torch.long, device=image.device) * 50256 # batch x context(1) x path
        text_prob = torch.ones((batch_cnt, 1), dtype=torch.float) # batch x path

        for i in range(step):
            text_pred_all = torch.zeros((batch_cnt, 50257, 0))
            path_cnt = text_id.size(2)
            for j in range(path_cnt):
                text_pred_path = F.softmax(self.decoder(text_id[:, :, j], image_emb), dim=-1).detach().cpu() # batch x 1 x 50257
                text_pred_all = torch.cat((text_pred_all, text_pred_path[:, i, :].unsqueeze(2)), dim=-1)
            text_pred_topk = torch.topk(text_pred_all, beam, dim=1)
            text_pred_id = text_pred_topk.indices # batch x beam x path
            text_pred_scores = text_pred_topk.values # batch x beam x path

            text_prob_prev = text_prob.unsqueeze(1).repeat(1, beam, 1) # batch x beam x path
            text_pred_scores = text_prob_prev * text_pred_scores * 2 # batch x beam x path

            new_text_id = text_id.repeat(1, 1, beam).detach().cpu() # batch x context x beam*path
            new_text_pred_id = text_pred_id.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            new_text_pred_scores = text_pred_scores.view(batch_cnt, 1, beam*path_cnt) # batch x 1 x beam*path
            final = torch.cat((new_text_id, new_text_pred_id), dim=1) # batch x context+2 x beam*path
            
            # Sort by probability
            text_prob = torch.zeros((batch_cnt, beam*path_cnt), dtype=torch.float)
            for k in range(batch_cnt):
                sort_score, indexes = new_text_pred_scores[k].squeeze().sort(descending=True)
                final[k] = final[k][:, indexes]
                text_prob[k] = sort_score
            text_id = final.to(image.device)
            text_id = text_id[:, :, :beam]
            text_prob = text_prob[:, :beam]

        return text_id[:,:,0]
    def beam_search(self, img, beams=3, max_length=30):
        self.eval()

        def forward_prob(x, encoder_feature):
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)
            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            for block in self.decoder.transformer.h:
                (x, encoder_feature) = block((x, encoder_feature))
            # Generator
            # 根據seq的最後一個字分類
            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        if img.dim() < 4:
            img = img.unsqueeze(0)
        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.projection_layer(encoder_feature)
        cur_state = torch.tensor([50256]).to(img.device).unsqueeze(1)
        ### Beam Search Start ###
        # get top k words
        next_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = next_probs.shape[-1]
        # 選擇概率最高的beams個單詞作為初始候選序列

        # probs, pred id
        cur_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            # print("current state: ", cur_state)
            next_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_probs = cur_probs.unsqueeze(-1) + next_probs
            cur_probs = cur_probs.flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            # cur_probs / (len(cur_state[0]) + 1) -> nomalized
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            # print("next char: ",next_chars)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()  # 找回屬於哪個beam
            cur_state = cur_state[top_candidates]
            cur_state = torch.cat((cur_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == 50256:
                    ans_ids.append(cur_state[idx].cpu().tolist())
                    # print(cur_probs[idx].item()," / ",len(ans_ids[-1]))
                    ans_probs.append(cur_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()

        # 把50256抽離
        ans_ids[max_idx] = [x for x in ans_ids[max_idx] if x != 50256]
        # print(ans_ids)
        return ans_ids[max_idx]







if __name__=="__main__":
    vision_encoder = "vit_large_patch14_clip_224"
    text_decoder = "../hw3_data/p2_data/decoder_model.bin"
    input_imgs = torch.rand(7, 3, 224, 224)
    token_id = torch.randint(0, 50257, (7, 23))
    model = ImageCaptionModel(vision_encoder, text_decoder)
    ofmap = model(token_id, input_imgs)
    print(ofmap.shape)  # torch.Size([BS, K, 50257])

