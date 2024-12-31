from torch import nn
import torch
import timm
from decoder_v2 import Config, Decoder


class ImageCaptionModel(nn.Module):
    """ ImageCaptionModel only include Text Decoder(Language Model) and Visual Encoder(ViT).
        Doesnt inculde tokenizer."""
    def __init__(self, vision_encoder, text_decoder):
        super().__init__()

        # Text Decoder (Language Model)
        self.decoder_cfg = Config(text_decoder)  # create the Decoder config
        self.decoder = Decoder(self.decoder_cfg)

        # Visual Encoder (ViT)
        self.encoder = timm.create_model(vision_encoder, pretrained=True, num_classes=0)
        self.projection_layer = nn.Linear(1024, 768) #self.decoder_cfg.n_embd = 768, project to same dim for concat 


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
        output_prob = self.decoder(token_id, img_embed) # output_prob.shapde = torch.Size([BS, K, 50257])

        return output_prob
    

    def autoreg_greedy(self, images, max_token = 20):
        """Assume BS = 1"""
        START_TOKEN = 50256
        END_TOKEN = START_TOKEN

        # img_feature.shape = torch.Size([BS, 257, 1024]), where 257 = Patch^2 + 1, 1 is for class embedding
        img_feature = self.encoder.forward_features(images)
        # img_embed.shape = torch.Size([BS, 257, 768])
        img_embed = self.projection_layer(img_feature)
        
        # initial input token id
        # input_ids.shape = torch.Size([1, 1])
        input_ids = torch.tensor([[START_TOKEN]], device=img_embed.device) # START_TOKEN: start of generated sequence
        generated_tokens = [] # store generated tokens
        # auto-regressive loop
        for _ in range(max_token):
            outputs = self.decoder(input_ids, img_embed)# Shape: (BS, k, 50257), k is len of input_ids
            next_token_logits = outputs[:, -1, :]  # Shape: (BS, 50257) # Get the last token's logits
            #print(next_token.shape) #torch.Size([1])
            next_token = next_token_logits.argmax(dim=-1)  # Greedy decoding
            generated_tokens.append(next_token.item()) # Append the predicted token

            # Break if END_TOKEN is generated
            if next_token.item() == END_TOKEN:
                break
            
            # Update input_ids to include the new token
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            # print(input_ids.shape): (1, 2)->(1, 3)->(1, 4)...

        return generated_tokens # list on cpu
    
    def autoreg_beam_search(self, images, max_token=20, beam_size=3):
        """Assume BS = 1"""
        START_TOKEN = 50256
        END_TOKEN = START_TOKEN

        # img_feature.shape = torch.Size([BS, 257, 1024]), where 257 = Patch^2 + 1, 1 is for class embedding
        img_feature = self.encoder.forward_features(images)
        # img_embed.shape = torch.Size([BS, 257, 768])
        img_embed = self.projection_layer(img_feature)

        # initial input token id
        # input_ids.shape = torch.Size([1, 1])
        input_ids = torch.tensor([[START_TOKEN]], device=img_embed.device) # START_TOKEN: start of generated sequence
        beams = [(input_ids, 0)]  # List of tuples (sequence, cumulative log-probability)

        completed_sequences = []  # Store completed sequences and their scores

        for _ in range(max_token):
            all_candidates = []
            # Expand each beam
            for seq, score in beams:
                # Generate logits for the current sequence
                outputs = self.decoder(seq, img_embed)  # Shape: (BS, k, vocab_size)
                next_token_logits = outputs[:, -1, :]   # Get logits for the last token
                next_token_probs = torch.log_softmax(next_token_logits, dim=-1)  # Convert logits to log-probabilities

                # Get the top beam_size next tokens
                top_k_probs, top_k_tokens = next_token_probs.topk(beam_size, dim=-1)

                # Create new candidate beams
                for i in range(beam_size):
                    next_token = top_k_tokens[0, i].unsqueeze(0).unsqueeze(0)  # Shape: [1, 1]
                    new_score = score + top_k_probs[0, i].item()
                    new_seq = torch.cat([seq, next_token], dim=-1)

                    # If END_TOKEN is generated, move to completed sequences
                    if next_token.item() == END_TOKEN:
                        completed_sequences.append((new_seq, new_score))
                    else:
                        all_candidates.append((new_seq, new_score))

            # Sort all candidates by score and keep top `beam_size` sequences
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # If all beams are completed, break the loop
            if len(beams) == 0:
                break

        # If no completed sequences, use the best beam as the final sequence
        if len(completed_sequences) == 0:
            completed_sequences = beams

        # Select the best sequence based on its score
        best_seq, _ = sorted(completed_sequences, key=lambda x: x[1], reverse=True)[0]

        # Convert the tensor sequence to a list of tokens and return
        return best_seq[0, 1:].tolist()




if __name__=="__main__":
    vision_encoder = "vit_large_patch14_clip_224"
    text_decoder = "../hw3_data/p2_data/decoder_model.bin"
    input_imgs = torch.rand(1, 3, 224, 224)
    token_id = torch.randint(0, 50257, (7, 23))
    model = ImageCaptionModel(vision_encoder, text_decoder)
    ofmap = model.autoreg_greedy(input_imgs)
    print(ofmap)  # torch.Size([BS, K, 50257])

