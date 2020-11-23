
import torch

def collate_fn(batch):    
    txt = [x[0] for x in batch]
    sentence_num = [x[1] for x in batch]
    
    max_input_len = max([len(x[2]) for x in batch])

    text_padded = torch.LongTensor(len(batch), max_input_len)
    text_padded.zero_()
    for i in range(len(batch)):
        text = batch[i][2]
        text_padded[i, :text.size(0)] = text
    
    speaker_ids = torch.LongTensor(len(batch))
    emotion_ids = torch.LongTensor(len(batch))

    for i in range(len(batch)):
        speaker_ids[i] = batch[i][3]
        emotion_ids[i] = batch[i][4]
            
    return (txt, sentence_num, text_padded, speaker_ids, emotion_ids)