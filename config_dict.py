emo_dict = {
    'happy': 20*8+1,
    'anger': 20*8+3,
    'disgust': 20*8+5,
    'fear': 20*8+4,
    'neutral': 0,
    'sad': 20*8+2,
    'surprise': 20*8+6,
}

reverse_emo_dict = {
    20*8+1: 'happy',
    20*8+3: 'anger',
    20*8+5: 'disgust',
    20*8+4: 'fear',
    0: 'neutral',
    20*8+2: 'sad',
    20*8+6: 'surprise',
}

spk_dict = {
    'etri_w': 6,
    'etri_m1': 7,
    'etri_m2': 8,
    'ketts_w': 20,
    'ketts_m': 19,
}

for i in range(705):
    spk_dict[str(i)] = i
