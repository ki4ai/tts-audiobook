import tensorflow as tf
from text.symbols import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=50000,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=[''],
        freeze_layers=[''],
        
        ################################
        # Data Parameters             #
        ################################
        training_files='filelists/total_ten_second.txt',
        validation_files='filelists/total2_validation.txt',
        text_cleaners=['korean_english_cleaners'],
        p_arpabet=1.0,
        cmudict_path="data/cmu_dictionary",

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=300,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,
        p_teacher_forcing=1.0,
        p_dropout_teacher=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,
        p_contents_attention=0.0,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,
        
        # prior filter parameters
        prior_n_filter=11,
        prior_alpha=0.1,
        prior_beta=0.9,
        prior_floor=10**-6,
        
        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,
        
        # style embedding
        speakers_name=['ETRI_1', 'ETRI_2', 'ETRI_3', 'KETTS_0', 'KETTS_1', 'KCTTS_0', 'KCTTS_1', 'Drama'],
        speakers_index=[6, 7, 8, 19, 20, 691, 692, 742],
        n_speakers=879, # 705
        speaker_embedding_dim=128,
        emotions_name=["None", "Happy", "Sad", "Angry", "Fear", "Disgust", "Surprise", "Neutral", "Joy"],
        emotions_index=[0, 161, 162, 163, 164, 165, 166, 167, 168],
        n_emotions=241, # 1 (No emotion) + 8 * 30 (mod 30 of speaker): 9~20, 450~451
        emotion_embedding_dim=128,
        freeze_style_encoder=True,
        not_freeze_speaker=[6, 566],
        
        VC_speaker=[5],
        p_VC_ratio=0.5,
        
        # reference embedding (GST, Noise, Token)
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=[3, 3],
        ref_enc_strides=[2, 2],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=128,
        token_embedding_size=256,
        token_num=10,
        num_heads=8,
        p_gst_using=1.0,
        
        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        learning_rate_min=1e-5,
        learning_rate_anneal=50000,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=16, # 56,
        mask_padding=True,  # set model's padded outputs to padded values

    )

    if hparams_string:
        tf.compat.v1.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.compat.v1.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
