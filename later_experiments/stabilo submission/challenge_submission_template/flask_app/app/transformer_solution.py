# importing relevant preprocessing functions.
from app.preprocessing import load_data, numpify_samples, remove_hover, \
    normalize, pad_or_resample_samples  # locally defined module, see preprocessing.py

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers.experimental import preprocessing
import random
import logging
import os
import time

#####################################################################################
# This file is a stripped down version of the file that was used to train our model.
# This is done


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
    except RuntimeError as e:
        print(e)

random.seed(218)

logging.getLogger('tensorflow').setLevel(logging.DEBUG)

def log(o):
    logging.getLogger('tensorflow').debug(o)

## text tokenization
def tf_lower_and_split_punct(text): # name of the function is from example on https://www.tensorflow.org/text/tutorials/nmt_with_attention#the_encoderdecoder_model
    # Add spaces around every item in the text.
    text = tf.strings.regex_replace(text, '.', r'\0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def split_train_val(samples, labels):
    indices = list(range(len(samples)))
    random.shuffle(indices)
    sep_index = int((len(indices)/5)*4)
    train_indices, val_indices = indices[:sep_index], indices[sep_index:]
    train_samples, train_labels = [samples[i] for i in train_indices], [labels[i] for i  in train_indices]
    val_samples, val_labels = [samples[i] for i in val_indices], [labels[i] for i in val_indices]
    return train_samples, train_labels, val_samples, val_labels


# custom tokenizing layer
class OutputTextProcessor(preprocessing.TextVectorization):
    def __init__(self):
        super(OutputTextProcessor, self).__init__(standardize=self.tf_lower_and_split_punct)

    def tf_lower_and_split_punct(self, text): # name of the function is from example on https://www.tensorflow.org/text/tutorials/nmt_with_attention#the_encoderdecoder_model
        # Add spaces around every item in the text.
        text = tf.strings.regex_replace(text, '.', r'\0 ')
        # Strip whitespace.
        text = tf.strings.strip(text)

        text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
        return text

# create the layer that will be used for embedding the labels into integer tokens.
output_text_processor = OutputTextProcessor()

# checkpoints are located in flask_app/app/checkpoints/train
# the virtualenv root is in flask_app, so we use the relative path 'app/checkpoints/train' to access the checkpoints.
# if there are no checkpoints, we must first initialize the OutputTextProcessor on the labels in the dataset
if not os.path.isdir('app/checkpoints/train') or len(list(os.scandir('app/checkpoints/train'))) <= 1:
    samples, labels = load_data()
    train_samples, train_labels, val_samples, val_labels = split_train_val(samples, labels)
    train_samples, train_labels, val_samples, val_labels = tf.constant(train_samples, dtype=tf.float32), tf.constant(
        train_labels), tf.constant(val_samples, dtype=tf.float32), tf.constant(val_labels)

    output_text_processor.adapt(train_labels)
    log(output_text_processor.get_vocabulary())


output_text_processor.adapt(['1234567890-·:+='])
checkpoint_path = 'app/checkpoints/train'




EMBEDDING_DIM = 128
UNITS = 1024
HEADS = 8






def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * ( i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def pad_input_sensor_data(seq, d_model):
    seq = tf.pad(seq, tf.constant([[0, 0], [0, 0], [0, d_model - 13]]), 'CONSTANT')
    return seq


def create_input_padding_mask(seq):
    seq = tf.cast(tf.reduce_all(tf.math.equal(seq, tf.zeros((13))), axis=2), dtype=tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]



def create_token_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]



def create_token_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask





############## scaled dot product attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights






################### multi-head attention

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        #         log(v.get_shape())
        #         log(k.get_shape())
        #         log(q.get_shape())
        #         if mask is not None:
        #             log(mask.get_shape())
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights



def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])



class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask
        )
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2



class SensorFeatureEmbedding(layers.Layer):
    def __init__(self, dim_emb, maxlen=2000):
        super().__init__()
        kernel_size = 10
        num_hid = dim_emb
        self.conv1 = tf.keras.layers.Conv1D(num_hid, kernel_size, strides=2, padding="same", activation="relu")
        self.conv2 = tf.keras.layers.Conv1D(num_hid, kernel_size, strides=2, padding="same", activation="relu")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #         x = self.ffn(x)
        return x




class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.maximum_position_encoding = maximum_position_encoding
        self.embedding = SensorFeatureEmbedding(d_model)
        #         self.embedding = point_wise_feed_forward_network(self.d_model, 13)
        self.pos_encoding = positional_encoding(self.maximum_position_encoding,
                                                self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, training, mask):
        # no embedding needed, since sensor data is already a really dense representation
        x = self.embedding(x)
        seq_len = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x





class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.rate = rate
        self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model)
        self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
        self.dec_layers = [DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate)
                           for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x += self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2
        return x, attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, None)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, None
        )
        final_output = self.final_layer(dec_output)
        return final_output, attention_weights


num_layers = 2
d_model = EMBEDDING_DIM
dff = UNITS
num_heads = HEADS
dropout_rate = 0.1


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_epochs=15, steps_per_epoch=236):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_epochs * steps_per_epoch

    def __call__(self, step):
        arg1 = 1 / tf.math.pow(step, 0.5)
        arg2 = step * (self.warmup_steps ** -1.5)
        #         arg2 = tf.constant(1.0)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)


transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    # hardcoded output_text_processor vocab, because the output_text_processor might not be initialized due to checkpointing
    target_vocab_size=len(['', '[UNK]', '[START]', '[END]', '3', '7', '1', '9', '4', '2', '6', '5', '8', '0', '-', '·', ':', '+', '=']),
    pe_input=2000,
    pe_target=100,
    rate=dropout_rate
)


def create_masks(inp, tar):
    enc_padding_mask = create_input_padding_mask(inp)
    dec_padding_mask = create_input_padding_mask(inp)
    look_ahead_mask = create_token_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_token_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


# checkpoint initialization
ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer,
                           output_text_processor=output_text_processor)
# keep track of 5 most recent checkpoints, every 5 checkpoints we make a new checkpoint
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
print(output_text_processor.get_vocabulary())

def tokenize(labels_batch):
    return output_text_processor(labels_batch)

def detokenize(tokens_batch):
    detokenizer = np.array(output_text_processor.get_vocabulary())
    @tf.autograph.experimental.do_not_convert
    def _detokenize(tokens):
        return ''.join(detokenizer[tokens.numpy()])
    return tf.map_fn(_detokenize, tokens_batch, dtype=tf.string)


print(tokenize(['']))

def tokenize_dataset(ds):
    return (ds
           .cache()
           .map(lambda x, y: (x ,tokenize(y))))


def evaluate(sentence, max_length=40):
    # todo: preprocess sentence
    sentence = tf.convert_to_tensor([sentence], dtype=tf.float32)
    encoder_input = sentence
    start, end = tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)
    for i in range(max_length):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output
        )
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output = tf.concat([output, predicted_id], axis=-1)
        if predicted_id == end:
            break
    text = detokenize(output[:, 1:-1])[0]
    return text



def preprocess(samples):
    numpify_samples(samples)
    remove_hover(samples)
    normalize(samples)
    pad_or_resample_samples(samples)
    return samples

def preprocess_and_predict(sentence):
    sentence = preprocess([sentence])[0]
    return evaluate(sentence)