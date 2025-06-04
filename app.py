from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
from flasgger import Swagger, swag_from

# Custom layer used in the model
class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs):
        e = tf.tanh(tf.matmul(inputs, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

# Register the custom layer
tf.keras.utils.register_keras_serializable()(SimpleAttention)

# Define the custom focal loss function
def focal_loss(gamma=2.0, alpha=None):
    if alpha is None:
        alpha = tf.constant([2.5, 2.5, 2.0], dtype=tf.float32)
    def focal_loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        alpha_t = tf.gather(alpha, tf.argmax(y_true, axis=-1))
        alpha_t = tf.expand_dims(alpha_t, -1)
        focal_weight = alpha_t * tf.math.pow(1 - y_pred, gamma)
        loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fn

app = Flask(__name__)
Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "SafeTalks API",
        "description": "API for predicting sentiment in text using a trained LSTM model. Classes: 0 = Hate Speech, 1 = Offensive Chat, 2 = Neither.",
        "version": "1.0.0"
    }
})  # Initialize Swagger UI with custom description

# Load model Keras
model = tf.keras.models.load_model('lstm_model.keras',
                                   custom_objects={"SimpleAttention": SimpleAttention, "focal_loss_fn": focal_loss(gamma=2.0)})

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def home():
    return "SafeTalks Flask API with Keras model and tokenizer ready"

@app.route('/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'description': 'Predict sentiment from texts. Returns probabilities and class labels (0: Hate Speech, 1: Offensive Chat, 2: Neither).',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'texts': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'example': ['I love this!', 'This is bad.']
                    }
                },
                'required': ['texts']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Prediction results with probabilities and class labels',
            'schema': {
                'type': 'object',
                'properties': {
                    'predictions': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'number'}
                        },
                        'description': 'Probabilities for each class [Hate Speech, Offensive Chat, Neither]'
                    },
                    'classes': {
                        'type': 'array',
                        'items': {'type': 'string'},
                        'description': 'Predicted class labels (Hate Speech, Offensive Chat, Neither)'
                    }
                }
            },
            'examples': {
                'application/json': {
                    'predictions': [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]],
                    'classes': ['Neither', 'Offensive Chat']
                }
            }
        },
        '400': {
            'description': 'Error',
            'schema': {
                'type': 'object',
                'properties': {
                    'error': {'type': 'string'}
                }
            }
        }
    }
})
def predict():
    try:
        data = request.json
        texts = data.get('texts', None)

        if not texts:
            return jsonify({"error": "Missing 'texts' key in JSON"}), 400

        # Tokenize and pad texts
        sequences = tokenizer.texts_to_sequences(texts)
        maxlen = 30  # Match max_length from training script
        sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

        # Predict
        preds = model.predict(sequences)
        preds_list = preds.tolist()
        class_names = ['Hate Speech', 'Offensive Chat', 'Neither']
        preds_classes = [class_names[np.argmax(pred)] for pred in preds]

        return jsonify({
            "predictions": preds_list,
            "classes": preds_classes,
            "class_mapping": {
                "0": "Hate Speech",
                "1": "Offensive Chat",
                "2": "Neither"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)