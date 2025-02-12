from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_datasets as tfds

app = Flask(__name__)

# SubwordTextEncoder 로드 (STEP 3에서 생성한 tokenizer)
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file('./vocab_chatbot') # vocab_chatbot 경로 수정 필요

# START_TOKEN_INDEX 및 END_TOKEN_INDEX 정의 (STEP 4, 5에서 사용한 값과 동일하게 설정)
START_TOKEN_INDEX = tokenizer.vocab_size
END_TOKEN_INDEX = tokenizer.vocab_size + 1
MAX_LENGTH = 40 # MAX_LENGTH 값 STEP 3, 4, 5 에서 사용한 값과 동일하게 설정

# 저장된 Transformer 모델 로드 (STEP 6.1 에서 저장한 경로)
MODEL_SAVE_PATH = './chatbot_model' # 모델 저장 경로와 일치해야 함
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# 전처리 함수 (STEP 5 decoder_inference 함수와 동일한 전처리)
def preprocess_sentence(sentence):
    sentence = sentence.strip()
    return sentence

# 디코더 추론 함수 (STEP 5 decoder_inference 함수와 동일)
def decoder_inference(sentence):
    sentence = preprocess_sentence(sentence)
    sentence_encoded = tokenizer.encode(sentence)
    sentence_with_tokens = [START_TOKEN_INDEX] + sentence_encoded + [END_TOKEN_INDEX]
    sentence = tf.expand_dims(sentence_with_tokens, axis=0)
    output_sequence = tf.expand_dims([START_TOKEN_INDEX], 0)

    for _ in range(MAX_LENGTH):
        predictions, _, _ = model(inputs=[sentence, output_sequence], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if tf.equal(predicted_id, END_TOKEN_INDEX):
            break
        output_sequence = tf.concat([output_sequence, predicted_id], axis=-1)
    return tf.squeeze(output_sequence, axis=0)

# 문장 생성 함수 (STEP 5 sentence_generation 함수와 동일)
def sentence_generation(sentence):
    prediction = decoder_inference(sentence)
    predicted_sentence = tokenizer.decode(
        [i for i in prediction if i < tokenizer.vocab_size])
    return predicted_sentence

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_prompt = request.json['prompt'] # 'prompt' 키로 사용자 입력 받기
    if not user_prompt:
        return jsonify({'error': 'Prompt is missing'}), 400

    predicted_sentence = sentence_generation(user_prompt)
    return jsonify({'response': predicted_sentence}) # 'response' 키로 챗봇 응답 반환

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # 외부 접속 허용 및 포트 5000으로 설정