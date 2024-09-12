import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json
import numpy as np
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description="Predict the type of a given flower out of 102 different species")
    parser.add_argument("image_path", help="Path to the input image", type=str)
    parser.add_argument("saved_keras_model", help="Path to the saved Keras model", type=str)
    parser.add_argument("-k", "--top_k", default=3, help="Top k class probabilities", type=int)
    parser.add_argument("-n", "--category_names", default="./label_map.json", help="Path to a JSON file mapping labels to flower names", type=str)
    return parser.parse_args()

def load_class_names(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)

def process_image(image, image_size=224):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    return (image / 255).numpy()

def predict(image_path, model, top_k):
    image = np.asarray(Image.open(image_path))
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(expanded_image)
    top_k_probs, top_k_classes = tf.nn.top_k(predictions, k=top_k)
    return top_k_probs.numpy()[0], top_k_classes.numpy()[0]

def main():
    args = parse_arguments()
    print(f'Path to the input test image: {args.image_path}')
    print(f'Path to the saved keras model: {args.saved_keras_model}')
    print(f'Top k class probabilities: {args.top_k}')
    print(f'Path to the json file: {args.category_names}')

    class_names = load_class_names(args.category_names)
    model = load_model(args.saved_keras_model)
    print(model.summary())

    top_k_probs, top_k_classes = predict(args.image_path, model, args.top_k)

    print('List of flower labels along with corresponding probabilities:', list(top_k_classes), list(top_k_probs))
    for class_index, prob in zip(top_k_classes, top_k_probs):
        flower_name = class_names.get(str(class_index + 1))
        print(f'Flower Name: {flower_name}')
        print(f'Class Probability: {prob}')

if __name__ == "__main__":
    main()