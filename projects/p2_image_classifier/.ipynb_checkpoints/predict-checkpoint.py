import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorflow_hub as hub

def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [224, 224])
    image = image / 255.0
    return image.numpy()

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_k_probs, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    top_k_probs = top_k_probs.numpy().flatten()
    top_k_indices = top_k_indices.numpy().flatten()
    return top_k_probs, top_k_indices

def load_class_names(json_file):
    with open(json_file, 'r') as f:
        class_names = json.load(f)
    return class_names

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a pre-trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('model_path', type=str, help='Path to the saved model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    # Load the model
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})

    # Predict the class of the image
    probs, classes = predict(args.image_path, model, args.top_k)

    # Load class names
    class_names = load_class_names(args.category_names)

    # Map the class indices to class names
    class_labels = [class_names[str(index + 1)] for index in classes]  # Adjusting for 1-based index

    # Print the results
    print("Probabilities:", probs)
    print("Classes:", class_labels)

if __name__ == '__main__':
    main()
