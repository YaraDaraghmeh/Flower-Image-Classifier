# Flower Image Classifier

## Project Overview
This project is part of Udacity's Intro to Machine Learning with TensorFlow Nanodegree program. It involves developing a deep learning image classifier using TensorFlow to recognize different species of flowers. The classifier is then converted into a command-line application for easy use.

## Project Structure
- `Project_Image_Classifier_Project.ipynb`: The main Python script containing the image classification code
- `label_map.json`: JSON file mapping class indices to flower names
- `flowerclassifier_model.h5`: Saved Keras model file (not included in the repository due to size)

## Requirements
- Python 3.x
- TensorFlow 2.14
- TensorFlow Hub
- NumPy
- Pillow (PIL)

You can install the required packages using:
```
pip install tensorflow tensorflow-hub numpy pillow
```

## Dataset
The dataset consists of 102 different types of flowers, with approximately 20 images per flower for training. Due to its large size, the dataset is not included in this repository. To access the dataset:

1. Use the GPU-enabled workspaces within the Udacity classroom.
2. Alternatively, download the data from the Udacity workspace if you wish to work locally (Note: Local execution may be challenging without a GPU).

## Usage
To use the flower classifier, run the following command:

```
python flower_classifier.py [image_path] [saved_keras_model_path] [-k top_k] [-n category_names_path]
```

Arguments:
- `image_path`: Path to the flower image you want to classify (required)
- `saved_keras_model_path`: Path to the saved Keras model file (required)
- `-k` or `--top_k`: Number of top predictions to return (optional, default is 3)
- `-n` or `--category_names`: Path to the JSON file mapping labels to flower names (optional, default is "./label_map.json")

Example:
```
python flower_classifier.py ./flower_image.jpg ./flower_model.h5 -k 5 -n ./flower_names.json
```

## GPU Usage
This project requires significant computational resources. It is recommended to use a GPU for training and testing the model. If you're using the Udacity workspace:

1. Enable the GPU when you need it for training or batch predictions.
2. Disable the GPU when not in use to conserve your allocated time.

## Future Improvements
- Implement data augmentation to improve model performance
- Experiment with different pre-trained models as the base
- Create a web interface for the classifier

## Contributing
Contributions to improve the project are welcome. Please feel free to fork the repository and submit pull requests.

## License
[Apache License 2.0]

## Acknowledgments
- Udacity for providing the project framework and dataset
