"""
Project Title:
Content-Based Image Retrieval (CBIR) using Support Vector Machine (SVM)

Project Description:
In this project, you'll build a Content-Based Image Retrieval (CBIR) system using a Support Vector Machine (SVM). The goal of CBIR is to retrieve images from a dataset that are visually similar to a query image based on its content, such as color, texture, and shape. Instead of traditional image searching by metadata (like filenames or tags), this approach retrieves images based on visual features.

This project combines OpenAI's image generation capabilities with the use of SVM for classification and retrieval. First, you generate images with OpenAI, extract features from them, and then train an SVM model. Afterward, given a query image, the system will return the most similar images from the dataset based on their features.

Python Code:
"""
import os
import numpy as np
from sklearn import svm
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from apikey import apikey  # I have an apikey.py file that stores the key

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey
OpenAI.api_key = apikey
client = OpenAI()


def generate_images(prompt, num_images=5):
    """
    Generate a list of images based on a given prompt.

    Parameters:
        prompt (str): The base concept for image generation.
        num_images (int): Number of images to generate.

    Returns:
        List of image URLs.
    """
    image_urls = []

    for _ in range(num_images):
        response = client.images.generate(
            prompt=prompt,
            size="1024x1024"
        )
        image_urls.append(response.data[0].url)

    return image_urls


def extract_image_features(image_urls):
    """
    Placeholder function to simulate feature extraction from images.
    In practice, this would involve a pre-trained deep learning model (e.g., VGG16)
    to extract feature vectors from each image.

    Parameters:
        image_urls (list): List of image URLs.

    Returns:
        Feature matrix of extracted features.
    """
    # Simulate image feature extraction with random vectors
    np.random.seed(42)
    features = np.random.rand(len(image_urls), 2048)  # Simulating 2048-dim feature vectors
    return features


def train_svm(features, labels):
    """
    Train an SVM classifier using the image features.

    Parameters:
        features (np.array): Feature matrix.
        labels (list): List of labels corresponding to each feature.

    Returns:
        Trained SVM model.
    """
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(features, labels)
    return model


def retrieve_similar_images(query_features, dataset_features, top_n=3):
    """
    Retrieve the most similar images from the dataset using cosine similarity.

    Parameters:
        query_features (np.array): Feature vector of the query image.
        dataset_features (np.array): Feature matrix of the dataset images.
        top_n (int): Number of similar images to retrieve.

    Returns:
        Indices of the most similar images.
    """
    similarities = cosine_similarity([query_features], dataset_features)[0]
    similar_indices = similarities.argsort()[-top_n:][::-1]
    return similar_indices


# Example Use Cases

# Step 1: Generate a dataset of images
image_prompts = ["mountain landscape", "city skyline", "ocean sunset", "forest path", "desert oasis"]
image_dataset = []
for prompt in image_prompts:
    image_dataset.extend(generate_images(prompt, num_images=3))  # Generate 3 images per prompt

# Step 2: Extract features from the generated images
image_features = extract_image_features(image_dataset)

# Step 3: Train an SVM model with the extracted features and arbitrary labels
labels = [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3  # Assigning labels based on prompts
svm_model = train_svm(image_features, labels)

# Step 4: Given a new query image, retrieve similar images from the dataset
query_image_url = generate_images("snowy mountain", num_images=1)[0]
query_features = extract_image_features([query_image_url])[0]  # Extract features of the query image
similar_images_indices = retrieve_similar_images(query_features, image_features, top_n=1)

print("Query Image URL:", query_image_url)
print("Similar Images URLs:", [image_dataset[i] for i in similar_images_indices])
"""
Example Inputs and Expected Outputs:
Input:

Prompt: "snowy mountain"
Dataset: Images generated from prompts like ["mountain landscape", "city skyline", "ocean sunset", "forest path", "desert oasis"]
Expected Output:

A list of 3 image URLs most similar to the query "snowy mountain."
Input:

Prompt: "beach during sunset"
Dataset: Images generated from prompts ["mountain landscape", "ocean sunset", "forest path", "desert oasis", "city skyline"]
Expected Output:

A list of 3 image URLs most similar to the query "beach during sunset."
Input:

Prompt: "desert with cactus"
Dataset: Images generated from prompts like ["desert oasis", "forest path", "mountain landscape", "city skyline", "ocean sunset"]
Expected Output:

A list of 3 image URLs most similar to the query "desert with cactus."
Input:

Prompt: "city at night"
Dataset: Images generated from prompts ["city skyline", "mountain landscape", "ocean sunset", "forest path", "desert oasis"]
Expected Output:

A list of 3 image URLs most similar to the query "city at night."
Input:

Prompt: "tropical forest"
Dataset: Images generated from prompts ["forest path", "ocean sunset", "mountain landscape", "city skyline", "desert oasis"]
Expected Output:

A list of 3 image URLs most similar to the query "tropical forest."
Project Summary:
This project introduces Content-Based Image Retrieval (CBIR) using a combination of OpenAI's image generation capabilities and SVM for classification. By extracting image features and applying SVM to categorize and retrieve similar images, you'll have a robust tool for finding images based on visual content instead of metadata. This approach can be further refined by integrating deep learning models for more accurate feature extraction.
"""