"""
@author: Ranuja Pinnaduwage

This file is part of pyprofilerai, a Python package that combines profiling 
with AI-based performance optimization suggestions.

Description:
This file provides example usages of the pyproflerai package

Based on Python's cProfile module for performance analysis and utilizes Google Gemini.  
Copyright (C) 2025 Ranuja Pinnaduwage  
Licensed under the MIT License.  

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the "Software"), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:  

The above copyright notice and this permission notice shall be included in all  
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.
"""

import pyprofilerai
import tensorflow as tf
from tensorflow import keras
import numpy as np

def train_and_evaluate():
    """Loads MNIST, trains a simple neural network, and evaluates it."""
    
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the images to [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Define the model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # Flatten input images
        keras.layers.Dense(128, activation='relu'),  # Hidden layer
        keras.layers.Dense(10, activation='softmax') # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, verbose=1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f'\nTest accuracy: {test_acc:.4f}')


def test_function(num, exp):
    for i in range(num):
        i**exp

# Uncomment one of these to use the profiler on one of the functions    
    
pyprofilerai.analyze_performance(test_function,100,2)
# pyprofilerai.analyze_performance(train_and_evaluate)