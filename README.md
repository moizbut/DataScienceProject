Project Details
A deep learning model to recognize Urdu character sequences from grayscale images using convolutional neural networks (CNN) and TensorFlow.

Install the required libraries. 

There project has two parts, in the first part we perform OCR on individual urdu alphabets and in the second part we do OCR on sequence of disconnected urdu alphabets, maximum upto 4. 

The files for first part are as follows:
1.create_dataset.py is used to create a synthetic dataset
2.train.py is used to train a model on this
3.predict.py is used to predict single alphabet

In the second part
1.seq_dataset.py is used to create the dataset
2.seq_train.py is used to train this dataset
3.seq_predict is used to predict on this model








MIT License

Copyright (c) 2025 [Muhammad Moiz Butt]

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
