# Sentiment-Analysis-of-Restaurant-Reviews-using-NLP
This project applies Natural Language Processing (NLP) and Artificial Neural Networks (ANN) to perform sentiment analysis on restaurant reviews. The system classifies user feedback as positive or negative based on textual content.

By using a Bag of Words (BoW) model to convert text data into numerical features and training an ANN for binary classification, this project demonstrates how AI can understand human sentiments and assist businesses in improving their services through data-driven insights.

🧠 Project Objectives

To preprocess textual restaurant reviews and remove noise such as punctuation and stopwords.

To apply the Bag of Words technique for feature extraction.

To design and train an Artificial Neural Network for sentiment classification.

To evaluate model accuracy and visualize learning progress through accuracy and loss plots.

To enable real-time user input prediction for new reviews.

⚙️ Technologies Used

Python

NLTK (for text preprocessing and stopword removal)

Scikit-learn (for feature extraction and train-test splitting)

TensorFlow / Keras (for ANN model creation and training)

Matplotlib (for visualization)

📂 Dataset

The project uses a dataset named Restaurant_Reviews.tsv, which contains 1,000 customer reviews of restaurants along with a binary label (1 for positive, 0 for negative).
Each review undergoes:

Removal of non-alphabetic characters

Conversion to lowercase

Stopword removal (excluding “not” for better sentiment detection)

Word stemming using PorterStemmer

🧩 Model Architecture

The ANN model consists of:

Input Layer: 1,500 features (from Bag of Words)

Hidden Layers: Two Dense layers with 6 neurons each and ReLU activation

Dropout Layers: 0.1 rate to prevent overfitting

Output Layer: 1 neuron with Sigmoid activation for binary classification

Loss Function: Binary Cross-Entropy
Optimizer: Adam
Metric: Accuracy

📈 Model Training and Evaluation
The model is trained for 25 epochs with a batch size of 10, achieving up to 98% accuracy on test data.
Training accuracy and loss are plotted for visual performance tracking.

🧍‍♀️ User Interaction
After training, users can input a custom restaurant review.
The system preprocesses and predicts sentiment as:

Positive feedback
or
Negative feedback

🚀 Key Takeaways

Demonstrates how ANN models can be combined with NLP for sentiment prediction.

Achieved 98% accuracy, proving efficiency for binary text classification.

Can be extended for multi-class sentiment or aspect-based analysis in the future.
