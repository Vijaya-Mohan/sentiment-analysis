# Sentiment-analysis
## Overview
This repository is designed to serve as a beginner-friendly guide to natural language processing (NLP)-based emotion classification. The project leverages a Hugging Face dataset initially containing 27 emotions, which has been refined to 14 core emotions to enhance focus and simplify classification. The dataset is preprocessed through word tokenization and vectorization using GloVe 300-dimensional word embeddings. Subsequently, the data is split into an 80% training set and a 20% testing set. Four different neural architectures—Bidirectional Long Short-Term Memory (BiLSTM), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), and Convolutional Neural Network (CNN)—are evaluated for performance, with CNN emerging as the best-performing model.

## Dataset
- **Source**: Hugging Face dataset
- **Original Emotions**: 27
- **Reduced Emotions**: 14 (Selected based on frequency and relevance)
- **Preprocessing Steps**:
  - Text cleaning (removing special characters, converting to lowercase, etc.)
  - Word tokenization using NLP libraries (NLTK)
  - Vectorization using pre-trained **GloVe 300-dimensional embeddings**
  - Splitting data into **80% training** and **20% testing**
  
## Model Architectures
We experimented with the following deep learning architectures for emotion classification:

1. **Bidirectional Long Short-Term Memory (BiLSTM)**
   - Captures long-range dependencies by processing the input in both forward and backward directions.
   - Effective for sequential text data.

2. **Long Short-Term Memory (LSTM)**
   - A widely used recurrent neural network (RNN) model for handling sequential dependencies.
   - Helps mitigate the vanishing gradient problem in standard RNNs.

3. **Gated Recurrent Unit (GRU)**
   - A simpler alternative to LSTM with fewer parameters.
   - Faster training and effective for text-based applications.

4. **Convolutional Neural Network (CNN)**
   - Traditionally used for image processing, but effective in capturing spatial relationships in text embeddings.
   - Performed the best in terms of accuracy and computational efficiency in this project.

### Performance Comparison
- The CNN-based model outperformed the other architectures in terms of **accuracy, efficiency, and computational speed**.
- LSTM and BiLSTM showed promising results but required more training time.
- GRU provided a balanced trade-off between performance and computational efficiency.

## Usage
1. **Preprocess the Dataset**
   - Clean and tokenize the text data.
   - Convert categorical labels into numerical format.
   - Apply GloVe embeddings for vector representation.
   - Split the data into training and testing sets.

2. **Train the Model**
   - Select one of the architectures: BiLSTM, LSTM, GRU, or CNN.
   - Train the model on the preprocessed dataset.
   - Save the trained model for future inference.

3. **Evaluate Model Performance**
   - Compare accuracy, precision, recall, and F1-score across different models.
   - CNN performed the best in this case.

4. **Make Predictions**
   - Use the trained model to classify emotions from new text inputs.
   - Deploy the model for real-world applications such as chatbot integration.

## Results
- **CNN achieved the best accuracy and performance** among all the tested architectures.
- The project provides an accessible entry point for beginners interested in NLP and emotion classification.
- A well-structured dataset and efficient preprocessing techniques significantly contribute to model performance.

## Future Enhancements
- Implementing **transformer-based models** like BERT and GPT for improved results.
- **Fine-tuning hyperparameters** for better accuracy and robustness.
- Expanding the dataset to include **more diverse emotional expressions**.
- Deploying the trained model as an **API or web-based application**.

## Contributions
Contributions are welcome! If you find any issues or have ideas for improvements, feel free to fork the repository and submit a pull request.
