# ♻️ EcoSort Waste Management Assistant

## AI-Powered Waste Classification and Recycling Guidance

EcoSort is an intelligent waste management assistant designed to help residents correctly identify, classify, and dispose of waste.

The project combines **Computer Vision, Natural Language Processing (NLP), Transformers, and Retrieval-Augmented Generation (RAG)** to create a multimodal waste management system capable of processing both **images and text descriptions**.

The assistant performs three main tasks:

1. 🖼️ **Waste Image Classification** — identifies waste materials from uploaded images using a Convolutional Neural Network (CNN).
2. 📝 **Waste Description Classification** — classifies waste from text descriptions using DistilBERT.
3. ♻️ **Recycling Instruction Generation** — retrieves relevant waste-management information and generates disposal instructions using a RAG pipeline.

---

## 📌 Project Overview

Improper waste sorting can increase the amount of manual sorting required at recycling and waste-management facilities.

EcoSort was developed as an AI-based assistant that helps users determine what type of waste they have and provides guidance on how it should be disposed of.

The system integrates multiple machine-learning approaches into one workflow:

```text
User Input
   │
   ├── Image ──► MobileNetV2 CNN ──► Waste Category
   │
   └── Text ───► DistilBERT ───────► Waste Category
                                      │
                                      ▼
                           Semantic Document Retrieval
                                      │
                              Sentence Transformers
                                      │
                                      ▼
                                 FLAN-T5
                                      │
                                      ▼
                         Recycling / Disposal Guidance
```

---

## 🎯 Project Objectives

The project aims to:

- Explore and preprocess image and text waste datasets.
- Build an image classifier capable of identifying different waste materials.
- Develop a transformer-based model for classifying written waste descriptions.
- Build a semantic retrieval system for waste-management information.
- Generate recycling and disposal instructions using Retrieval-Augmented Generation.
- Integrate the individual models into a single waste management assistant.
- Evaluate the strengths and limitations of the resulting system.

---

## 📊 Datasets

### 1. RealWaste Image Dataset

The computer-vision component uses the **RealWaste dataset**, containing **4,752 waste images** across nine categories.

| Waste Category | Number of Images |
|---|---:|
| Plastic | 921 |
| Metal | 790 |
| Paper | 500 |
| Miscellaneous Trash | 495 |
| Cardboard | 461 |
| Vegetation | 436 |
| Glass | 420 |
| Food Organics | 411 |
| Textile Trash | 318 |

The dataset contains RGB JPEG images with a consistent resolution of **524 × 524 pixels**.

One important characteristic identified during exploratory analysis was **class imbalance**, with Plastic and Metal having considerably more examples than Textile Trash and some other categories.

---

### 2. Waste Descriptions Dataset

The text-classification component uses `waste_descriptions.csv`.

The dataset contains textual descriptions of waste items together with information such as:

- Waste category
- Material composition
- Disposal instructions
- Common waste-classification confusion

The descriptions are used to train the transformer-based text classifier and to construct part of the RAG knowledge base.

---

### 3. Waste Policy Documents

The project also uses `waste_policy_documents.json`.

These documents provide contextual waste-management and recycling information that can be converted into embeddings and retrieved when generating recycling instructions.

---

## 🧹 Data Preprocessing

### Image Processing

Images are resized to:

```text
224 × 224 × 3
```

The image dataset is divided into training, validation, and test sets.

Data augmentation is applied to the training images using:

- Random horizontal flipping
- Random rotation
- Random zoom

These transformations increase variation in the training data and help reduce overfitting.

---

### Text Processing

Waste descriptions are cleaned by:

- Converting text to lowercase
- Removing URLs
- Removing unnecessary special characters
- Removing extra whitespace
- Encoding waste categories into numerical labels

The data is split using an **80/20 train-test split** with stratification to maintain category distributions.

A TensorFlow `TextVectorization` layer is also used during the exploratory text-processing stage.

---

# 🖼️ Part 1: Waste Image Classification

## MobileNetV2 Transfer Learning

The image-classification system uses **MobileNetV2**, pretrained on ImageNet.

Instead of training a CNN entirely from scratch, transfer learning allows the project to take advantage of visual features already learned by MobileNetV2.

The initial architecture consists of:

```text
MobileNetV2
      ↓
Global Average Pooling
      ↓
Dropout
      ↓
Dense Layer (128, ReLU)
      ↓
Dropout
      ↓
9-Class Softmax Output
```

The pretrained MobileNetV2 layers are initially frozen while the custom classification layers learn the nine waste categories.

The model uses:

- **Adam optimizer**
- **Categorical cross-entropy loss**
- **Accuracy** as a primary training metric
- **EarlyStopping**
- **ReduceLROnPlateau**

---

## 📈 CNN Performance

The initial CNN produced a test accuracy of approximately:

**31.7%**

Further fine-tuning improved performance. The notebook's evaluation summary reports a fine-tuned test accuracy of approximately:

**36.46%**

with a test loss of approximately:

**1.784**

Performance varied substantially between waste categories.

For example, Vegetation achieved high recall, while categories such as Glass were considerably more difficult for the model to identify.

The results suggest that the model struggles with visually similar waste materials and the imbalance between waste categories.

---

## 🔧 CNN Fine-Tuning

To improve the CNN:

- Later MobileNetV2 layers were unfrozen.
- Earlier layers remained frozen.
- Batch-normalization layers remained frozen.
- Dropout was increased.
- The learning rate was reduced to `5e-6`.
- Early stopping and learning-rate reduction were used.

Training behaviour indicated that overfitting began to appear during later epochs, reinforcing the importance of early stopping.

Potential future improvements include:

- Class weighting
- Additional data augmentation
- More balanced training data
- Testing alternative architectures such as EfficientNet
- Additional hyperparameter tuning

---

# 📝 Part 2: Waste Description Classification

## DistilBERT Transformer

Text descriptions are classified using:

**`distilbert-base-uncased`**

DistilBERT provides a pretrained language representation that can be fine-tuned for the project's waste-classification task.

Descriptions are tokenized using `DistilBertTokenizerFast` with:

```text
Maximum sequence length: 128
Batch size: 16
Epochs: 3
Learning rate: 2e-5
```

The model's final classification layer is configured according to the number of waste categories.

All pretrained DistilBERT layers remain trainable during fine-tuning.

---

## Text Classification Function

The project implements:

```python
classify_waste_description(description)
```

The function:

1. Cleans the user's description.
2. Tokenizes the text.
3. Passes the tokens through DistilBERT.
4. Selects the highest-probability class.
5. Converts the encoded label back to the corresponding waste category.

Example inputs include:

```text
"empty plastic water bottle"

"old newspaper and cardboard"

"banana peel and leftover food"
```

---

# 🤖 Part 3: Retrieval-Augmented Generation

Classification tells the user **what type of waste an item is**, but users also need to know:

> What should I do with it?

EcoSort therefore implements a **Retrieval-Augmented Generation (RAG)** pipeline.

---

## RAG Knowledge Base

Waste-management documents are constructed using information including:

```text
Waste Category
Material Composition
Disposal Instructions
Common Confusion
```

Duplicate documents are removed before embeddings are generated.

The project creates semantic embeddings using:

**SentenceTransformer — `all-MiniLM-L6-v2`**

The resulting embeddings contain **384 dimensions**.

---

## Semantic Retrieval

When a user asks a recycling question:

1. The question is converted into an embedding.
2. The embedding is compared with stored document embeddings.
3. **Cosine similarity** measures semantic similarity.
4. The most relevant documents are retrieved.
5. The retrieved information is supplied to the generative model.

The default retrieval setting is:

```python
top_k = 3
```

---

## Generative Model

The generation component uses:

**`google/flan-t5-small`**

Generation parameters include:

```text
Maximum new tokens: 120
Beam search: 4 beams
Sampling: Disabled
Repetition penalty: 1.2
No-repeat n-gram size: 3
```

This approach encourages the model to generate instructions based on retrieved waste-management information rather than relying only on its pretrained knowledge.

---

## Example RAG Questions

The system is tested with questions such as:

```text
How should I dispose of a plastic bottle?

What should I do with an old glass bottle?

How should I dispose of used batteries?

Can I recycle a cardboard box?

How should I dispose of food waste?
```

---

# 🔗 Part 4: Integrated Waste Management Assistant

The final stage combines the different models into one assistant.

The main function is:

```python
waste_management_assistant(input_data, input_type="image")
```

The assistant accepts either:

```text
input_type = "image"
```

or

```text
input_type = "text"
```

---

## Image Workflow

```text
Waste Image
    ↓
MobileNetV2
    ↓
Predicted Waste Category
    ↓
RAG Retrieval
    ↓
FLAN-T5
    ↓
Recycling Instructions
```

The image workflow can additionally return the classifier's confidence score.

---

## Text Workflow

```text
Waste Description
       ↓
   DistilBERT
       ↓
Predicted Category
       ↓
Semantic Retrieval
       ↓
    FLAN-T5
       ↓
Recycling Instructions
```

The resulting response contains information such as:

```python
{
    "waste_category": category,
    "confidence": confidence,
    "recycling_instructions": instructions,
    "relevant_documents": relevant_documents
}
```

---

# 🧪 Integrated System Testing

The assistant is tested with text descriptions including:

```text
An empty plastic water bottle
A used cardboard box
An old glass bottle
Leftover food and vegetable peels
A used battery
```

Image tests include examples from categories such as:

- Cardboard
- Glass
- Batteries

These tests evaluate whether classification, retrieval, and generation work together successfully.

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Computer Vision

- TensorFlow
- Keras
- MobileNetV2

### Natural Language Processing

- NLTK
- Hugging Face Transformers
- DistilBERT

### Retrieval-Augmented Generation

- Sentence Transformers
- `all-MiniLM-L6-v2`
- Cosine Similarity
- FLAN-T5

### Deep Learning Frameworks

- TensorFlow
- PyTorch

---

# 📂 Suggested Repository Structure

```text
EcoSort-Waste-Management-Assistant/
│
├── waste_management_summative.ipynb
│
├── waste_descriptions.csv
│
├── waste_policy_documents.json
│
├── RealWaste/
│   ├── Cardboard/
│   ├── Food Organics/
│   ├── Glass/
│   ├── Metal/
│   ├── Miscellaneous Trash/
│   ├── Paper/
│   ├── Plastic/
│   ├── Textile Trash/
│   └── Vegetation/
│
├── README.md

```

> Large datasets may be excluded from the repository and downloaded separately.

---

# 🚀 Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd EcoSort-Waste-Management-Assistant
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk tensorflow torch transformers sentence-transformers pillow tqdm
```

Then open the notebook:

```bash
jupyter notebook waste_management_summative.ipynb
```

Ensure the required datasets are placed in the expected project directories before running the notebook.

---

# ⚠️ Limitations

The project demonstrates an end-to-end multimodal AI architecture, but several limitations remain.

### Image Classification Performance

The CNN currently achieves modest accuracy and performs unevenly across waste categories.

This is likely influenced by:

- Class imbalance
- Similar visual characteristics between waste materials
- Limited examples for some classes
- Variations in object appearance and image backgrounds

### RAG Knowledge Base

The quality of generated recycling guidance depends directly on the quality and coverage of the documents available to the retrieval system.

### Local Recycling Rules

Waste-management policies can differ between cities and regions. A production implementation would therefore require an authoritative and regularly updated local policy knowledge base.

---

# 🔮 Future Improvements

Future development could include:

- Improving CNN performance through class weighting and stronger augmentation.
- Experimenting with EfficientNet or other image-classification architectures.
- Expanding the waste image dataset.
- Performing additional hyperparameter optimization.
- Improving DistilBERT evaluation and error analysis.
- Expanding the RAG knowledge base with authoritative local waste policies.
- Adding retrieval-quality metrics.
- Saving trained models for deployment.
- Building a Streamlit, Flask, or FastAPI interface.
- Deploying the system as a web or mobile application.
- Allowing users to photograph waste directly from a mobile device.
- Returning confidence scores and alternative classifications when the model is uncertain.

---

# 💡 Key Learning

This project demonstrates how multiple AI techniques can be combined to solve a practical problem.

Rather than relying on a single model, EcoSort combines:

**Computer Vision + Transformers + Semantic Search + Generative AI**

to create an end-to-end waste management assistant.

The project also highlights an important machine-learning lesson: building a functional pipeline does not automatically mean that every component is production-ready. Model evaluation revealed important weaknesses in the image classifier, providing clear directions for further experimentation and improvement.

---

## 👩🏽‍💻 Authors

**Belinder Akoth**

**Jacinta Osoro**

**John Mbuthia**

**Monica Nyagaya**


Data Science Project  
Moringa School — Data Science Bootcamp 2026

---

## 📄 Project Status

**Portfolio Project**

The current implementation demonstrates the complete machine-learning pipeline from data exploration and preprocessing through classification, retrieval, generation, and system integration.
