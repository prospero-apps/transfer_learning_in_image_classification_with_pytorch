# Transfer Learning in Image Classification with PyTorch
In this project, we’ll use transfer learning to train a model to classify images.

This project demonstrates how to use transfer learning with a pre-trained ResNet-50 model to classify satellite images from the EuroSAT dataset.

## Dataset

The [EuroSAT dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.EuroSAT.html#torchvision.datasets.EuroSAT) contains Sentinel-2 satellite images covering 10 different land cover classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

## Transfer Learning

Transfer learning involves using a pre-trained model (trained on a large dataset like ImageNet) and adapting it to a new, but related, task. In this project, we use a pre-trained ResNet-50 model and fine-tune its classifier to classify EuroSAT images.

## Project Structure

1. **Data Preparation:**
   - Download the EuroSAT dataset.
   - Apply necessary transformations (resizing, normalization) to match the pre-trained model's input requirements.
   - Split the data into training and testing sets.
   - Organize the data into a standard image classification format (subfolders for each class).
  
   - ![image](https://github.com/user-attachments/assets/c80f51fb-2eae-4e0a-b4e8-38c1e597888a)


2. **Adjusting the Pretrained Model:**
   - Load the pre-trained ResNet-50 model.
   - Freeze all layers except the classifier to preserve pre-trained knowledge.
   - Replace the classifier with a new one that outputs the correct number of classes (10 for EuroSAT).

3. **Training the Model:**
   - Define a loss function (Cross Entropy Loss) and an optimizer (Adam).
   - Create training and testing DataLoaders.
   - Train the model for a specified number of epochs.

4. **Evaluating the Model:**
   - Plot training and testing loss/accuracy curves to assess model performance and identify potential overfitting.

5. **Making Predictions:**
   - Load and preprocess a sample image.
   - Use the trained model to predict the class of the image.
   - Visualize the prediction and associated probability.

## Results

- The model was able to achieve reasonably high accuracy and very high probability in its predictions.
- Overall, the project demonstrates the effectiveness of transfer learning in satellite image classification tasks.

![image](https://github.com/user-attachments/assets/eb011a07-f2cd-4648-a123-c16231797fc4)

A prediction example:

![image](https://github.com/user-attachments/assets/f526d3c0-1998-4c3b-814a-497b3270657a)


## Usage

1. Clone the repository: `git clone https://github.com/your-username/eurosat-classification.git`
2. Run the Jupyter notebook: `jupyter notebook EuroSAT_Classification.ipynb`

## Future Work

- Explore hyperparameter tuning to further optimize model performance.
- Implement data augmentation techniques to improve model robustness and reduce overfitting.
- Investigate different pre-trained models or architectures for comparison.

## Acknowledgments

- The EuroSAT dataset creators.
- The PyTorch and torchvision libraries.
