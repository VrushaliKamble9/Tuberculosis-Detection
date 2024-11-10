# Tuberculosis Detection using chest X-ray
The goal is to create accurate, reliable, and scalable diagnostic tools that can effectively distinguish between "Normal" and "Tuberculosis (TB)" cases. By addressing key challenges in the classification of medical images, this study seeks to contribute to the enhancement of diagnostic accuracy.
The specific objectives of this project include:
1.	Development of Deep Learning Models: To design and implement deep learning models specifically tailored for the classification of chest X-ray images.
2.	Performance Evaluation: To rigorously evaluate the models' performance using metrics such as accuracy, robustness, and generalization capability.
3.	Comparative Analysis: To conduct a comparative analysis of the developed models CNN and transfer learning model using Resnet18, identifying their strengths and limitations in the context of chest X-ray classification.
<picture>
 <img alt="Tuberculosis" src="https://github.com/user-attachments/assets/d1d44501-5b63-436b-873e-081eedc3ed95">
</picture>

## 1.1 Data Description
A collaborative effort between researchers from Qatar University, the University of Dhaka, and Malaysian institutions, in conjunction with medical professionals from Hamad Medical Corporation and Bangladesh, has resulted in the development of a chest X-ray image database comprising both tuberculosis (TB) positive and normal cases.
The dataset of chest X-ray images for Tuberculosis (TB) has positive cases images along with Normal images. In the dataset 4200 images are available where normal are 3500 and 700 are Tuberculosis infected chest x-ray.
Here, the data has been segregated in train and test.


## Dataset

| Dataset  | Normal | Tuberculosis | Total  | 
|----------|--------|--------------|--------|
| Train    | 2800   | 560          | 3360   |
| Test     | 700    | 140          | 840    |

### Image data distribution
![image](https://github.com/user-attachments/assets/87047592-7458-4965-a4ef-164e33d783a7)

## Convolution Neural Network Result
### 1	Optimizer performance
To verify the behaviour of each optimizer, the experiment is carried out with 10 epoch with batch size 64 also in this experiment I have given default learning rate of each optimizer.
![image](https://github.com/user-attachments/assets/25da1773-2e2a-4d46-939d-511898afb97e)
Using the Momentum optimizer, the model achieved the highest performance, with a training accuracy of 97.51% and a validation accuracy of 97.32%. The minimal gap between these two accuracies suggests that the model generalizes extremely well while learning effectively from the training data. 
Momentum accelerates convergence by incorporating previous gradient information, which may explain its superior performance in this experiment. This optimizer proved to be the most effective, outperforming the other methods.
SGD also performs well with accuracies of 95.91% (training) and 95.09% (validation), indicating a solid balance between learning and generalization. In contrast, Adam and RMSProp, while preventing overfitting, achieve lower accuracies,
with Adam recording 83.26% training accuracy and 83.63% validation accuracy, and RMSProp recording 83.04% training accuracy and 84.52% validation accuracy. Based on this analysis, Momentum appears as the most effective optimizer for this task.

### 2 Learning rate tuning
This experiment conducted to see which learning rate gives better performance with Momentum optimizer.
![image](https://github.com/user-attachments/assets/61c46be4-fdb1-4773-8e81-dff025c1a32e)
![image](https://github.com/user-attachments/assets/12dddc48-495e-4c17-9c79-5dcdda7fffc7)

The purple line corresponds to a learning rate of 0.001, where the training loss gradually decreases from approximately 0.4 to 0.11 by the end of the training, 
while the validation loss similarly declines to around 0.128. This gradual reduction indicates that the model is converging slowly and may be underfitting.
The green line, representing a learning rate of 0.002, shows faster convergence, though with some instability. At the highest learning rate of 0.02, depicted by the orange line, 
the training loss decreases most rapidly, from about 0.35 to 0.0799. However, despite an initial drop in validation loss to around 0.086, it exhibits significant fluctuations,
suggesting instability and a potential risk of overfitting, as the model struggles to consistently generalize across epochs. The pink line, indicating a learning rate of 0.01,
shows both training and validation losses decreasing rapidly, with training loss falling from 0.35 to around 0.0637 and validation loss reaching a low of 0.0739. 
This behaviour suggests good convergence and generalization, making this learning rate the most effective among those tested.

### 3	Regularization Effects on CNN
#### 3.1	Dropout Effect
The graphs compare the training and validation losses across 10 epochs for the CNN model using various dropout rates (0.1, 0.2, 0.3, and 0.5). For this experiment I kept batch size 64 and learning rate 0.01.

![image](https://github.com/user-attachments/assets/910076ca-7349-4bce-b09e-aa4c80c086f8)
![image](https://github.com/user-attachments/assets/43607fa1-7c48-4c8c-abdb-671dbb397ac9)

Above two graphs give insight of model performance using different dropout rate. Let’s analyse the graph and conclude which dropout rate is suitable for Convolution neural network.
With dropout rate 0.1 we can observe significant decrease in training loss it indicates model learn quickly in other hand validation loss is not consistently the lowest. 
This suggests that the model might be overfitting to the training data, performing well during training but not generalizing as effectively to unseen data. 
Training loss for dropout rate 0.2 indicates more regularization and potentially improving generalization as compare to 0.1. Also, 
the validation loss decreases in a more stable manner compared to 0.1, with fewer fluctuations. This indicates better generalization, as the model is less likely to overfit while still learning effectively.

#### 3.2 L2 Regularization Result
Using similar hyperparameters as those applied for dropout, we employed L2 regularization to assess the model's performance.
![image](https://github.com/user-attachments/assets/223e23d8-78b7-4b08-b7ce-1a59cf2a8aff)
![image](https://github.com/user-attachments/assets/9a0365dd-c434-4d4c-9c87-2c49d1458bfa)

Above graph comparison shows the training loss for weight decay 0.0001 steadily decreases, with minor fluctuations, reaching about 0.1 by the final epoch. 
The validation loss starts high, drops sharply after the first epoch, and converges closely with the training loss by the end. Both losses show a smooth downward trend without significant plateaus. Whereas, 
model performance with weight decay 0.0005 was initially stable but hits a plateau after epoch 5 for both train and validation loss.

## 	Transfer Learning ResNet18 Result
Transfer learning involves taking a pre-trained model, such as those trained on the ImageNet dataset, and fine-tuning it for a different task. The initial layers of the pre-trained model capture general features like edges and textures, which are common across various tasks,
while the later layers are adapted to the specific task at hand.
 
### Momentum performance with 10 epochs
This experiment performs to verify the Momentum performance on chest x-ray dataset specifically for transfer learning using already trained model ResNet18.
This experiment conducted with learning rate 0.01 and batch size 64 for 10 epochs.
![image](https://github.com/user-attachments/assets/f4adcc00-5120-4349-b89c-6ebc9a62be05)

Above graph represented accuracy comparison between the training and validation sets for the ResNet18 model using momentum optimization across 10 epochs.
Initially, at epoch 1, the training accuracy is around 94%, and the validation accuracy near 96%. By epoch 2, the training accuracy increases to by 4.3%,
but the validation accuracy drops sharply to around 93%. However, by epoch 3, the validation accuracy quickly recovers, aligning with the training accuracy at around 98%.
From epoch 4 onwards, both training and validation accuracies stabilize and show minimal fluctuations, keeping close to 99%. By the final epoch (epoch 10), the training accuracy reaches approximately 99.9%, 
while the validation accuracy is around 99.7%. This performance suggests that the ResNet18 model with momentum optimization is highly effective, achieving rapid convergence to high accuracy values.

#### Confusion Matrix Results Interpretation for ResNet18
![image](https://github.com/user-attachments/assets/4dae9c41-dfe2-4fcc-97ff-e15fa343586d)

### L2 Regularization Result
To determine the optimal amount of weight decay required for effective convergence, we conducted experiments using two different weight decay rates: 0.0001 and 0.0005. 
This approach allows us to assess how each rate influences the model's performance and convergence behaviour.
![image](https://github.com/user-attachments/assets/aef9ea05-070f-4d56-bac0-9b5ad3d08c15)
With a weight decay of 0.0001, the training loss initially starts at approximately 0.18 and decreases rapidly during the first few epochs. By the second epoch, it drops to around 0.0545, continuing to 
decline and eventually reached at about 0.0320 by the fifth epoch. This steady and consistent reduction in training loss indicates effective learning and model optimization.
Similarly, the validation loss starts at a level comparable to the initial training loss, around 0.18, and steadily decreases throughout the epochs. By the third epoch, it reaches approximately 0.0545 and 
stabilizes around 0.0389 by the fifth epoch. This smooth and consistent decline in validation loss suggests that the model generalizes well to unseen data, effectively avoiding overfitting.

![image](https://github.com/user-attachments/assets/8293ce7a-25e5-4dec-a5a6-01ded697ae6e)
In contrast, the configuration with a weight decay of 0.0005 exhibits a different pattern. The training loss begins at approximately 0.14 and decreases rapidly, reaching nearly 0.01 by the fifth epoch. 
This swift decline indicates effective learning, with the model achieving lower training loss values compared to the 0.0001 configuration.
However, the validation loss in the 0.0005 configuration starts slightly above the training loss and initially decreases. It reaches its lowest point around the fourth epoch but begins to increase after 
the fourth epoch, signalling potential overfitting.
These graphs indicate the higher weight decay could result in overfitting, causing the model compromises its ability to generalize.

## CNN and RestNet18 test accuracy comparison  
![image](https://github.com/user-attachments/assets/2d9fa7f2-dcf6-483a-b961-1a0df92ff8be)

## How to run Program
There are certain crucial steps that has to be taken while running program.
First download ipynb file from given path and then follow below steps
1.	Dataset preparation
Download the chest x-ray dataset from below URL
https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
After downloading dataset two folders needs to create one for train and other for test under dataset folder on particular drive.
 Inside each of these folders create Normal and TB folder.
After creation of folder keep 2800 Normal label images in Normal folder and 560 Tuberculosis images in TB folder under train folder.
Similarly, add 700 Normal label images in Normal folder and 140 Tuberculosis label images in TB folder under test folder.

2.	After downloading dataset and segregating it we need to mention folder path in ipynb file.
Under Data Preparation heading in ipynb file we have to replace “dataset_dir” path with the folder path created in step 1.
	Below are the path present in the code which we have to replace 	
dataset_dir = 'E:\\ Dataprocessing\\dataset\\'
example: dataset_dir = 'C:\\ dataset'

3.	Similar to data setup, we need one more crucial step for visualization process. Where in one experiment I gave 6 images to model for prediction. For this experiment we need to do one more similar setup mentioned in step 2
For experiment I created one folder “predictimg” and kept random 6 images with their label including Normal and Tuberculosis.
And then in ipynb file in experiment 9 section we have to change the path same as mentioned in step 2.

image_dir = '\\Data processing\\dataset\\predictimg
we need to replace this path for visualization.


4.	Tensorboard Logs
Tensorboard is great tool to view and compare different graph generated by model. To view tensorbaord logs we need to follow some steps.
Before running ipynb file, TensorBoard needs to be installed using the following command.
 pip install tensorbaord
 -	When we run a ipynb file tensorbaord will create folder name logs.
    Also, I have attached log folder generated specific experiment for learning rate when I train and test model
 -  To view these graphs, we need to enter below command in command prompt
     -       
      1.	Open command prompt 
      2.	On command prompt navigate to the folder where logs folder has kept or generated
      3.	Type command tensorbaord --logdir=logs
      4.	This command will provide hyperlink like below http://localhost:6006/
      5.	Click on the link and then we can view the loss and accuracy graphs generated during training model
  
5.	After all setup completed run every cell present in ‘ipynb’ file step by step












