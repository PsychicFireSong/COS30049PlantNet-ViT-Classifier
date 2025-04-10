This README provides steps for setting up and running a Vision Transformer (ViT) model based off the PlantNet300K dataset, using a ViT-b-16 model with pretrained weights.

Prerequisites:

Python: Ensure that Python 3.7+ is installed. 
Anaconda: Anaconda prompt is used to run the Model, tested on conda 4.10.3

You can download the ViT‑B/16 pretrained weights (~660 MB) from the latest release:

https://github.com/PsychicFireSong/COS30049PlantNet-ViT-Classifier/releases/latest

Running the Model:

Place the pretrained weights file 'vit_base_patch16_224_weights_best_acc.tar' into the Utilities directory 
(e.g. InnovPlantModel\Utilities\vit_base_patch16_224_weights_best_acc.tar)

Prepare the Image:
   - Place the image that you want to identify in the images project folder (e.g.: `1.jpg` in the directory InnovPlantModel\images\<image file>).

Running the Model:
   - Open a Anaconda prompt terminal and navigate to the project directory (e.g.: cd C:\Users\master\Downloads\InnovPlantModel ).

   - Create the Python Environment: 
   
   conda env create -f plantnet_300k_env.yml

   - Initiate the Environment: 
   
   conda activate plantnet_300k_env

   - Run the following command to load the pre-trained model and classify the image:

  python main.py 1.jpg 
  (Replace 1.jpg with your own image file, 5 sample files: 1-4.jpg are included for demonstration)


Output:
   - The model will output the classification results for the provided image.

Closing the terminal:

   - Run this command to close out of the Environment before closing the window

conda deactivate

