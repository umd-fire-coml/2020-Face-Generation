# 2020-Face-Generation
Developed by Sean Huang, Kyle Hassold, Khushi Bhansali, and Saachi Sahni.
## Product and Model Architecture 
In Face Generation project, we defined and trained a Deep Convolutional Generative Adversarial Network (DCGAN) two part model on a dataset of faces. The main objective of our model was to generate new images of fake human faces that look as realistic as possible.  We used the CelebFaces Attributes Dataset (CelebA) which is a large-scale face attributes dataset with more than 200K celebrity images. The Attribute Data Noise is the input to the Generator and GAN Models. It is a random list of 40 elements (1s and 0s) with a float value between 0-1 at the end. This corresponds to the true/false attribute data that comes with the CelebA dataset with a random float value tacked on for noise. The Generator Model uses the Attribute Data Noise as input into a Deep Convolutional Network that starts with a small image and upscales it through each grouping of layers. In the end it creates a batch of rgb images designed to trick the Discriminator Model into thinking it’s a face. The Discriminator model is trained half with real images from the CelebA dataset and half with fake images generated by Generator Model to output  a single value on a scale of 0-1 for each image that represents whether the Model thinks the image contains a face (a value closer to 0) or no face (a value closer to 1). Finally, the complete GAN model is trained so that the generator can train to defeat the discriminator. Through the Attribute Data Noise and fake truth values, Generator Model tricks the Discriminator Model thus producing a trained GAN model. In summary, the training process comprised of defining the loss functions, selecting optimizer, and finally training the model, to generate fake human faces. 

## Video Presentation
[![YouTube Description](http://img.youtube.com/vi/VH3ndNVXptg/0.jpg)](http://www.youtube.com/watch?v=VH3ndNVXptg "Try here")

## Directory Guide
* check-env.py - Verify the enviornment to run the code, making sure every all the package needed are installed and functional.
* data_validator.ipynb - Notebook visualizing the dataset.
* dataGens.py - Generate fake attributes randomly and image sequence from the data given.
* dataset.py - Loads all data in a directory in a batch.
* dcgan.py - Main training file. Generates and trains our facial recognition software with the built generator and dataset.
* testing.ipynb - Testing notebook for testing and visualizing the results.

## Step-by-step instruction
1. Clone the project locally (In a terminal)
   * `git clone https://github.com/umd-fire-coml/2020-Face-Generation`
2. Enter the 2020-Face-Generation folder
   * `cd 2020-Face-Generation`
3. Create a Conda environment using the environment.yml file
   * `conda env create -f environment.yml`
4. Activate the Conda environment
   * `conda activate face-generation`
5. Run the environment checker in the current directory to check if the environment has required packages installed
   * `python check_env.py`
6. Run the file downloader to download the dataset
   * `check-env.py`
7. Run the data checker script to check if the data is correctly downloaded
   * `python dataset.py`
8. Run the data validation script to check if the data is valid
   * `data_validator.ipynb`
9. Run the training script. This uses the data generator to generate input data, builds the model, and trains it. 
   * `python dcgan.py`

## Testing and Visualizing Results
For testing and visualizing results run data_validator.ipynb 

## Citations
Karras, T, 2018, Progressive Growing of GANS, GitHub repository. https://github.com/tkarrasprogressive_growing_of_gans.

Kehl, C and Linder-Norén, E 2018, Keras-GAN, GitHub repository. https://github.com/eriklindernoren/Keras-GAN
