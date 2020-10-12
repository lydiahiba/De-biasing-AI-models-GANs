import numpy as np
import streamlit as st
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from GAN_architecture import Generator, Classifier
torch.manual_seed(0) # Set for our testing purposes, please do not change!

n_classes = 40
z_dim = 64
batch_size = 128
device = 'cpu'

feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
"BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
"DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Female", # change 'Female' to 'Male' for male model
"MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose", 
"RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair", "WearingEarrings", 
"WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]



def main():
    st.title("Streamlit Face-GAN Demo")

    n_images = 1
    fake_image_history = []
    grad_steps = 10 # Number of gradient steps to take
    skip = 2 # Number of gradient steps to skip in the visualization



    st.sidebar.title('Features')
    seed = 27834096
    # If the user doesn't want to select which features to control, these will be used.
    # default_control_features = ['Young','Smiling','Male']
    # if st.sidebar.checkbox('Show advanced options'):
    #     # Randomly initialize feature values. 
    #     features = get_random_features(feature_names, seed)
    #     # Let the user pick which features to control with sliders.
    #     control_features = st.sidebar.multiselect( 'Control which features?',
    #         sorted(features), default_control_features)
    # else:
    #     features = get_random_features(feature_names, seed)
    #     # Don't let the user pick feature values to control.
    #     control_features = default_control_features


    ### Change me! ###
       # Let the user pick which features to control with sliders.
    # control_features = st.sidebar.multiselect( 'Control which features?',
    #     feature_names)


    features = get_random_features(feature_names, seed)
    control_features= st.text_input('Input your sentence here:') 
    # control_features='Smiling'
    if control_features:
    # Insert user-controlled values from sliders into the feature vector.

        features[control_features] = st.sidebar.slider(control_features, 0, 100, 50, 5)

    target_indices = feature_names.index(str(control_features)) # Feel free to change this value to any string from feature_names!

    gen=load_gen_model()
    classifier=load_classif_model()

    opt = torch.optim.Adam(classifier.parameters(), lr=0.01)

    noise = get_noise(n_images, z_dim).to(device).requires_grad_()
    for i in range(grad_steps):
        opt.zero_grad()
        fake = gen(noise)
        fake_image_history += [fake]
        fake_classes_score = classifier(fake)[:, target_indices].mean()
        fake_classes_score.backward()
        noise.data = calculate_updated_noise(noise, 1 / grad_steps)

    plt.rcParams['figure.figsize'] = [n_images , grad_steps]
    st.image(show_tensor_images(torch.cat(fake_image_history[::skip], dim=2), num_images=n_images, nrow=n_images).numpy(),width=200)
    # should be numpy array 



# Ensure that load_pg_gan_model is called only once, when the app first loads.
@st.cache(allow_output_mutation=True)
def load_gen_model():
    """
    Open the pretrained model.
    """
    gen = Generator(z_dim).to(device)
    gen_dict = torch.load("model/dcgan_generator_25.pth", map_location=torch.device(device))["state_dict"]
    gen.load_state_dict(gen_dict)
    gen.eval()
    return gen 

@st.cache(allow_output_mutation=True)
def load_classif_model():
    n_classes = 40
    classifier = Classifier(n_classes=n_classes).to(device)
    class_dict = torch.load("model/dcgan_classifier_3_male.pth", map_location=torch.device(device))["state_dict"]
    classifier.load_state_dict(class_dict)
    classifier.eval()
    # print("Loaded the models!")

    opt = torch.optim.Adam(classifier.parameters(), lr=0.01)
    return classifier

def calculate_updated_noise(noise, weight):
    '''
    Function to return noise vectors updated with stochastic gradient ascent.
    Parameters:
        noise: the current noise vectors. You have already called the backwards function on the target class
          so you can access the gradient of the output class with respect to the noise by using noise.grad
        weight: the scalar amount by which you should weight the noise gradient
    '''
    #### START CODE HERE ####
    new_noise = noise + (noise.grad * weight)
    #### END CODE HERE ####
    return new_noise

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples in the batch, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)

feature_names = ["5oClockShadow", "ArchedEyebrows", "Attractive", "BagsUnderEyes", "Bald", "Bangs",
"BigLips", "BigNose", "BlackHair", "BlondHair", "Blurry", "BrownHair", "BushyEyebrows", "Chubby",
"DoubleChin", "Eyeglasses", "Goatee", "GrayHair", "HeavyMakeup", "HighCheekbones", "Female", # change 'Female' to 'Male' for male model
"MouthSlightlyOpen", "Mustache", "NarrowEyes", "NoBeard", "OvalFace", "PaleSkin", "PointyNose", 
"RecedingHairline", "RosyCheeks", "Sideburn", "Smiling", "StraightHair", "WavyHair", "WearingEarrings", 
"WearingHat", "WearingLipstick", "WearingNecklace", "WearingNecktie", "Young"]

def get_random_features(feature_names, seed):
    """
    Return a random dictionary from feature names to feature
    values within the range [40,60] (out of [0,100]).
    """
    np.random.seed(seed)
    features = dict((name, 40+np.random.randint(0,21)) for name in feature_names)
    return features

def show_tensor_images(image_tensor, num_images=16, size=(3, 64, 64), nrow=3):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    image = image_grid.permute(1, 2, 0).squeeze()
    return image


if __name__ == "__main__":
    main()