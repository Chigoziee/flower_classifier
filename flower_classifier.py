import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import streamlit as st


def load_checkpoint(filepath, map_location):
    model = models.densenet121() # we do not specify pretrained=True, i.e. do not load default weights
    model.classifier = nn.Sequential(nn.Linear(1024, 500),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(500, 102),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(filepath, map_location))
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img = Image.open(image_path)
    ratio_aspect = max(img.size)/min(img.size)
    
    new_size = [0, 0]
    new_size[0] = 256
    new_size[1] = int(new_size[0] * ratio_aspect)
    
    img = img.resize(new_size)
    
    width, height = new_size
    
    #defining left, top, right, bottom margin
    l_margin = (width - 224)/2
    t_margin = (height - 224)/2
    r_margin = (width + 224 )/2
    b_margin = (height + 224)/2
    # croping
    img = img.crop((l_margin, t_margin, r_margin, b_margin))
    
    # converting to numpy array
    img = np.array(img)

    #Normalizing
    img = img/255    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean)/std
    
    # transpose to get color channel to 1st position
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image, ax=None, title=None):
    """Displays the Image to be classified"""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, class_index, cat_to_name, topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    
    model.cpu()
    model.eval()
    image = process_image(image_path)
    imshow(image)
    image = torch.FloatTensor(image)
    image.unsqueeze_(0) # add a new dimension in pos 0
    
    output = model(image)
    # get the top k classes of prob
    ps = torch.exp(output).data[0]
    topk_prob, topk_idx = ps.topk(topk)

    # bring back to cpu and convert to numpy
    topk_probs = topk_prob.cpu().numpy()
    topk_idxs = topk_idx.cpu().numpy()

    # map topk_idx to classes in model.class_to_idx
    topk_classes = topk_classes = [key for key, value in class_index.items() if topk_idxs[0] == value]
    # map class to class name
    topk_names = [cat_to_name[f'{i}'] for i in topk_classes] 
    
    return topk_names, topk_probs

def main():
    f = open('cat_to_name.json')
    cat_to_name = json.load(f)

    #Loading the model
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    model = load_checkpoint('checkpoint.pth', map_location)
    class_index = torch.load('class_index.pth', map_location)

    # giving the webpage a title
    st.title("Flower Classificaton")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Flower Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    flower_image = st.file_uploader(label = ":sunflower: FLOWER :sunflower:", type = ['png', 'jpg', 'jfif', 'webp'])
    
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Classify"):
        topk_names, topk_probs = predict(flower_image, model, class_index, cat_to_name)
        flower_name = topk_names[0].upper()
        if topk_probs[0] > 0.6:
            st.image(flower_image)
            st.success(f'THIS IS THE {flower_name} FLOWER')
        else:
            st.success("Please upload a clearer image!")
if __name__ == '__main__':
    main()