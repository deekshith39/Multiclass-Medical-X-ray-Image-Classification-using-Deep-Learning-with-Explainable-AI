import streamlit as st
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lime

from lime import lime_image
from PIL import Image
from keras.models import load_model
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
from grad_cam_resnet import predict_and_visualize_resnet
from grad_cam_vgg import predict_and_visualize_vgg


def load_image(image_file):
    img = Image.open(image_file)
    return img


def image_resizing(img):
    img_a = np.array(img)
    img_a = img_a/255
    img = resize(img_a, (224, 224), anti_aliasing=True)
    img = img.reshape(1, 224, 224, 3)
    return img


def predict(model, img):
    classes = ["Covid", "Normal", "Viral Pneumonia"]
    
    img = image_resizing(img)
    probs = model.predict(img)
    predictic_class = classes[np.argmax(probs)]
    probalility = max(probs)

    return predictic_class, probalility


def get_explanations(img, model):
    explainer = lime_image.LimeImageExplainer()
    img_l = img.reshape(224, 224, 3)
    explanation = explainer.explain_instance(img_l.astype(
        'double'), model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp_1, mask_1 = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
    temp_2, mask_2 = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

    return temp_1, mask_1, temp_2, mask_2

def setup():
    st.title("Multiclass Medical X-ray Image Classification using Deep Learning with Explainable AI")
    st.sidebar.title("Select a model")
    options_models = ['XCEPTION', 'RESNET', 'VGG']
    selection_model = st.sidebar.radio('', options_models)
    img = st.file_uploader("Please upload the Image", type=["png", "jpg", "jpeg"])

    return img, selection_model

def visualisation_lime_explanations(model, img):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 3, figsize=(20, 20))
    plt.axis('off')
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    img = image_resizing(img)
    temp_1, mask_1, temp_2, mask_2 = get_explanations(img, model)
    ax[1].imshow(mark_boundaries(temp_1 / 2 + 0.5, mask_1))
    ax[1].axis('off')
    ax[2].imshow(mark_boundaries(temp_2 / 2 + 0.5, mask_2))
    ax[2].axis('off')
    st.pyplot(fig)

def xception(img):
    model = load_model('../models/xception_model.h5')
    class_name, prob = predict(model, img)
    st.markdown("#### Predicted class is {}".format(class_name))
    st.write(" ")
    st.markdown("### Explanations : ")
    visualisation_lime_explanations(model, img)

def resnet(img):
    input_img_path = "G:/My Drive/FP/ui/util/input_image.jpg"
    img.save(input_img_path)
    fig, class_name = predict_and_visualize_resnet(input_img_path)
    st.markdown("#### Predicted class is {}".format(class_name))
    st.markdown("### Explanations : ")
    st.pyplot(fig)

def vgg(img):
    input_img_path = "G:/My Drive/FP/ui/util/input_image.jpg"
    img.save(input_img_path)
    fig, class_name = predict_and_visualize_vgg(input_img_path)
    st.markdown("#### Predicted class is {}".format(class_name))
    st.markdown("### Explanations : ")
    st.pyplot(fig)

if __name__ == '__main__':
    img, selection_model = setup()

    if img is not None:
        img = load_image(img)
        img = img.convert('RGB')
        # st.write(img)
        if selection_model == 'XCEPTION':
            xception(img)
        elif selection_model == 'RESNET':
            resnet(img)
        elif selection_model == 'VGG':
            vgg(img)


        


