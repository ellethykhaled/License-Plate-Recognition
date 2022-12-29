import os
from skimage import io
from preprocessing import Preprocessing
from detect import LicenceDetection
import streamlit as st

directory_path = "./images"
images = os.listdir(directory_path)
availableCars = []
sampleNumber = 13
for i in range(sampleNumber):
    sampleName = "sample" + str(i + 1) + ".jpg"
    availableCars.append(sampleName)

imageName = st.sidebar.selectbox("Select a car image:", availableCars, index=0)

@st.cache
def pipelineExecution():
    preprossedImgage, grayImage = Preprocessing.preprocessPhoto(img)
    lp, lpr, segmented_char, ocr_output = LicenceDetection.detectLicense(preprossedImgage, grayImage)
    return lp, lpr, segmented_char, ocr_output

if imageName:
    img = io.imread(directory_path + "/" + imageName)
    st.subheader("License Plate Recognition")
    st.subheader("Selected Image:")
    st.image(img, channels="RGB")
    st.markdown("---")
    
    # st.sidebar.text("License Plate Recognition Progress:")
    # bar = st.sidebar.progress(0)
    with st.spinner('Preparing plate...'):
        licensePlateImage, lpr, segmentedCharacter, ocrOutput = pipelineExecution()
    
    if licensePlateImage is not None:
        # if st.sidebar.checkbox("Show plate"):
            # bar.progress(25)
        st.subheader("Detected Plate:")
        st.image(licensePlateImage, use_column_width = True, clamp = True)
        st.markdown("---")
            # if st.sidebar.checkbox("Binarize"):
                # bar.progress(50)
        st.subheader("Binarized Plate:")
        st.image(lpr, use_column_width = True, clamp = True)
        st.markdown("---")
        st.subheader("Segmented Characters:")
        st.image(segmentedCharacter, use_column_width = True, clamp = True)
        st.markdown("---")
        st.subheader("OCR output:")
        st.markdown("**" + ocrOutput + "**")
        st.markdown("---")