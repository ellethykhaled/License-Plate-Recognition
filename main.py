from Detection import Detection 
import os
from skimage import io
import streamlit as st

directory_path = "./dataset/images"
images = os.listdir(directory_path)
availableCars = []
sampleNumber = 24
for i in range(1, sampleNumber + 1):
    sampleName = "Cars (" + str(i) + ").png"
    availableCars.append(sampleName)

imageName = st.sidebar.selectbox("Select a car image:", availableCars, index=0)

if imageName:
    img = io.imread(directory_path + "/" + imageName)
    st.header("License Plate Recognition")
    st.subheader("Selected Image:")
    st.image(img, channels="RGB")
    st.markdown("---")
    
    with st.spinner('Preparing plate...'):
        licensePlateImage, segmentedCharacter, ocrOutput = Detection.singleTesting(imageName)
    
    if licensePlateImage is not None:
        st.subheader("Detected Plate:")
        st.image(licensePlateImage, use_column_width = True, clamp = True)
        st.markdown("---")
        if len(segmentedCharacter) > 0:
            st.subheader("Segmented Characters:")
            st.image(segmentedCharacter, use_column_width = False, clamp = True)
            st.markdown("---")

            st.subheader("OCR output:")
            st.markdown("**" + ocrOutput + "**")
            st.markdown("---")