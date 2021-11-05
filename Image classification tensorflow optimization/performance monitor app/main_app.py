import streamlit as st 
import streamlit as st
import tkinter as tk
from tkinter import filedialog
from keras_preds import pred_and_plot
import tensorflow as tf 
import time 

class_names = [
    "rose", "sunflower"
]
def main():
    st.title("Performance Checking App for OpenVino/Tf2 models")
    # Set up tkinter
    root = tk.Tk()
    root.withdraw()
    col1, col2 = st.columns(2)

    col1.header("Chose the path of keras .h5 file")
    # Folder picker button
    col1.title('File Picker')
    keras_weight =  st.text_area("Enter Path", height =100)

    

    col2.header("Chose the folder path of OpenVino model")
    # Make folder picker dialog appear on top of other windows
    #root.wm_attributes('-topmost', 1)

    # Folder picker button
    col2.title('Folder Picker')
    col2.write('Please select a folder:')
    clicked = st.button('Folder Picker')
    if clicked:
        fodler_name = filedialog.askdirectory(master=root)
        dirname = col2.text_input('Selected folder:', fodler_name)
        print(fodler_name)

    st.write("Select Image")
    image_file = st.text_area("Enter Image Path", height =100)


    if st.button('predict'):

        model = tf.keras.models.load_model(keras_weight)
        filename = image_file
        start_time = time.time()
        preds = pred_and_plot(model =model, filename= filename, class_names=class_names)
        end_time = time.time()
        infertime = str(round((end_time - start_time), 2))+" sec"
        st.write(type(image_file))
        prediction_dict = {"prediction_label": preds, "Inference Time": infertime }
        st.write(prediction_dict)



if __name__ == "__main__":
    main()