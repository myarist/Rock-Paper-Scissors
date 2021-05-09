from io import BytesIO
from PIL import Image
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf

def main():
    from tensorflow.keras.applications.inception_v3 import InceptionV3

    pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                    include_top = False, 
                                    weights = 'imagenet')

    for layer in pre_trained_model.layers:
        layer.trainable = False
    
    last_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_layer.output

    from tensorflow.keras.optimizers import RMSprop

    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dropout(0.4)(x) 
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)                  
    x = tf.keras.layers.Dense(3, activation='softmax')(x)           

    model_inception = tf.keras.Model(pre_trained_model.input, x) 

    model_inception.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['accuracy'])
    model_inception.load_weights('model_inception_weights.h5')

    def predict_image(image_upload, model = model_inception):
        im = Image.open(image_upload)
        resized_im = im.resize((150, 150))
        im_array = np.asarray(resized_im)
        im_array = im_array*(1/225)
        im_input = tf.reshape(im_array, shape = [1, 150, 150, 3])

        predict_array = model.predict(im_input)[0]
        mac_proba = predict_array[0]
        ipad_proba = predict_array[1]
        iphone_proba = predict_array[2]

        s = [mac_proba, ipad_proba, iphone_proba]

        import pandas as pd
        df = pd.DataFrame(predict_array)
        df = df.rename({0:'Probability'}, axis = 'columns')
        prod = ['Paper', 'Rock', 'Scissor']
        df['Product'] = prod
        df = df[['Product', 'Probability']]

        predict_label = np.argmax(model.predict(im_input))

        if predict_label == 0:
            predict_product = 'Paper'
        elif predict_label == 1:
            predict_product = 'Rock'
        else:
            predict_product = 'Scissor'

        return predict_product, df, im, s
    
    st.sidebar.header('Please Enter Image Link')
        
    image_url = st.sidebar.text_input("Paste the image file's link here. (JPG or JPEG only)")

    if image_url == "":
        st.markdown("<h1 style='text-align: center;'>Rock ✊🏼 Paper ✋🏼 Scissors ✌🏼</h1>", unsafe_allow_html=True)
        st.markdown("""
                    """)
        st.image('RPS.png', width=700)
        st.markdown("<h3 style='text-align: center;'>Project by <a href='https://www.linkedin.com/in/myarist/' style='text-decoration: none; color:white;'>Muhammad Yusuf Aristyanto</a></h3>", unsafe_allow_html=True)

    else:
        try:
            file = BytesIO(urlopen(image_url).read())
            img = file
            label, df_output, uploaded_image, s = predict_image(img)
            st.sidebar.image(uploaded_image, width = None)

            st.title("The Image is Detected as " + label)
            st.markdown("""
                        """)
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10,6))
            ax = sns.barplot(x = 'Product', y = 'Probability', data = df_output)
            plt.xlabel('')

            for i,p in enumerate(ax.patches):
                height = p.get_height()
                ax.text(p.get_x()+p.get_width()/2.,
                    height + 0.003, str(round(s[i]*100,2))+'%',
                    ha="center") 

            st.pyplot(fig)
        except:
            st.markdown("<h1 style='text-align: center; color:red;'>Oh, No! 😱</h1>", unsafe_allow_html=True)
            st.markdown("""
                        """)
            st.image('RPS.png', width=700)
            st.markdown("<h2 style='text-align: center;'>Please Use Other Link Image 🙏🏻</h2>", unsafe_allow_html=True)
        
if __name__ == '__main__':
    main()