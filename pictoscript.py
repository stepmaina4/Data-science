import tensorflow as tf
from tensorflow.keras.layers import Layer
from PIL import Image
from IPython.display import display
import matplotlib.pyplot as plt

# Custom layer definition
class ExpandDimsLayer(Layer):
    def __init__(self, axis, **kwargs):
        super(ExpandDimsLayer, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.expand_dims(inputs, axis=self.axis)


# Translation dictionary
input_texts = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M","N", "O", "P", "Q", "R", "S", "T", "U", "V","W", "X","Y","Z",
              "a", "b" ,"c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y","z"]
target_images = [
    r"C:\Users\BOBOH\Desktop\symbols\symbol_A.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_B.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_C.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_D.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_E.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_F.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_G.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_H.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_I.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_J.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_K.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_L.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_M.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_N.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_O.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_P.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_Q.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_R.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_S.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_T.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_U.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_V.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_W.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_X.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_Y.png",
    r"C:\Users\BOBOH\Desktop\symbols\symbol_Z.png",

    
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_a.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_b.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_c.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_d.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_e.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_f.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_g.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_h.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_i.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_j.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_k.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_l.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_m.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_n.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_o.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_p.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_q.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_r.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_s.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_t.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_u.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_v.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_w.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_x.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_y.png",
    r"C:\Users\BOBOH\Desktop\symbols2\symbol_z.png"
]  # Paths to images

# Function to translate image path to letter
def translate_image_to_letter(img_path):
    if img_path in image_to_letter:
        letter = image_to_letter[img_path]
        print(f"The letter corresponding to {img_path} is {letter}.")
    else:
        print("Image path not found.")

# Function to translate word to combination of images
def translate_word_to_images(word):
    images = []
    for letter in word:
        if letter in letter_to_image:
            img_path = letter_to_image[letter]
            img = Image.open(img_path)
            images.append(img)
        else:
            print(f"Letter {letter} not found")
    
    # Display images in a horizontal row
    fig, ax = plt.subplots(1, len(images), figsize=(4,1))
    for i, img in enumerate(images):
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()


    #usage
# Translate image path to letter
#translate_image_to_letter(r"C:\Users\BOBOH\Desktop\symbols\symbol_A.png")

# Translate word to combination of images
print("**************************")
translate_word_to_images("KENYA")
translate_word_to_images("kenya")
translate_word_to_images("Ooooooh")
translate_word_to_images("My")
translate_word_to_images("Beautiful")
translate_word_to_images("Country")
print("***************************")


