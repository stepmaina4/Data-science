
translation_dict = {
    'a': r"C:\Users\BOBOH\Desktop\path\to\symbol_a.png",
    'b': r"C:\Users\BOBOH\Desktop\path\to\symbol_b.png",
    'c': r"C:\Users\BOBOH\Desktop\path\to\symbol_c.png",
    'd': r"C:\Users\BOBOH\Desktop\path\to\symbol_d.png",
    'e': r"C:\Users\BOBOH\Desktop\path\to\symbol_e.png",
    'f': r"C:\Users\BOBOH\Desktop\path\to\symbol_f.png",
    'g': r"C:\Users\BOBOH\Desktop\path\to\symbol_g.png",
    'h': r"C:\Users\BOBOH\Desktop\path\to\symbol_h.png",
    'i': r"C:\Users\BOBOH\Desktop\path\to\symbol_i.png",
    'j': r"C:\Users\BOBOH\Desktop\path\to\symbol_j.png",
    'k': r"C:\Users\BOBOH\Desktop\path\to\symbol_k.png",
    'l': r"C:\Users\BOBOH\Desktop\path\to\symbol_l.png",
    'm': r"C:\Users\BOBOH\Desktop\path\to\symbol_m.png",
    'n': r"C:\Users\BOBOH\Desktop\path\to\symbol_n.png",
    'o': r"C:\Users\BOBOH\Desktop\path\to\symbol_o.png",
    'p': r"C:\Users\BOBOH\Desktop\path\to\symbol_p.png",
    'q': r"C:\Users\BOBOH\Desktop\path\to\symbol_q.png",
    'r': r"C:\Users\BOBOH\Desktop\path\to\symbol_r.png",
    's': r"C:\Users\BOBOH\Desktop\path\to\symbol_s.png",
    't': r"C:\Users\BOBOH\Desktop\path\to\symbol_t.png",
    'u': r"C:\Users\BOBOH\Desktop\path\to\symbol_u.png",
    'v': r"C:\Users\BOBOH\Desktop\path\to\symbol_v.png",
    'w': r"C:\Users\BOBOH\Desktop\path\to\symbol_w.png",
    'x': r"C:\Users\BOBOH\Desktop\path\to\symbol_x.png",
    'y': r"C:\Users\BOBOH\Desktop\path\to\symbol_y.png",
    'z': r"C:\Users\BOBOH\Desktop\path\to\symbol_z.png"
}

# Function to translate text to a list of image file paths
def translate(text):
    translated_symbols = []
    for char in text:
        if char.lower() in translation_dict:
            translated_symbols.append(translation_dict[char.lower()])
        else:
            translated_symbols.append(char)
    return translated_symbols
from PIL import Image  # Import the Image class from the PIL library

def generate_image(symbol_paths):
    images = [Image.open(path) for path in symbol_paths if path.lower().endswith('.png')]
    widths, heights = zip(*(image.size for image in images))
    total_width, max_height = sum(widths), max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    
    for image in images:
        new_im.paste(image, (x_offset, 0))
        x_offset += image.size[0]
    
    new_im.save('translated_image.png')
    return 'translated_image.png'
Image.show()
text="a b"
translated_image=translate(text)
print(translated_image)






    


