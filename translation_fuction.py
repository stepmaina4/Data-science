import PIL as pil
from PIL import Image, ImageDraw, ImageFont
translation_dict = {
    'a':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_a.png",
    'b':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_b.png",
    'c':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_c.png",
    'd':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_d.png",
    'e':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_e.png",
    'f':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_f.png",
    'g':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_g.png",
    'h':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_h.png",
    'i':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_i.png",
    'j':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_j.png",
    'k':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_k.png",
    'l':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_l.png",
    'm':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_m.png",
    'n':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_n.png",
    'o':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_o.png",
    'p':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_p.png",
    'q':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_q.png",
    'r':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_r.png",
    's':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_s.png",
    't':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_t.png",
    'u':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_u.png",
    'v':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_v.png",
    'w':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_w.png",
    'x':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_x.png",
    'y':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_y.png",
    'z':
        r"C:\Users\BOBOH\Desktop\symbols\symbol_z.png"
}


def translate(text):
    translated_symbols = []
    for char in text:
        if char.lower() in translation_dict:
            translated_symbols.append(translation_dict[char.lower()])
        else:
            translated_symbols.append(char)
            return translated_symbols

            
