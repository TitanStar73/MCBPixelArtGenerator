# --- Main Settings ---

tile_dir = "tiles/"
tiles_json = "tiles/tiles.json" #For Wall
gravity_tile_json = "tiles/gtiles.json" #For Gravity thing

frame_material = "black_concrete" #No border -> air | black border -> black_concrete


import shutil


try:
    shutil.rmtree("tempMinecraftBedrockPixelArtGenerator")
except:
    pass

shutil.copytree("MinecraftBedrockPixelArtGenerator", "tempMinecraftBedrockPixelArtGenerator", dirs_exist_ok=True)

#Attempts to locate BP/RP (if installed)
main_js_loc = f"tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorBP/scripts/main.js"
main_rp_loc = f"tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorRP/textures/blocks/"


# --- 

#3D profile, determines minimum opacity to be allowed in this layer. A 0.0 layer is always added to the front.
#E.g. if you set DL = [0.2, 0.5, 0.99] -> Foreground image taken, shrunk to form IMG -> You will have 4-layers, 
# 1) Everything visible 
# 2) All pixels in IMG which ahve opacity >= 0.2 
# 3) All pixels in IMG which ahve opacity >= 0.5 
# 4) All pixels in IMG which ahve opacity >= 0.99

#These are available to the user. Each one must be in all uppercase
DL_PROFILES = {
    "3D": [0.2, 0.5, 0.8, 0.99],
    "4D": [0.2, .35, 0.5, .75, 0.8, 0.99],
    "HIGHLIGHT": [0.6],
    "MINIMUM": [0.99],
    "MODEST": [0.25, 0.75, 0.99],
}
DL_DEFAULT = "3D"

DEPTH_LAYERS = DL_PROFILES["3D"] #Gives a good "3D look"

#For falling block animations
ANIMATIONS = [] #-> anim(pixels) -> yield [(i1,j1),(i2,j2)] one by one, as they should fall -> refers to pixel[i][j], so i = z-axis and j = x-axis
SWAP_X_Z = True #-> Change plane direction


import sys
import cv2
import colorsys
import os
import pyperclip
from PIL import Image, ImageDraw
import json
import tqdm
import numpy as np
import math
import shutil
import rembg
from numba import jit
import ctypes
from tkinter import filedialog

try:
    img_path = sys.argv[1]
except IndexError:
    """Makes the Tkinter application aware of the system's DPI scaling."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    img_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")])


##Temporary code, this ensures its a square, need to refactor later to allow non-square shapes (technically any arbtiary shape should work, currently works w/ barrier blocks)
FORCE_SQUARE = True
if FORCE_SQUARE:
    img = Image.open(img_path)
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(tile_dir + "temp_cropped.png")
    img_path = tile_dir + "temp_cropped.png"


SIZE = input("Enter the block size the side length should be (defaults to 124 if left blank): ")
if SIZE == "":
    SIZE = 124
else:
    SIZE = int(SIZE)

print("""
Which Image type would you like:
1) Large Wall (More detailed)
2) Gravity Block
3) Large Wall BUT ON THE GROUND
""")

IMAGE_TYPE = 0
while True:
    try:
        IMAGE_TYPE = int(input("Enter your image type (id no): "))
        break
    except:
        pass

print("\nTip: If you don't know what to write for any option, you can just click enter! (The default options should work for most people)\n")

ON_GROUND = False
if IMAGE_TYPE == 3:
    IMAGE_TYPE = 1
    ON_GROUND = True

CREATE_DEPTH = False
if IMAGE_TYPE in {1}:
    Z_pos = input("Enter the Z-pos (Defaults to 0): ")
    if Z_pos == "":
        Z_pos = 0
    else:
        Z_pos = int(Z_pos)

    if ON_GROUND:
        Z_pos -= 60

    CREATE_DEPTH = True if input("Create depth/parallax-effect? (Enter 'y' for yes): ").lower() in {"yes", "y"} else False


NORMALIZE = False

if CREATE_DEPTH:
    print("Which Depth-Profile would you like?: ")
    for key in DL_PROFILES.keys():
        print(f"{key} -> {DL_PROFILES[key]}")

    key = input(f"Enter the key (Defaults to {DL_DEFAULT}): ").upper()
    while key not in DL_PROFILES.keys():
        if key == "":
            key = DL_DEFAULT
            break
        key = input(f"That was not an option. Enter the key (Defaults to {DL_DEFAULT}): ").upper()
    

    DEPTH_LAYERS = DL_PROFILES[key]

    print("Creating depth!!")

DEBUG = False #Enables debug mode

if CREATE_DEPTH and not DEBUG:
    input_path = str(img_path)
    output_path = tile_dir + "temp123.png"

    with open(input_path, "rb") as input_file:
        input_data = input_file.read()


    session = rembg.new_session("isnet-general-use")
    output_data = rembg.remove(
        input_data,
        session=session,
        #post_process_mask=True     
    )

    with open(output_path, "wb") as output_file:
        output_file.write(output_data)
    
    cv2.imshow('Image', cv2.imread(output_path))
    cv2.waitKey(0)

    if input("Is this unsatisfactory? (Enter 'yes' if yes, 'no' for no): ").lower() in {'y', 'yes'}:
        print("Ok, trying a normalization aglorithm")
        NORMALIZE = True


if CREATE_DEPTH and NORMALIZE and not DEBUG:
    ICON_FILENAME = img_path

    def get_color_diff(color1, color2):
        return (color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2 + (color1[3] - color2[3])**2

    def generate_coords():
        start = 2
        while True:
            for y in range(0,start+1):
                yield (start-y,y)
            start += 1

    with Image.open(ICON_FILENAME) as img_raw:
        tolerance = ((255 * (5 / 100))**2) * 4 #5 here is tolerance level in %
        width, height = img_raw.size
        img = img_raw.convert("RGBA")
        pixel_data = img.load()
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        draw.point((0, 0), 255)
        draw.point((0, 1), 255)
        draw.point((1, 0), 255)
        
        for x,y in generate_coords():
            if x >= width and y >= height:
                break
            for xt, yt in [(x-1,y-1), (x+1,y), (x,y+1), (x+1,y+1), (x-1,y+1), (x+1,y-1), (x-1,y), (x,y-1)]:
                try:
                    if (mask.getpixel((xt,yt)) == 255) and get_color_diff(img.getpixel((0,0)), img.getpixel((x,y))) <= tolerance:
                        draw.point((x,y), 255)
                        break
                except:
                    pass

        for xj,yj in generate_coords():
            x = width - xj - 1
            y = height - yj - 1
            if x < 0 and y < 0:
                break
            for xt, yt in [(x-1,y-1), (x+1,y), (x,y+1), (x+1,y+1), (x-1,y+1), (x+1,y-1), (x-1,y), (x,y-1)]:
                try:
                    if (mask.getpixel((xt,yt)) == 255) and get_color_diff(img.getpixel((0,0)), img.getpixel((x,y))) <= tolerance:
                        draw.point((x,y), 255)
                        break
                except:
                    pass

        for x in range(width):
            for y in range(height):
                if mask.getpixel((x, y)) == 255:
                    pixel_data[x, y] = (255, 255, 255, 0)

        img.save(tile_dir + "temp123.png")
        
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", cv_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if input("Is this unsatisfactory? (Enter 'yes' if yes): ").lower() in {'y', 'yes'}:
            print("Ok, trying a normalization aglorithm")
            NORMALIZE = True
        else:
            NORMALIZE = False

if CREATE_DEPTH and NORMALIZE and not DEBUG:
    file_path = img_path

    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    #foreground = image_rgb * mask2[:, :, np.newaxis]
    foreground_with_alpha = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    foreground_with_alpha[:, :, :3] = image_rgb * mask2[:, :, np.newaxis]
    foreground_with_alpha[:, :, 3] = mask2 * 255
    foreground = foreground_with_alpha

    background = image_rgb * (1 - mask2[:, :, np.newaxis])

    mask_inpaint = np.where((mask2 == 0), 0, 1).astype('uint8')
    inpainted_image = cv2.inpaint(background, mask_inpaint, 3, cv2.INPAINT_TELEA)
    alpha_channel = np.ones((inpainted_image.shape[0], inpainted_image.shape[1]), dtype=np.uint8) * 255

    background = cv2.merge((inpainted_image, alpha_channel))

    cv2.imwrite(tile_dir + "temp123.png", cv2.cvtColor(foreground, cv2.COLOR_RGB2BGRA))

    cv_img = cv2.cvtColor(foreground, cv2.COLOR_RGB2BGRA)
    cv2.imshow("Image", cv_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    NORMALIZE = False
    if input("Would you like to cancel depth effect (enter 'y' for yes, 'n' for no)?: ").lower() in {'y', 'yes'}:
        CREATE_DEPTH = False


def make_mosaic(grid, tile_raw, tile_folder=tile_dir[:-1]):
    rows, cols = len(grid), len(grid[0])

    tile_cache = {}
    for key, fname in tile_raw.items():
        img = Image.open(f"{tile_folder}/{fname}").convert("RGBA")
        tile_cache[key] = img

    tile_w, tile_h = next(iter(tile_cache.values())).size

    mosaic = Image.new("RGBA", (cols * tile_w, rows * tile_h))

    print("Loading preview...")
    for r in range(rows):
        for c in range(cols):
            mosaic.paste(tile_cache[grid[r][c]], (c * tile_w, r * tile_h))

    # final resize
    true_mosaic = mosaic.resize((1920,1920), Image.LANCZOS).copy() #High-def
    mosaic = mosaic.resize((800, 800), Image.LANCZOS)

    cv_img = cv2.cvtColor(np.array(mosaic.convert("RGB")), cv2.COLOR_RGB2BGR)
    cv2.imshow("Preview", cv_img)
    cv2.moveWindow("Preview", 100, 100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cv2.cvtColor(np.array(true_mosaic.convert("RGB")), cv2.COLOR_RGB2BGR)

image = Image.open(img_path).convert("RGBA")
PERFECT_MATCH = False
if IMAGE_TYPE == 2:
    MAX_COLOURS = 1
    print("Since there are only a few gravity blocks in the game, we need to 'pixelate' your image")
    print("We'll now iteratively add in more detail as needed. It will start by only having 2 colours. Then iteratively adding in more.")
    print("The lower the number of colours, the better the end result (and you can create a perfect recreation if number of colours < 20).")
    while True:
        MAX_COLOURS += 1
        image2 = image.resize((SIZE, SIZE), resample=Image.NEAREST).quantize(colors=MAX_COLOURS, method=2).convert("RGBA")


        cv_img = cv2.cvtColor(np.array(image2.convert("RGB").resize((800,800),resample=Image.NEAREST)), cv2.COLOR_RGB2BGR)
        cv2.imshow("Preview", cv_img)
        cv2.moveWindow("Preview", 100, 100)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        k = input(f"Is this fine {MAX_COLOURS}? (default = no, p = go back 1, y = yes this is fine): ").lower()

        if k in {'y', 'yes'}:
            break
        if k == 'p':
            MAX_COLOURS -= 2
        else:
            try:
                MAX_COLOURS = int(k) - 1
            except:
                pass


    image = image.convert("RGBA").resize((SIZE, SIZE), resample=Image.NEAREST).quantize(colors=MAX_COLOURS, method=2).convert("RGBA").transpose(Image.FLIP_LEFT_RIGHT)

    USED_COLOURS = 0

    palette = image.getcolors(maxcolors=MAX_COLOURS)  # [(count, (R,G,B,A)), ...] format
    for idx, (_, color) in enumerate(palette, start=1):
        color_img = Image.new("RGBA", (16,16), color)
        color_img.save(f"{tile_dir}generated/{idx}.png")
        USED_COLOURS += 1


    if USED_COLOURS < 20:
        PERFECT_MATCH = input("Would you like to create a perfect match (It will create a resource pack for your world)? Enter 'y' if yes : ").lower() in {'y', 'yes'}
    else:
        PERFECT_MATCH = False
        print("Colours used >= 20, so no perfect match RP possible...")

else:
    image = image.convert("RGBA").resize((SIZE, SIZE), resample=Image.LANCZOS)

image.save(tile_dir + "temp1.png")
width, height = image.size

if CREATE_DEPTH:
    foreground = Image.open(tile_dir + "temp123.png")
    foreground = foreground.convert("RGBA").resize((SIZE, SIZE), resample=Image.LANCZOS)
    foreground.save(tile_dir + "temp12.png")

def shrink_gun(filename):
    image = Image.open(filename).convert("RGBA")
    return image.resize((1, 1), resample=Image.LANCZOS)

print(shrink_gun(f"{tile_dir}red_sand.png").getpixel((0,0)))

if IMAGE_TYPE == 2 and not PERFECT_MATCH:
    a = 0.06
    b = 2

    def hue(p1):
        r, g, b, a = p1
        r, g, b = r/255, g/255, b/255

        h, l, s = colorsys.rgb_to_hls(r, g, b)

        return h,l,s

    def my_func(x):
        # x: (0 to 100) -> y: (0 to 1)
        return (1-(math.e**(-a*x)))/(1-(math.e**(-100*a*x)))

    def distance(c1, c2):
        h1, l1, s1 = hue(c1)
        h2, l2, s2 = hue(c2)

        scale_factor = my_func((s1 + s2) * 50)

        dh = min(abs(h1-h2), 1 - abs(h1-h2))
        dl = abs(l1-l2)
        ds = abs(s1-s2) * b

        return (scale_factor * dh) + ((1-scale_factor) * (dl + ds))

else:
    def distance(p1, p2): # -> p1: (r,g,b,a) tuples
        return ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2) + (4*((p1[3] - p2[3]) ** 2)) #Heavily weighted transparency



    @jit(nopython=True)
    def rgb2lab(r, g, b):
        r, g, b = [x/255 for x in (r,g,b)]
        
        def f(c): 
            return c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4
        
        r,g,b = f(r), f(g), f(b)
        
        X = r*0.4124 + g*0.3576 + b*0.1805
        Y = r*0.2126 + g*0.7152 + b*0.0722
        Z = r*0.0193 + g*0.1192 + b*0.9505
        
        X, Y, Z = X/0.95047, Y/1.00000, Z/1.08883
        
        def gfunc(t): 
            return t**(1/3) if t>0.008856 else 7.787*t+16/116
        
        fx, fy, fz = gfunc(X), gfunc(Y), gfunc(Z)
        
        L = 116*fy - 16
        a = 500*(fx - fy)
        b = 200*(fy - fz)
        
        return (L,a,b)

    @jit(nopython=True)
    def deltaE2000(c1, c2):
        L1,a1,b1 = rgb2lab(*c1)
        L2,a2,b2 = rgb2lab(*c2)

        avg_L = (L1+L2)/2
        C1, C2 = math.hypot(a1,b1), math.hypot(a2,b2)
        avg_C = (C1+C2)/2

        G = 0.5*(1-math.sqrt((avg_C**7)/(avg_C**7+25**7)))
        a1p, a2p = (1+G)*a1, (1+G)*a2
        C1p, C2p = math.hypot(a1p,b1), math.hypot(a2p,b2)
        avg_Cp = (C1p+C2p)/2

        h1p = math.degrees(math.atan2(b1,a1p))%360
        h2p = math.degrees(math.atan2(b2,a2p))%360
        deltahp = h2p-h1p
        if abs(deltahp)>180:
            deltahp -= math.copysign(360,deltahp)
        avg_hp = (h1p+h2p+360)/2 if abs(h1p-h2p)>180 else (h1p+h2p)/2

        dLp = L2-L1
        dCp = C2p-C1p
        dhp = 2*math.sqrt(C1p*C2p)*math.sin(math.radians(deltahp)/2)

        T = 1 - 0.17*math.cos(math.radians(avg_hp-30)) 
        T += 0.24*math.cos(math.radians(2*avg_hp)) 
        T += 0.32*math.cos(math.radians(3*avg_hp+6)) 
        T -= 0.20*math.cos(math.radians(4*avg_hp-63))

        SL = 1 + (0.015*(avg_L-50)**2)/math.sqrt(20+(avg_L-50)**2)
        SC = 1 + 0.045*avg_Cp
        SH = 1 + 0.015*avg_Cp*T

        dRo = 30*math.exp(-((avg_hp-275)/25)**2)
        RC = 2*math.sqrt((avg_Cp**7)/(avg_Cp**7+25**7))
        RT = -RC*math.sin(math.radians(2*dRo))

        return math.sqrt((dLp/SL)**2 + (dCp/SC)**2 + (dhp/SH)**2 + RT*(dCp/SC)*(dhp/SH))

    def distance_CIELAB(p1,p2):
        return ((p1[3] - p2[3])**3 + 1) * deltaE2000(p1[:3], p2[:3])

    distance = distance_CIELAB #Optimizing. Remove if you don't have numba



def WaveAnimation(pixels):
    for total in range(len(pixels) + len(pixels[0]) - 1):
        out = []
        for i in range(len(pixels)):
            for j in range(len(pixels[0])):
                if i + j == total:
                    out.append([i,j])

        yield out

def Swoosh(pixels):
    border_size = 1 #Not changeable, check Adaptive Swoosh

    yield [[0,0]]
    for i in range(1,len(pixels)):
        yield [[i,0],[0,i]] #Border till the end
    for i in range(1,len(pixels) - 1):
        yield [[len(pixels)-1, i], [i, len(pixels)-1]]
    
    released = False
    for out in WaveAnimation([x[border_size:-border_size] for x in pixels[border_size:-border_size]]):
        if not released:
            yield [[len(pixels)-1, len(pixels)-1], [len(pixels)-2, len(pixels)-2]]
            released = True
        else:
            yield [[len(pixels) - 2 - i, len(pixels) - 2 - j] for i,j in out]

def AdapdtiveSwoosh(pixels):
    border_size = len(pixels)//14

    def in_border(i,j):
        return i < border_size or j < border_size or i >= len(pixels) - border_size or j >= len(pixels) - border_size

    def can_depend_on(i1,j1,i2,j2):
        return (i1==i2 and abs(j2-j1) == 1) or (j1==j2 and abs(i2-i1) == 1)

    out = [[[0,0]]]

    waved = [[[len(pixels) - border_size - 1 - i, len(pixels) - border_size - 1 - j] for i,j in coords] for coords in WaveAnimation([x[border_size:-border_size] for x in pixels[border_size:-border_size]])]
    activator, wave_animation = waved[0][0], waved[1:]

    activated = False

    used = {(0,0), (activator[0], activator[1])}

    while True:
        out.append([])
        for i in range(len(pixels)):
            for j in range(len(pixels)):
                if (i,j) in used:
                    continue
                if in_border(i,j):
                    for i1,j1 in out[-2]:
                        if can_depend_on(i1,j1,i,j):
                            out[-1].append([i,j])
                            used.add((i,j))
                            break

                elif activated:
                    for i1,j1 in out[-2]:
                        if can_depend_on(i1,j1,i,j):
                            out[-1].append([i,j])
                            used.add((i,j))
                            break

        if not activated:
            for i,j in out[-2]:
                if can_depend_on(activator[0], activator[1], i, j):
                    out[-1].append([activator[0], activator[1]])
                    activated = True

        if out[-1] == []:
            break

    for item in out[:-1]:
        yield item


ANIMATIONS.append(WaveAnimation)
ANIMATIONS.append(Swoosh)
ANIMATIONS.append(AdapdtiveSwoosh)


def complementary_color_hsl(pixel):
    r, g, b, a = pixel
    r, g, b = r/255, g/255, b/255
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    h = (h + 0.5) % 1.0

    r, g, b = colorsys.hls_to_rgb(h, l, s)

    return [int(r*255), int(g*255), int(b*255), a]




new_img = [] #Each element -> [x,y,z,"name"]

if IMAGE_TYPE == 1:
    dropper = False #Can be enabled to create a "dropper"
    
    p2 = []

    with open(tiles_json) as f:
        tile_raw = json.load(f)

    tiles_blended_image = {key:shrink_gun(tile_dir + tile_raw[key]) for key in tile_raw.keys()}
    tiles_blended_pixel = {key:tiles_blended_image[key].getpixel((0,0)) for key in tiles_blended_image.keys()}


    for x in range(-1, width + 1):
        new_img.append([x,height+1,Z_pos,frame_material])
        if CREATE_DEPTH:
            for DEPTH_SIZE in range(1, len(DEPTH_LAYERS) + 1):
                new_img.append([x,height+1,Z_pos + DEPTH_SIZE,"barrier"])
                new_img.append([x,height+1,Z_pos - DEPTH_SIZE,"barrier"])
            
        
    for x in range(-1, width + 1):
        new_img.append([x,0,Z_pos,frame_material])
        if CREATE_DEPTH:
            for DEPTH_SIZE in range(1, len(DEPTH_LAYERS) + 1):
                new_img.append([x,0,Z_pos + DEPTH_SIZE,"barrier"])
                new_img.append([x,0,Z_pos - DEPTH_SIZE,"barrier"])
        

    for y in tqdm.trange(height):
        new_img.append([-1,height-y,Z_pos,frame_material])
        p2.append([])
        if CREATE_DEPTH:
            for DEPTH_SIZE in range(1, len(DEPTH_LAYERS) + 1):
                new_img.append([-1,height-y,Z_pos + DEPTH_SIZE,"barrier"])
                new_img.append([-1,height-y,Z_pos - DEPTH_SIZE,"barrier"])
            
        for x in range(width):
            p1 = image.getpixel((x, y))  # Get RGB values at (x, y)
            score = float("inf")
            prev = "dirt"

            for key in tiles_blended_pixel.keys():
                my_score = distance(p1, tiles_blended_pixel[key])
                if my_score < score:
                    score = float(my_score)
                    prev = key

            new_img.append([x,height-y,Z_pos,prev])
            p2[-1].append(prev)

            if CREATE_DEPTH:
                p1 = foreground.getpixel((x, y))  # Get RGBA values at (x, y)
                
                for DEPTH_SIZE_RAW, DEPTH_THRESHOLD in enumerate(DEPTH_LAYERS):
                    DEPTH_SIZE = DEPTH_SIZE_RAW + 1
                    if p1[3] < 255 * DEPTH_THRESHOLD:
                        new_img.append([x,height-y,Z_pos + DEPTH_SIZE,"barrier"])
                        new_img.append([x,height-y,Z_pos - DEPTH_SIZE,"barrier"])
                    else:                
                        new_img.append([x,height-y,Z_pos + DEPTH_SIZE,prev])
                        new_img.append([x,height-y,Z_pos - DEPTH_SIZE,prev])
                            
                

                

        new_img.append([width,height-y,Z_pos,frame_material])
        if CREATE_DEPTH:
            for DEPTH_SIZE in range(1, len(DEPTH_LAYERS) + 1):
                new_img.append([width,height-y,Z_pos + DEPTH_SIZE,"barrier"])
                new_img.append([width,height-y,Z_pos - DEPTH_SIZE,"barrier"])

    
    def offset_pos(pos_coords, offset):
        pos_coords[0] = pos_coords[0] - offset
        return pos_coords

    new_img = [offset_pos(pos, width//2) for pos in new_img]


   
if IMAGE_TYPE == 2:
    frame_material = "black_concrete_powder"

    if PERFECT_MATCH:
        with open(gravity_tile_json) as f:
            tile_raw2 = json.load(f)

        potentials = [x for x in list(tile_raw2.keys()) if x != frame_material]
        potentials = [frame_material] + potentials

        tile_raw = {}
        for idx in range(1, USED_COLOURS + 1):
            tile_raw[potentials[idx - 1]] = f"generated/{idx}.png" 


        for idx in range(1, USED_COLOURS + 1):
            shutil.copy(f"{tile_dir}generated/{idx}.png", f"{main_rp_loc}{tile_raw2[potentials[idx-1]]}")  

    else:
        with open(gravity_tile_json) as f:
            tile_raw = json.load(f)


    tiles_blended_image = {key:shrink_gun(tile_dir + tile_raw[key]) for key in tile_raw.keys()}
    tiles_blended_pixel = {key:tiles_blended_image[key].getpixel((0,0)) for key in tiles_blended_image.keys()}
    
    print("Chose you animation function: ")
    for i,anim in enumerate(ANIMATIONS):
        print(f"{i + 1}) " + str(anim))

    try:
        my_anim = ANIMATIONS[int(input("Which one would you like?: ")) - 1]
    except:
        my_anim = ANIMATIONS[int(input("Which one would you like?: ")) - 1]


    imagel = [] #X-Z plane atm


    imagel.append([])
    for _ in range(width + 2):
        imagel[-1].append(frame_material)        
        
    

    for y in tqdm.trange(height):    
        imagel.append([frame_material])    
        

        for x in range(width):
            p1 = image.getpixel((x, y))  # Get RGB values at (x, y)
            score = float("inf")
            prev = "dirt"

            for key in tiles_blended_pixel.keys():
                my_score = distance(p1, tiles_blended_pixel[key])
                if my_score < score:
                    score = float(my_score)
                    prev = key

            imagel[-1].append(prev)
                

        imagel[-1].append(frame_material)


    imagel.append([])
    for _ in range(width + 2):
        imagel[-1].append(frame_material)        


    stop = True
    y_level = -59
    new_img.append([0,-60,0,"torch", "top"])
    new_img.append([0,-60,-1,"piston", "2"])
    new_img.append([0,-60,-2,"target"])
    new_img.append([0,-60,-3,"redstone_lamp"])
    new_img.append([0,-60,-4,"repeating_command_block"])
    
    

    prev_layer = None
    for blocks in my_anim(imagel):
        print("\n")

        print(f"Blocks: {blocks}")
        print(f"prev_layer: {prev_layer}")
        
        for i,j in blocks:
            if not stop:
                direction = None

                for si, sj in prev_layer:
                    if si == i + 1 and sj == j:
                        direction = "east"
                    elif si == i - 1 and sj == j:
                        direction = "west"
                    elif sj == j + 1 and si == i:
                        direction = "south"
                    elif sj == j - 1 and si == i:
                        direction = "north"
                    if direction is not None:
                        print(f"{(i,j)} as {direction} |", end = " ")
                        break
                    
                if direction is None:
                    raise Exception("Animation invalid, non-supported blocks returned")

                new_img.append([i, y_level - 1, j, 'torch', direction])
            else:
                pass
                
            new_img.append([i, y_level, j, imagel[i][j]])

        if stop:
            stop = False

        prev_layer = list(blocks)
        y_level += 1



    make_mosaic([row[::-1] for row in imagel], tile_raw) #Flip horizontally to undo prior flip
    input("This is how it will look... Click enter to continue...")

    print("P.S. To give yourself a repeating command block type in '/give @s command_block'")
    print("Then I've copied a command to your clipboard, make sure to paste this command in a command block (with settings: repeating, no redstone, always on)")
    pyperclip.copy("/kill @e[type=item]")
    print("You can then start the animation from afar by shooting the target block with a bow & arrow or just break a torch.")
    


if ON_GROUND:
    for i in range(len(new_img)):
        new_img[i][1], new_img[i][2] = new_img[i][2], new_img[i][1]

if SWAP_X_Z: 
    for i in range(len(new_img)):
        new_img[i][0], new_img[i][2] = new_img[i][2], new_img[i][0] #Now X is constant

def custom_key(item):
    x, y, z = item[0:3]
    return y*10000 + (x+z)

new_img = sorted(new_img, key=custom_key)

main_js_stuff= """

import { world } from "@minecraft/server";

world.afterEvents.playerBreakBlock.subscribe((event) => {
    //portal block only works on z-axis, furnaces only work on x-axis
    const arr = """ + str(new_img) + """;


    const player = event.player; // Player that broke the block for this event.
    const block = event.block; // Block impacted by this event. Note that the typeId if this block will ALWAYS be air.
    const permutation = event.brokenBlockPermutation; // Returns permutation information about this block before it was broken.
    
    if(permutation.type.id != "minecraft:grass_block"){
        return;
    }
    player.sendMessage(
        `You have broken ${permutation.type.id} at ${block.x}, ${block.y}, ${block.z}`
    ); // Sends a message to player.

    const dim = world.getDimension("overworld");
    

    let flip_axis = false;
    for (let i = 0; i < arr.length; i++) {
            if(arr[i][3] == "portal"){
                flip_axis = true;
                break;
            }
    } 
    

        for (let i = 0; i < arr.length; i++) {
            try{
                if (arr[i][3] == "torch"){
                        dim.runCommand("setblock " + arr[i][0] + " " + arr[i][1] + " " + arr[i][2] + ' torch ["torch_facing_direction"="' + arr[i][4] + '"]');
                }else if (arr[i][3] == "piston"){
                        dim.runCommand("setblock " + arr[i][0] + " " + arr[i][1] + " " + arr[i][2] + ' piston ["facing_direction"=' + arr[i][4] + ']');
                }else if (flip_axis){
                    dim.getBlock({ x: arr[i][2], y: arr[i][1], z: arr[i][0] }).setType("minecraft:" + arr[i][3]);
                }else{
                    dim.getBlock({ x: arr[i][0], y: arr[i][1], z: arr[i][2] }).setType("minecraft:" + arr[i][3]);
                }
            }catch(e){

            }
        }

    player.sendMessage(
        `Filled`
    ); // Sends a message to player.
});

"""

with open(main_js_loc, "w+") as file:
    file.write(main_js_stuff)

import uuid
from PIL import Image

img = Image.open(img_path)
img = img.resize((512, 512), Image.LANCZOS)
    

dependency_template = """,
        {
            "uuid": "$uid3",
            "version": [1, 0, 0]
        }
"""
if PERFECT_MATCH:
    with open("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorRP/manifest.json") as f:
        template = f.read()
    
    ref_uuid = str(uuid.uuid4())
    template = template.replace('$uid3', ref_uuid)
    template = template.replace('$uid4', str(uuid.uuid4()))

    with open("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorRP/manifest.json", "w+") as f:
        f.write(template)

    img.save("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorRP/pack_icon.png")


else:
    shutil.rmtree("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorRP")

with open("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorBP/manifest.json") as f:
    template = f.read()


if PERFECT_MATCH:
    template = template.replace("$dependency", dependency_template.replace("$uid3", ref_uuid))
else:
    template = template.replace("$dependency", "")

template = template.replace('$uid1', str(uuid.uuid4()))
template = template.replace('$uid2', str(uuid.uuid4()))
img.save("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorBP/pack_icon.png")

with open("tempMinecraftBedrockPixelArtGenerator/MinecraftBedrockPixelArtGeneratorBP/manifest.json", "w+") as f:
    f.write(template)


print("\n\nTO SUMMON IN EVERYTHING, YOU NEED TO BREAK A GRASS BLOCK NEAR 0,0")

shutil.make_archive('tempMinecraftBedrockPixelArtGenerator', 'zip', 'tempMinecraftBedrockPixelArtGenerator')
os.rename("tempMinecraftBedrockPixelArtGenerator.zip", "MinecraftBedrockPixelArtGenerator.mcaddon")
shutil.rmtree('tempMinecraftBedrockPixelArtGenerator')