"""
    Code to overlay GT masks and self annotated segmentation masks
"""
import PIL.Image as Image
import os
import cv2

def annotation_images(rgb_folder, annotation_folder):
    i = 0
    for image in os.listdir(rgb_folder):
        if(image.endswith(".png")):
            title = image[0:21]
            i += 1
            for annotation in os.listdir(annotation_folder):
                if(annotation[0:21] == title):
                    background = Image.open(f'.\\{rgb_folder}\\'+image)
                    overlay = Image.open(f'.\\{annotation_folder}\\'+annotation)
                    
                    background = background.convert("RGBA")
                    overlay = overlay.convert("RGBA")
                    
                    new_img = Image.blend(background, overlay, 0.5)
                    new_img.save(f'annon_rgb_blue{i}.png', "PNG")
   

def main():
    annotation_images("50_img", "SegmentationClass")
    
    
if __name__ == '__main__':
    main()