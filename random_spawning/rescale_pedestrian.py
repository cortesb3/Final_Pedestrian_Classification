"""
    Rescaling code for random spawning
"""
import numpy as np
import PIL.Image as Image
import cv2

def calc_bottom_pixel_position(input_array):
    # Find bottom pixel and index
    bottom_coord = input_array.nonzero()[0].max()
    bottom_coord_idx = input_array.nonzero()[0].argmax()
    
    # Use index of bottom pixel to find corresponding x coordinate
    left_coord = input_array.nonzero()[1][bottom_coord_idx]
    
    return bottom_coord, left_coord

def resize_mask(mask_image, factor, mode):
    # If mode is mask, convert PIL image to grayscale
    if (mode == 'mask'):
        mask_image = mask_image.convert('L')
    
    # Create original mask array and resized mask array   
    mask_array = np.array(mask_image)
    print(mask_array.shape)
    resized_mask = mask_image.resize((int(factor*mask_image.size[0]), int(factor*mask_image.size[1])), Image.NEAREST) #in RGB
    resized_array = np.array(resized_mask)
    
    # Calculate original position
    mask_bottom, mask_x_position = calc_bottom_pixel_position(mask_array)

    # Calculate resized position
    resized_bottom, resized_x_position = calc_bottom_pixel_position(resized_array)
    
    # Define the transition matrix
    x_shift = mask_x_position - resized_x_position
    y_shift = mask_bottom - resized_bottom
    M = np.float64([[1,0,x_shift], [0,1,y_shift]])
    
    # Convert PIL to cv2 format to perform translation
    resized_img = cv2.cvtColor(np.array(resized_mask), cv2.COLOR_RGB2BGR)
    
    # Choose correct referance image
    if(factor >= 1):  
        height, width = resized_img.shape[:2] 
    else:
        width, height = mask_image.size
    
    # Convert back to PIL image
    shifted_resized_mask = cv2.warpAffine(resized_img, M, (width,height))
    print(shifted_resized_mask.nonzero())
    resized_img = Image.fromarray(cv2.cvtColor(shifted_resized_mask, cv2.COLOR_BGR2RGB))
    resized_img.show()
    
    # Create new blank PIL image of the original mask size
    new_mask_img = Image.new('RGB', mask_image.size, color=(0, 0, 0))

    # Paste the resized PIL image to the blank PIL image at the specified position
    new_mask_img.paste(resized_img, (0,0))
    return new_mask_img

def calc_mask_dimension(input_array):
    width = abs(input_array.nonzero()[1].max() - input_array.nonzero()[1].min())
    height = abs(input_array.nonzero()[0].max() - input_array.nonzero()[0].min())
    
    return width, height

def mask_size_checker(mask_image, resized_image, factor, mode):
    # Convert PIL image to numpy array
    mask_array = np.array(mask_image)
    resized_array = np.array(resized_image)
    
    # Calculate the original and resized bottom-left position
    mask_bottom, mask_x_position = calc_bottom_pixel_position(mask_array)
    resized_bottom, resized_x_position = calc_bottom_pixel_position(resized_array)
    
    # Check if image was scaled at correct location
    if (mask_bottom != resized_bottom or mask_x_position != resized_x_position):
        print(f'Mask with a factor of {factor} was not resized in the correct location.')
        return False
    
    # Calculate the original and resized mask width and height
    mask_width, mask_height = calc_mask_dimension(mask_array)
    resized_width, resized_height = calc_mask_dimension(resized_array)
    
    # Check if image was scaled out of frame
    if((resized_height < mask_height*factor-4 or resized_height > mask_height*factor+4) or (resized_width < mask_width*factor -4 or resized_width > mask_width*factor+4)):
        print(f'The mask with a factor of {factor} was scaled out of bounds.')
        return False
    
    # If no condition was violated print and save outcome
    print(f'Factor: {factor}')
    print(f'Original w x h: {mask_width} x {mask_height} \nResized: {resized_width} x {resized_height} \nFrom top-left corner: right: {resized_x_position} down: {resized_bottom}')
    print('--------------------')
    resized_mask_image.save(f'036358_factor_{factor}_{mode}.png')
    
    return True
    
mode = 'person'
original_mask_image = Image.open(f".\\input\\input_{mode}\\2022-02-07-13-29-49_016527_right_rectilinear_gt_panoptic_{mode}.png")
original_mask_image.show()
factor_list = [0.25, 0.5, 0.75, 1.25, 5]
print(f'Mode: {mode}')

for factor in factor_list:
    resized_mask_image = resize_mask(original_mask_image, factor, mode)
    resized_mask_image.show()
    scaled_correctly = mask_size_checker(original_mask_image, resized_mask_image, factor, mode)

