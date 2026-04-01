from PIL import Image

def crop_and_save_image(input_path, output_path, top_left, bottom_right):

    image = Image.open(input_path)
    # Define the coordinates of the top-left and bottom-right corners
    left, top = top_left
    right, bottom = bottom_right
    # Crop the image using the provided coordinates
    cropped_image = image.crop((left, top, right, bottom))
    cropped_image.save(output_path)

    print("Crop completed")