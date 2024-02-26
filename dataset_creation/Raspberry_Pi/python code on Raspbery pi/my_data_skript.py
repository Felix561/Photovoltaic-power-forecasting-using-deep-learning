"""
To-do:
- Fixe camera Parameter einstellen nach Testlauf! 
"""

from picamera2 import Picamera2
import os
import datetime
import logging
import time
from PIL import Image
import subprocess

# Configure logging
logging.basicConfig(filename='/home/Pi/BA_Skripte/code/logging/my_data_skript_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

def configure_camera(camera):
    # Set the camera settings to constant values for fixed image quality
    """
    camera.resolution = (2592, 1944)  # Max resolution for "raw" data
    camera.shutter_speed = 10000  # Set a specific shutter speed in microseconds
    camera.iso = 100  # Set a specific ISO value
    camera.awb_mode = 'daylight'  # To a fixed value preset
    camera.awb_gains = (1.0, 1.0)  # Set white balance gains to fixed values
    camera.brightness = 50  # (0 to 100)
    camera.sharpness = 0  # (-100 to 100)
    camera.contrast = 0  # (-100 to 100)
    camera.saturation = 0  # (-100 to 100)
    camera.iso = 0  # Automatic (100 to 800)
    camera.exposure_compensation = -1  # (-25 to 25)  # Need to be set to a fixed value
    camera.exposure_mode = 'auto'
    camera.meter_mode = 'average'
    camera.awb_mode = 'auto'
    camera.rotation = 0
    camera.hflip = False
    camera.vflip = False
    camera.crop = (0.0, 0.0, 1.0, 1.0)"""
    config = camera.create_still_configuration(main={"size" : (3400,3040)}) # original max sensor. (4056,3040)
    camera.configure(config)


def capture_image(save_path_sd, save_path_usb):
    try:
        # Create a PiCamera instance
        with Picamera2() as camera:
            # Configure camera settings 
            configure_camera(camera)

            # Save the image to the specified path with the right name
            current_datetime = datetime.datetime.now()
            file_name = current_datetime.strftime("%Y%m%d%H%M.jpg")
            save_path_sd = os.path.join(save_path_sd, file_name)
            save_path_usb = os.path.join(save_path_usb, file_name)
            
            # Image 1
            camera.start()
            # Sleep for 2 seconds to allow the camera to warm up
            time.sleep(2)
            # Capture image array
            image_array = camera.capture_array()
            camera.stop()
            # prozessed image
            # Apply distortion correction
           
            p_image = Image.fromarray(image_array)
            # Resize the image to 224x224                     
            p_image = crop_image(p_image, (224,224))
                    
            # Save the prozessed image
            p_image.save(save_path_usb)
            
            time.sleep(2)
            
            # Image 2
            camera.start()
            # Sleep for 2 seconds to allow the camera to warm up
            time.sleep(2)
            # Capture image array
            image_array = camera.capture_array()
            camera.stop()
            # raw image
            # Save the raw image to the specified path (SD card)
            raw_image = Image.fromarray(image_array)
            raw_image.save(save_path_sd)
            


    except Exception as e:
        logging.error(f"Error capturing image: {str(e)}")



def crop_image(image, target_resolution=(224, 224)):
    try:
        # Open the input image
        original_image = image
        
        # Resize the image while maintaining the aspect ratio
        original_image.thumbnail(target_resolution)
        
        # Get the center coordinates of the image
        width, height = original_image.size
        left = (width - target_resolution[0]) / 2
        top = (height - target_resolution[1]) / 2
        right = (width + target_resolution[0]) / 2
        bottom = (height + target_resolution[1]) / 2
        
        # Crop the image to the target resolution
        cropped_image = original_image.crop((left, top, right, bottom))
        
        return cropped_image

    except Exception as e:
        logging.error(f"Error while croping the image: {e}")



def create_daily_folder(base_dir):
    # Create a daily folder based on the current date (e.g. "20220911" for September 11, 2022)
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    daily_folder = os.path.join(base_dir, current_date)
    
    # Create the folder if it doesn't exist
    if not os.path.exists(daily_folder):
        os.makedirs(daily_folder)
        
    
    return daily_folder



def main():
    # Define the base paths for the SD card and USB stick
    sd_card_base_dir = '/mnt/SD-Backup/MY_BACKUP/Data'
    usb_stick_base_dir = '/mnt/DATABASE/Data'

    # Create the daily folders for both storage locations
    sd_card_daily_dir = create_daily_folder(sd_card_base_dir)
    usb_stick_daily_dir = create_daily_folder(usb_stick_base_dir)
    
    # Capture the image and save it to the daily folders
    capture_image(sd_card_daily_dir, usb_stick_daily_dir)

    logging.info("------------------------------------------------")


def log_cpu_temperature():
    try:
        # Run a shell command to get the CPU temperature
        cpu_temperature = subprocess.check_output(["vcgencmd", "measure_temp"]).decode("utf-8").strip()

        # Log the CPU temperature        
        logging.info(f"CPU Temperature: {cpu_temperature}")

    except Exception as e:
        logging.error(f"Error logging CPU temperature: {str(e)}")


if __name__ == "__main__":
    log_cpu_temperature()
    time_houre = int(datetime.datetime.now().strftime("%H"))
    if time_houre >= 6 and time_houre < 18 :
        main()
    else:
        exit
