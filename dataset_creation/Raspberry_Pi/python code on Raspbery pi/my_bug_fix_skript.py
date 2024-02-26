import os
import subprocess
import logging
import datetime
#import time
#from picamera2 import Picamera2
#from PIL import Image


logging.basicConfig(filename='/home/Pi/BA_Skripte/code/logging/BUG_FIX.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def check_images_exist():

    current_datetime = datetime.datetime.now()
    image_filename = current_datetime.strftime("%Y%m%d%H%M.jpg")
    
    dir1 = '/mnt/SD-Backup/MY_BACKUP/Data'
    dir2 = '/mnt/DATABASE/Data'
    
    path1 = os.path.join(dir1, image_filename)
    path2 = os.path.join(dir2, image_filename)

    return os.path.exists(path1) and os.path.exists(path2)


def execute_command():
    command = "sudo systemctl stop my_data_skript.service"
    try:
        subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")

"""
def execute_command_2():
    command = "sudo systemctl stop BUG_FIX.service"
    try:
        subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
"""

"""
def take_missing_image():

    current_datetime = datetime.datetime.now()
    image_filename = current_datetime.strftime("%Y%m%d%H%M.jpg")
    dir1 = '/mnt/SD-Backup/MY_BACKUP/Data'
    save_path_sd = os.path.join(dir1, image_filename)

    try:
        # Create a PiCamera instance
        with Picamera2() as camera:
            # Configure camera settings 
            config = camera.create_still_configuration(main={"size" : (3400,3040)}) # original max sensor. (4056,3040)
            camera.configure(config)
            
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
            logging.info("Missing Image was taken Successfully!")
            
    except Exception as e:
        logging.error(f"Error capturing missing image: {str(e)}")
"""


if __name__ == "__main__":

    time_houre = int(datetime.datetime.now().strftime("%H"))
    if time_houre >= 6 and time_houre < 18 :
        if not check_images_exist():
            execute_command()
            logging.info("my_data_skript.service stopped due to missing images... Stopped the Data service...")
        
            #take_missing_image()
            #if not check_images_exist():
            #    execute_command_2()
            #    logging.info("BUG_FIX.service stopped due to ERROR wihle taking missing image!!!")
            #else:
            #    exit
    else:
        exit
    
