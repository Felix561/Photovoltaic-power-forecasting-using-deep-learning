import os
import subprocess
import logging
import datetime


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


if __name__ == "__main__":

    time_houre = int(datetime.datetime.now().strftime("%H"))
    if time_houre >= 6 and time_houre < 18 :
        if not check_images_exist():
            execute_command()
            logging.info("my_data_skript.service stopped due to missing images... Stopped the Data service...")
    else:
        exit
    
