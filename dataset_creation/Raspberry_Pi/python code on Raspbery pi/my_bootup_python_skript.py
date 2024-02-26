import subprocess
import logging

# will mount the USB and SD Drive to the raspberry pi so that they can be used for data storage.


##### Logging funktioniert nicht immer... why? 


# Configure logging
log_file_path = '/home/Pi/BA_Skripte/code/logging/my_bootup_python_skript_log.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

# The commands you want to execute as a list of separate strings
commands = ["sudo mount /dev/sda1 /mnt/DATABASE",
            "sleep 2",
            "sudo mount /dev/mmcblk0p2 /mnt/SD-Backup"
]

# Use subprocess.run() to execute each command
try:
    for command in commands:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

        # Log the command execution
        logging.info(f"Command executed successfully: {command}")

        # Log the output
        logging.info("Command Output:")
        logging.info(result.stdout)

except subprocess.CalledProcessError as e:
    # Log the error
    logging.error(f"Error executing command: {command}")
    logging.error(f"Error message: {e.stderr}")

    # Raise the error again to handle it or exit the script as needed
    raise e

logging.info(f"Successfully mounted USB and SD Drive to the Raspberry Pi!")
logging.info("-----------------------------------------------------------")