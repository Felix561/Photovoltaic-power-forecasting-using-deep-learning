import logging
import shutil
import os
import datetime
import psutil
# Task of the skript: 
# daily supervision/Monitoring of the RBpi (e.g. Memory, Tempriture, CPU usage...)
# and backup Data and Meatdata (e.g. log files and image data onto SD and/or USB Drive!)


# Configure logging
log_file_path = '/home/Pi/BA_Skripte/code/logging/my_daily_skript_log.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')

# Define a list of log file names
log_files = [
    '/var/log/syslog',
    '/home/Pi/BA_Skripte/code/logging/my_bootup_python_skript_log.log',
    '/home/Pi/BA_Skripte/code/logging/my_daily_skript_log.log',
    '/home/Pi/BA_Skripte/code/logging/my_data_skript_log.log',
]

# Define target paths for the SD card and USB drive
sd_card_path = '/mnt/SD-Backup/MY_BACKUP/Logs'
usb_drive_path = '/mnt/DATABASE/Logs'



def copy_log_files(destination_path):
    for log_file in log_files:
        try:
            shutil.copy(log_file, os.path.join(destination_path, os.path.basename(log_file)))
        
        except Exception as e:
            logging.error(f"Error copying {log_file} to {destination_path}: {str(e)}")



def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_cpu_temperature():
    try:
        temperature = float(os.popen("vcgencmd measure_temp").readline().replace("temp=", "").replace("'C\n", ""))
        return temperature
    except Exception as e:
        return None

def get_disk_space(path):
    try:
        disk = psutil.disk_usage(path)
        return disk.total, disk.used, disk.percent
    except Exception as e:
        return None

def check_usb_drive_usage(usb_path):
    total, used, percent = get_disk_space(usb_path)
    if percent > 80:
        return (f"USB drive usage exceeded 80% ({percent}%)", total, used)
    return (None, total, used)

def log_memory_usage():
    sd_total, sd_used, _ = get_disk_space('/mnt/SD-Backup')
    usb_path = '/mnt/DATABASE'

    logging.info(f"SD Card: Total={(sd_total/1e+9)} GB Used={(sd_used/1e+9)} GB")


def log_system_uptime():
    # Erhalte die Uptime des Systems in Sekunden
    uptime_seconds = psutil.boot_time()
    
    # Konvertiere die Uptime von Sekunden in Tage, Stunden, Minuten und Sekunden
    uptime_days, uptime_hours, uptime_minutes = uptime_seconds // 86400, (uptime_seconds // 3600) % 24, (uptime_seconds // 60) % 60
    
    # Logge die Uptime
    logging.info(f'System Uptime: {int(uptime_days)} Tage, {int(uptime_hours)} Stunden, {int(uptime_minutes)} Minuten')


def main():
    # Copy log files to the SD card
    copy_log_files(sd_card_path)

    # Copy log files to the USB drive
    copy_log_files(usb_drive_path)

    #logging.info("Log file backup completed.")

    cpu_usage = get_cpu_usage()
    cpu_temperature = get_cpu_temperature()

    log_memory_usage()
    log_system_uptime()
    
    usb_alert,usb_total, usb_used = check_usb_drive_usage('/mnt/DATABASE')
    logging.info(f"USB: Total={(usb_total/1e+9)} GB Used={(usb_used/1e+9)} GB")

    logging.info(f"CPU Usage: {cpu_usage}%")
    logging.info(f"CPU Temperature: {cpu_temperature}°C")
    
    if usb_alert:
        logging.warning(f"USB Drive Alert: {usb_alert} !!!")

   

main()
logging.info("-------------------------")

