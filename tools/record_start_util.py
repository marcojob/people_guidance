import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import os
import re

from time import sleep

SSD_FOLDER = "/media/pi/SSD/"
RECORD_TIME = 30

def button_callback(channel):
    folder_number = 0

    for name in os.listdir(SSD_FOLDER):
        out = re.search(r'outdoor_dataset_([0-9]*)', name)
        if out:
            number = int(out.group(1))
            if folder_number < number:
                folder_number = number

    if folder_number == 0:
        folder_number = 1

    folder_number += 1

    folder_name = SSD_FOLDER + f"outdoor_dataset_{folder_number:02d}"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    print(folder_name)
    os.system(f'timeout {RECORD_TIME} python3 /home/pi/people_guidance/main.py --record {folder_name}')

if __name__ == '__main__':
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    # GPIO.add_event_detect(10, GPIO.RISING, callback=button_callback)
    while True:
        sleep(0.01)
        if GPIO.input(10) == GPIO.HIGH:
            button_callback(0)

