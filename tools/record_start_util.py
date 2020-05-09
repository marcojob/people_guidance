import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import os
import re

SSD_FOLDER = "/media/pi/SSD/"

def button_callback():
    folder_number = 0

    for name in os.listdir(SSD_FOLDER):
        out = re.search(r'outdoor_dataset_([0-9]*)')
        if out:
            number = int(out.group(1))
            if folder_number < number:
                folder_number = number

    folder_name = SSD_FOLDER + f"outdoor_dataset_{folder_number:02d}"
    os.mkdir(folder_name)

    print(folder_name)

if __name__ == '__main__':
    GPIO.setwarnings(False) # Ignore warning for now
    GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
    GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    GPIO.add_event_detect(10, GPIO.RISING, callback=button_callback)
