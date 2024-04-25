import RPi.GPIO as GPIO

pin = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(pin, GPIO.OUT)

try:
    while True:
        GPIO.output(pin, GPIO.HIGH)  # Activate heating permanently
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
