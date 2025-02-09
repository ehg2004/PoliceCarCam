import time
import gpiod

# Configurar o chip e a linha GPIO
CHIP = "/dev/gpiochip4"  # Verifique qual gpiochip corresponde ao seu pino
LINE_OFFSET = 110  # Altere para o número correto do GPIO

# Configurar a linha como saída
chip = gpiod.Chip(CHIP)
line = chip.get_line(LINE_OFFSET)
line.request(consumer="buzzer", type=gpiod.LINE_REQ_DIR_OUT)

# Liga o buzzer por 1 segundo e depois desliga
line.set_value(1)  # Liga
time.sleep(1)  
line.set_value(0)  # Desliga

# Libera o GPIO
line.release()
