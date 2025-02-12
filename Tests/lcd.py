import smbus
import time
from RPLCD.i2c import CharLCD

# Configuração do LCD
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2, dotsize=8)

def escrever_lcd(texto_linha1, texto_linha2):
    lcd.clear()
    lcd.write_string(texto_linha1)
    lcd.crlf()  # Pula para a segunda linha
    lcd.write_string(texto_linha2)

try:
    escrever_lcd("Hello, world!", "Radxa Rock 5C")
    time.sleep(5)
    lcd.clear()
except KeyboardInterrupt:
    lcd.clear()
    print("Programa encerrado.")
