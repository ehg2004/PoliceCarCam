# PoliceCarCam

conda create --prefix ./Tests/plateTest/env python==3.10 -y
conda activate ./Tests/plateTest/env
python3.10 -m pip install -r ./Tests/plateTest/Automatic-License-Plate-Recognition-using-YOLOv8/requirements.txt
git clone https://github.com/abewley/sort Tests/plateTest/Automatic-License-Plate-Recognition-using-YOLOv8
python3.10 -m pip install filterpy scikit-image lap
## usar python3.10 e conda ou venv para ambiente
## baixar modelo de https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing para LPR (License Plate Recognition)
## Para modelo da UFGRS: (Esse fork foi atualizado para python3, porem nao testei inteiro)
cd Tests/plateTest
git clone https://github.com/SothanaV/alpr-unconstrained.git
## Para clonar os modelos da UFRGS
cd alpr-unconstrained/
rm -f get-networks.sh
wget https://raw.githubusercontent.com/sergiomsilva/alpr-unconstrained/refs/heads/master/get-networks.sh
bash get-networks.sh

## Falta utilizar o modelo do ALPR-using-YOLOv8 com o OCR da UFRGS
## Para testar o alpr-unconstrained sozinho, necessário baixar o darknet e instalar, recomendo instalar na CPU mesmo
## Não consegui usar o alpr-unconstrained com o keras para o LPR e o OCR dele ainda, deve ter algum problema
## dentro do ALPR-using-YOLOv8 tem o testMain.py, que eu estava utilando para testar placa por placa do dataset da UFPR, que ficou localizado em 'Tests/plateTest/'
## Se não for usar a placa dew video, tem que mudar na construtora do easyOCR no arquivo utils.py, e algumas coisas na main,
## Se quiser testar o ALPR-using-YOLOv8, clonar o https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8.git e seguir o tutorial dnv, posso ter mudado alguma coisa na main
##
