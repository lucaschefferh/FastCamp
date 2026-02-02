#Importação das bibliotecas
import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve

#Baixando e descompactando os arquivos
def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="", flush=True)

    urlretrieve(url, save_path)

    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)


URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), f"opencv_bootcamp_assets_12.zip")


if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)




#acessa o indice da camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

#cria um objeto de captura
source = cv2.VideoCapture(s)

#cria a saída da camera
win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#carregando modelo de inferencia
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

#parametros do modelo
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.6

#loop para a camera sair apenas com ESC
while cv2.waitKey(1) != 27:
    has_frame, frame = source.read() 
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    #roda o modelo
    net.setInput(blob)
    detections = net.forward()


       #percorre todas as detecções retornadas pela rede neural
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        #verifica se a confiança supera o limite mínimo definido
        if confidence > conf_threshold:
            #calcula as coordenadas 
            x_top_left = int(detections[0, 0, i, 3] * frame_width)
            y_top_left = int(detections[0, 0, i, 4] * frame_height)
            x_bottom_right  = int(detections[0, 0, i, 5] * frame_width)
            y_bottom_right  = int(detections[0, 0, i, 6] * frame_height)

            #desenha o retangulo
            cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right), (0, 255, 0))
            
            #prepara o texto do rótulo com o valor da confiança
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_top_left, y_top_left - label_size[1]),
                (x_top_left + label_size[0], y_top_left + base_line),
                (255, 255, 255),
                cv2.FILLED,
            )
            cv2.putText(frame, label, (x_top_left, y_top_left), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    
    #atualiza a janela com o frame processado
    cv2.imshow(win_name, frame)

#libera a camera
source.release()
cv2.destroyWindow(win_name)