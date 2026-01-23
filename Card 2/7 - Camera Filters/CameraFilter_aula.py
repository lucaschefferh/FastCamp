#Importação das bibliotecas
import cv2
import sys
import numpy

#tipos de modos
PREVIEW  = 0  #modo normal, sem filtro
BLUR     = 1  #modo desfoque
FEATURES = 2  #modo deteccção
CANNY    = 3  #modo deteccão de bordas  

#dicionario de parametros para a o modo deteccção de bordas
feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

#configurações para abrir a camera
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

image_filter = PREVIEW
alive = True

#nome da janela
win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

#fonte da captura
source = cv2.VideoCapture(s)

#loop
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame = cv2.flip(frame, 1)

    #processa o frame dependendo do filtro selecionado
    if image_filter == PREVIEW:
        result = frame #sem filtro, apenas mostra o frame original
    elif image_filter == CANNY:
        #aplica o detector de bordas 
        result = cv2.Canny(frame, 80, 150)
    elif image_filter == BLUR:
        #aplica um filtro de blur
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        #converte para escala de cinza para detecção de cantos
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #detecta cantos na imagem com os parâmetros definidos
        corners = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
        if corners is not None:
            #desenha círculos verdes em cada canto detectado
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    #exibe a imagem resultante na janela
    cv2.imshow(win_name, result)

    #aguarda interação do teclado por 1ms
    key = cv2.waitKey(1)
    #se pressionar Q ou ESC encerra o programa
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    #tecla C ativa o filtro detector de bordas
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    #tecla B ativa o filtro blur
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    #tecla F ativa a detecção de cantos
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    #tecla P retorna ao modo normal
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

#libera os recursos da camera e fecha a janela
source.release()
cv2.destroyWindow(win_name)