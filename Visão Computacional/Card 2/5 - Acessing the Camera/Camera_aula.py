import cv2
import sys

#verifica se foi passado algum argumento na linha de comando
s = 0 
if len(sys.argv) > 1:
    s = sys.argv[1]

#inicializa a captura de vídeo com a fonte especificada
source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
#cria uma janela para exibição permitindo que ela seja redimensionável
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

#aguarda 1ms por uma tecla se for ESC sai do loop
while cv2.waitKey(1) != 27: 
    has_frame, frame = source.read()
    #se não conseguir ler o frame, sai do loop
    if not has_frame:
        break
    #mostra o frame na janela
    cv2.imshow(win_name, frame)

#libera a fonte de vídeo e destrói as janelas
source.release()
cv2.destroyWindow(win_name)