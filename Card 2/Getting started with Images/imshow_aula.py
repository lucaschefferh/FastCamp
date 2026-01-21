
import cv2 
import matplotlib.pyplot as plt 

cb_img = cv2.imread("checkerboard_color.png")
coke_img = cv2.imread("coca-cola-logo.png")

#Usando o imshow do matplotlib
plt.imshow(cb_img)
plt.title("matplotlib imshow")
plt.show()

#Usando o imshow do openCV, a imagem vai ficar por 8 segundos
window1 = cv2.namedWindow("w1")
cv2.imshow(window1, cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

#Usando o imshow do openCV, a imagem vai ficar até pressionar uma tecla
window2 = cv2.namedWindow("w2")
cv2.imshow(window2, coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

#Usando o imshow do openCV, a imagem vai ficar até pressionar a tecla 'q'
window3 = cv2.namedWindow("w3")
Alive = True
while Alive:
    cv2.imshow(window3, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        Alive = False
cv2.destroyWindow(window3)

cv2.destroyAllWindows()
stop = 1