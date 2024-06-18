# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")

# Altura de la cámara desde el suelo (en metros)
altura_camara = 1.0  # Supongamos que la cámara está a 1 metro del suelo

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados en tiempo real
    resultados = model.predict(frame, imgsz=640, conf=0.7)

    # Verificar si se detectaron botellas
    if len(resultados) > 0:
        # Obtener el primer resultado
        primer_resultado = resultados[0]

        # Imprimir la estructura de los datos en primer_resultado.boxes
        print(primer_resultado.boxes)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()