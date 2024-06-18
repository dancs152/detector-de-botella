# Importamos las librerias
from ultralytics import YOLO
import cv2

# Leer nuestro modelo
model = YOLO("best.pt")

# Altura de la cámara desde el suelo (en metros)
altura_camara = 1.5  # Supongamos que la cámara está a 1 metro del suelo

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

        # Verificar si se detectaron botellas en el primer resultado
        if len(primer_resultado.boxes) > 0:
            # Iterar sobre las detecciones de botellas
            for deteccion in primer_resultado.boxes.data:
                # Obtener las coordenadas del cuadro delimitador
                x_min = deteccion[0]
                y_min = deteccion[1]
                x_max = deteccion[2]
                y_max = deteccion[3]

                # Calcular el tamaño de la botella en píxeles (ancho o alto, dependiendo de cómo está orientada)
                tamano_botella = max(x_max - x_min, y_max - y_min)

                # Calcular la distancia relativa entre la botella y la cámara
                # Suponemos que la altura de la botella es la mitad de la altura de la imagen
                # y usamos la fórmula de proporción para estimar la distancia
                distancia_estimada = 1 - ((altura_camara / 2) * tamano_botella / frame.shape[0])

                # Dibujar el cuadro delimitador en el fotograma
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

                # Mostrar la distancia estimada en el fotograma
                cv2.putText(frame, f'Distancia: {distancia_estimada:.2f} metros', (int(x_min), int(y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar fotogramas con detección y distancia estimada
    cv2.imshow("Detección de Botellas", frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()