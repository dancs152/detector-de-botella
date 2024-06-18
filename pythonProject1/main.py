from ultralytics import YOLO
import cv2

model = YOLO("best.pt")
ip_cam_url = "http://192.168.1.35:8080/video"
cap = cv2.VideoCapture(ip_cam_url)

while True:
    ret, frame = cap.read()
    resultados = model.predict(frame, imgsz=640, conf=0.7)
    anotaciones = resultados[0].plot()
    cv2.imshow("BOTELLAS", anotaciones)
    print(resultados)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()