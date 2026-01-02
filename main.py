import cv2 as cv #type: ignore
import time


class DetectorDeRosto:
    def __init__(self, modelo):
        self.modelo = modelo

    def detectar_rosto(self, frame):
        frame_cinza = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        rostos = self.modelo.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        return rostos


class Camera:
    def __init__(self, webcam, detector):
        self.detector = detector
        self.webcam = webcam

    def iniciando_webcam(self):
        self.webcam = cv.VideoCapture(0)

        if not self.webcam.isOpened():
            print("Não foi possível abrir a câmera")
            return False

        while True:
         ret, frame = self.webcam.read()
         if not ret:
             print("Não possível abrir a câmera, saindo...")
             time.sleep(2)
             break
         
         rostos = self.detector.detectar_rosto(frame)
         for (x, y, largura, altura) in rostos:
                cv.rectangle(frame, (x, y), (x + largura, y + altura), (0, 255, 0), 2)
         cv.imshow("Webcam:", frame)
         if cv.waitKey(1) == ord('q'):
            break
         
        self.webcam.release()
        cv.destroyAllWindows()
        
class App:
    def __init__(self):
        modelo = cv.CascadeClassifier(
            cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        detector = DetectorDeRosto(modelo)
        camera = Camera(0, detector)
        camera.iniciando_webcam()
       

App()
