import cv2
import pyautogui  # Necesario para obtener el tamaño de la pantalla

# Función para obtener las dimensiones de la pantalla de la notebook
def get_screen_size():
    screen_width, screen_height = pyautogui.size()
    return (screen_width, screen_height)

# Inicializar la captura de video
cap = cv2.VideoCapture(0)

# Cargar el clasificador Haar Cascade para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Obtener las dimensiones de la pantalla de la notebook
screen_width, screen_height = get_screen_size()

while cap.isOpened():
    # Leer un nuevo frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Invertir horizontalmente el frame
    frame = cv2.flip(frame, 1)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar recuadros alrededor de los rostros detectados y mostrar el mensaje
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Rostro detectado', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Ajustar el tamaño de la ventana a las dimensiones de la pantalla
    cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Face Detection', screen_width, screen_height)

    # Mostrar la imagen con los recuadros de los rostros detectados y el mensaje
    cv2.imshow('Face Detection', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
