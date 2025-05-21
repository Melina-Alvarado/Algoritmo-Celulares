from ultralytics import YOLO      #Libreria Ultralytics
import cv2                        #OPEN CV

#Cargar Modelo preentrenado
model = YOLO("best.pt")      

#MODELO DE DETECCIÓN DE OBJETOS EN IMAGENES

# Cargar imagen con OpenCV
image_path = "imagen5.jpg"
img = cv2.imread(image_path)   # Carga la imagen con OpenCV

#Obtener resultados del modelo
results = model(image_path)   #Aplicar la imagen al modelo 

for result in results:            # Itera sobre los resultados (puede haber más de uno si hay varias imágenes)
    boxes = result.boxes          # Obtiene todas las cajas (bounding boxes) detectadas en la imagen
    img = result.plot()           # FUNCION DE LA LIBRERIA ULTRALYTICS

# Muestra la imagen resultante en una ventana emergente
cv2.imshow("Resultado YOLO", img)
cv2.waitKey(0)                    # Espera a que el usuario presione una tecla
cv2.destroyAllWindows()          # Cierra todas las ventanas de OpenCV