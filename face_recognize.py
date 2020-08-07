
import cv2, sys, numpy, os
size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

print('Reconhecendo faces, por favor, esteja em boa luminosidade...')

(images, lables, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path, 0))
            lables.append(int(lable))
        id += 1
(width, height) = (220, 220)


(images, lables) = [numpy.array(lis) for lis in [images, lables]]


model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, lables)
model.read("classificadorLBPH.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor= 1.5, minSize= (30, 30))
    




    for (x, y, l, a) in faces:
        imagemFaces = cv2.resize(gray[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(im, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = model.predict(imagemFaces)
        if id == 1:
            nome = "Luciano"
        else:
            nome = "Desconhecido"
        cv2.putText(im, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(im, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow("Face", im)
    if cv2.waitKey(1) == ord('q'):
        break



