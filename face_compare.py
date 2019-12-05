from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

def read_img(link_img):
    img = cv2.imread(link_img)
    # img = pyplot.imread(link_img)
    return img

# def detect_face(img):
#     classifier = cv2.CascadeClassifier('../Read_IDCardVN/haarcascade_xml/haarcascade_frontalface_default.xml')
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = classifier.detectMultiScale(gray, 1.1, 1)
#     for (x, y, w, h) in faces:
#         img[y:y+h,x:x+w]
#     return img

def detect_face(img):
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face

def extract_face(img, required_size=(224, 224)):
    detector = MTCNN()
    results = detector.detect_faces(img)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face_img = img[y1:y2, x1:x2]
    image = Image.fromarray(face_img)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def get_embeddings(img):
    faces = [extract_face(f) for f in img]
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    yhat = model.predict(samples)
    return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
        print((1 - (score / 2)) *100, ' %')
        return (1 - (score / 2)) *100
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        print((1 - (score / 2)) *100, ' %')
        return (1 - (score / 2)) *100
    
def OUTPUT_scoreFace(img1, img2):
    known_embedding = get_embeddings(img1)
    candidate_embedding = get_embeddings(img2)
    is_match(known_embedding, candidate_embedding)


link_img1 = '../Read_IDCardVN/image/nhat1_cmnd.jpg'
link_img2 = '../Read_IDCardVN/image/nhat1_selfie.jpg'
img1 = read_img(link_img1)
img2 = read_img(link_img2)
# a = detect_face(img1)
# b = detect_face(img2)
# cv2.imshow('cmnd', a)
# cv2.imshow("selfie", b)
# cv2.waitKey(0)
OUTPUT_scoreFace(img1, img2)

