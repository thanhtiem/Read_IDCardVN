import cv2
import numpy as np 
from skimage.measure import compare_ssim
import imutils

def read_img(link):
    img = cv2.imread(link)
    return img

def crop_img(img):
    x_start = img.shape[1] / 11
    y_start = img.shape[0] / 10
    x_end = img.shape[1] / 4 + 5
    y_end = img.shape[0] / 3 + 20
    quoc_huy = img[int(y_start):int(y_end),int(x_start):int(x_end)]
    return quoc_huy

def OUTPUT_compare_vs_display(img_source, img_input_crop, source_gray, input_crop_gray):
    (score, diff) = compare_ssim(source_gray, input_crop_gray, full=True)
    diff = (diff * 255).astype("uint8")
    print("percent: ", int((score)*100), " %")

    if score >= 0.99:
        print('SAME!!')
        cv2.rectangle(img_input_crop,(int(img_input_crop.shape[1] / 11),int(img_input_crop.shape[0] / 10)), (int(img_input_crop.shape[1] / 4 + 5), int(img_input_crop.shape[0] / 3 + 20)), (0, 255, 0), 1)
    else:
        print('Not SAME!')
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if(len(cnts)>= 5):
            print('SAME!!')
            cv2.rectangle(img_source,(0,0), (img_source.shape[1], img_source.shape[0]), (0, 255, 0), 2)
            cv2.rectangle(img_input_crop,(int(img_input_crop.shape[1] / 11),int(img_input_crop.shape[0] / 10)), (int(img_input_crop.shape[1] / 4 + 5), int(img_input_crop.shape[0] / 3 + 20)), (0, 255, 0), 2)
            cv2.putText(img_input_crop, 'TRUE', (int(img_input_crop.shape[1] / 11),int(img_input_crop.shape[0] / 10) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(img_source, (x, y),(x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(img_input_crop,  (int(img_input_crop.shape[1] / 11) + x, int(img_input_crop.shape[0] / 10)+y), (int(img_input_crop.shape[1] / 11) + x + w, int(img_input_crop.shape[0] / 10)+y + h), (0, 0, 255), 1)
                cv2.rectangle(img_input_crop,(int(img_input_crop.shape[1] / 11),int(img_input_crop.shape[0] / 10)), (int(img_input_crop.shape[1] / 4 + 5), int(img_input_crop.shape[0] / 3 + 20)), (0, 0, 255), 2)
                cv2.putText(img_input_crop, 'NOT SAME', (int(img_input_crop.shape[1] / 11),int(img_input_crop.shape[0] / 10) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return [img_source, img_input_crop]

link_img_input = '../coderschool/image/cmnd1_fake2.jpg'
link_img_source = '../coderschool/image/cmnd1_QuocHuy.jpg'
def OUTPUT(link_img_input, link_img_source ):
    img_source = read_img(link_img_source)
    img_input= read_img(link_img_input)
    img_input_crop = crop_img(img_input)

    img_source_gray = cv2.cvtColor(img_source, cv2.COLOR_RGB2GRAY)
    img_input_crop_gray = cv2.cvtColor(img_input_crop, cv2.COLOR_RGB2GRAY)

    a = OUTPUT_compare_vs_display(img_source, img_input, img_source_gray, img_input_crop_gray)
    # cv2.imshow("source", a[0])
    # cv2.imshow("input", a[1])
    # cv2.waitKey(0)
    return a[1]
cv2.imshow('a',OUTPUT(link_img_input, link_img_source))
cv2.waitKey(0)
