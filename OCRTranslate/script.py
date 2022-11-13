## Getting all the required libraries
import re
import argparse
import requests

from bs4 import BeautifulSoup as bs

import pandas as pd
import numpy as np

from deep_translator import GoogleTranslator

import imutils
import cv2

import pytesseract

## Setting the API KEY to use in case of 'lang_detect' function
API_KEY = '9d83b7f02634514f56b7f59076eb15eb'

## Getting the GoogleTranslator as a dictionary for language and language code
langs_dict = GoogleTranslator().get_supported_languages(as_dict=True)
print(langs_dict)

## Load webpage content
r = requests.get('https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html')

## Convert to beautifulsoup object
soup = bs(r.content, features = 'lxml')

## Extracting table from the website
table = soup.find_all('table')
df = pd.read_html(str(table))[0]

## Doing preprocessing on the table to obtain the required dataframe
df = df.drop([0,1]).reset_index().drop(columns = 'index')
df = df.drop(df.columns[2:], axis = 1)
df = df.rename(columns = {'LangCode': 'tesseract_code'})
lower_column = df['Language'].apply(lambda x: x.lower())
df.insert(2, column = 'language', value = lower_column)
df.drop(columns = 'Language', inplace = True)
df['google_translate_code'] = ''

for i, row in df.iterrows():
    if row['language'] in list(langs_dict.keys()):
        row['google_translate_code'] = langs_dict[row['language']]

''' 
now the df dataframe contains the mapping for
both the languages codes of tesseract and googletranslate
'''

## Building an image parser
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required = True,
	help = '''path to input image to be OCR'd''')
ap.add_argument('-l', '--lang', required = True,
	help = '''language that Tesseract will use when OCR'ing''')
ap.add_argument('-t', '--to', type = str, default = 'english',
	help = '''language that we'll be translating to''')
ap.add_argument('-p', '--psm', type = int, default = 3,
	help = '''Tesseract PSM mode''')

args = vars(ap.parse_args())

image = cv2.imread(args["image"])

## Defining image preprocessing functions
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 3)
 
# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# skew correction
def deskew(image, angle):
    rotated = imutils.rotate_bound(image, float(angle))
    return rotated

# template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

## different preprocessing function applied to the image
osd = pytesseract.image_to_osd(image)
angle = re.search('(?<=Rotate: )\d+', osd).group(0)

if angle != 0:
    image = deskew(image, angle)

''' ---------------------------------------------------------------'''
## Starting the language translation and image preprocessing steps
result = []

source_tess = df[df.eq(args['lang']).any(1)]['tesseract_code'].tolist()[0]
source_google = df[df.eq(args['lang']).any(1)]['google_translate_code'].tolist()[0]
target_google = df[df.eq(args['to']).any(1)]['google_translate_code'].tolist()[0]

options = '-l {} --psm {}'.format(source_tess, args['psm'])

text = pytesseract.image_to_string(image, config = options)

image = get_grayscale(image)
image = remove_noise(image)
image = thresholding(image)

sentences = text.replace('\n', ' ').split('.')

print('\n', 'ORIGINAL')
print('-----------')
print(text, '\n')

my_translator = GoogleTranslator(source_google, target_google)

for sentence in sentences:
    result.append(my_translator.translate(sentence))

result = '. '.join(result)

print('\n', 'TRANSLATION')
print('-----------')
print(result, '\n')

## Converting the result to a CSV file
new_df = pd.DataFrame(columns=['original_language', 'text', 'translated_language', 'translation'], dtype = object)

new_df.loc[len(new_df.index)] = [args['lang'], text, args['to'], result]

new_df.to_csv('translation_file.csv')

