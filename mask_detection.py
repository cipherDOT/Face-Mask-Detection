# -------------------------------------------------------------------------------------------------------------------------#

import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# -------------------------------------------------------------------------------------------------------------------------#

video = cv2.VideoCapture(0)

np.set_printoptions(suppress=True)

model = tensorflow.keras.models.load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)

# -------------------------------------------------------------------------------------------------------------------------#

while True:
    ret, frame = video.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(img)

    im_np = np.asarray(im_pil)

    image = ImageOps.fit(im_pil, size)

    image_array = np.asarray(image)

    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)

    if float(prediction.flatten()[1]) > 0.50:
        text = 'With mask'
    else:
        text = 'Without mask'

    my_image = cv2.putText(frame, text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------------------------------------------------------------------#
