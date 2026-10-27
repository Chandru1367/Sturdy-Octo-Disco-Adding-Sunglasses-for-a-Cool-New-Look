# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

## Name : Chandru M
## Reg NO : 212224230041

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:

1. Detects the face in an image.

2. Places a stylish sunglass overlay perfectly on the face.

3. Works seamlessly with individual passport-size photos.

4. Customizable for different sunglasses styles or photo types.

## Technologies Used:

1. Python
2. OpenCV for image processing
3. Numpy for array manipulations

## How to Use:

1. Clone this repository.
2. Add your passport-sized photo to the images folder.
3. Run the script to see your "cool" transformation!

## Applications:

1. Learning basic image processing techniques.
2. Adding flair to your photos for fun.
3. Practicing computer vision workflows.

## Program and Output

```python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Face Image
faceImage = cv2.imread(r"E:\College\THIRD YEAR\5TH SEM\DIPT\chandru.jpg")
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")

```
<img width="254" height="190" alt="image" src="https://github.com/user-attachments/assets/43bd1570-face-4c44-a430-6b51cf4d5e76" />

```python
faceImage.shape
```

<img width="116" height="22" alt="image" src="https://github.com/user-attachments/assets/4ac95873-cdb5-4024-8056-377214effa34" />

```python
#resized_faceImage.shape
faceImage.shape
```

<img width="113" height="28" alt="image" src="https://github.com/user-attachments/assets/1a1c6989-5108-408d-ad3e-a41f1270ae53" />

```python
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread(r"E:\College\THIRD YEAR\5TH SEM\DIPT\glass.png",-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```

<img width="431" height="230" alt="image" src="https://github.com/user-attachments/assets/5be5eec0-e318-4661-a448-daa6c8c16180" />

```python
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(190,50))
print("image Dimension ={}".format(glassPNG.shape))
```

<img width="177" height="18" alt="image" src="https://github.com/user-attachments/assets/36beec01-8ecb-460f-af27-36d4aa9ae8cb" />

```python
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="1210" height="203" alt="download" src="https://github.com/user-attachments/assets/e8418f8f-9fd8-4fd9-905e-f1eb5e224dd4" />

```python
# Make a copy
faceWithGlassesNaive = faceImage.copy()

# Resize the sunglasses (width=190, height=50)
glassResized = cv2.resize(glassBGR, (650, 180))

# Place the sunglasses over the eyes
faceWithGlassesNaive[650:830, 440:1090] = glassResized

# Display the result
plt.figure(figsize=(6, 8))
plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.axis('off')
plt.show()
```

<img width="434" height="636" alt="download" src="https://github.com/user-attachments/assets/8094a5b0-0023-4d56-abe7-ff685107c08c" />

```python
# Make the dimensions of the mask same as the input image
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1]
glassMask = np.float32(glassMask / 255.0)

# Resize the sunglasses and mask to match the eye region
glassBGR = cv2.resize(glassBGR, (650, 180))
glassMask = cv2.resize(glassMask, (650, 180))

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[650:830, 440:1090]

# Create the masked eye region
maskedEye = cv2.multiply(eyeROI.astype(np.float32), 1 - glassMask)

# Create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR.astype(np.float32) / 255.0, glassMask)

# Combine the sunglasses with the eye region
eyeRoiFinal = cv2.add(maskedEye / 255.0, maskedGlass)

# Convert to uint8
eyeRoiFinal = np.uint8(np.clip(eyeRoiFinal * 255, 0, 255))

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131)
plt.imshow(maskedEye.astype(np.uint8)[...,::-1])
plt.title("Masked Eye Region")

plt.subplot(132)
plt.imshow((maskedGlass * 255).astype(np.uint8)[...,::-1])
plt.title("Masked Sunglass Region")

plt.subplot(133)
plt.imshow(eyeRoiFinal[...,::-1])
plt.title("Augmented Eye and Sunglass")
```

<img width="1606" height="191" alt="download" src="https://github.com/user-attachments/assets/4296861c-a099-4085-891f-f26900f3c22f" />

```python
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[650:830, 440:1090] = eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20])
plt.subplot(121)
plt.imshow(faceImage[:,:,::-1])
plt.title("Original Image")

plt.subplot(122)
plt.imshow(faceWithGlassesArithmetic[:,:,::-1])
plt.title("With Sunglasses")
```

<img width="1615" height="1111" alt="download" src="https://github.com/user-attachments/assets/8fa599cf-3aca-49fb-8154-fd382b196767" />

