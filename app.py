import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog
from skimage.transform import resize
from skimage.exposure import equalize_hist
from scipy.stats import kurtosis, skew

# Load the trained model
try:
    model = joblib.load('banknote_auth_model.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'banknote_auth_model.joblib' is in the same directory as this script.")
    st.stop()

st.title('Advanced Bank Note Authenticator')

st.write("""
This app uses advanced machine learning and image processing techniques to predict whether a bank note is authentic or counterfeit based on its visual features.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a banknote image...", type=["jpg", "png"])

def extract_features(image):
    # Resize image to a fixed size
    image = resize(image, (256, 256), anti_aliasing=True)
    gray = rgb2gray(image)
    
    # Apply histogram equalization
    gray_eq = equalize_hist(gray)
    
    # Extract LBP features
    lbp = local_binary_pattern(gray_eq, P=8, R=1)
    
    # Extract HOG features
    hog_feat = hog(gray_eq, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    # Calculate statistical features
    variance = np.var(gray_eq)
    skewness = skew(gray_eq.flatten())
    curtosis = kurtosis(gray_eq.flatten())
    lbp_mean = np.mean(lbp)
    lbp_var = np.var(lbp)
    
    return np.concatenate(([variance, skewness, curtosis, lbp_mean, lbp_var], hog_feat))

if uploaded_file is not None:
    # Display the uploaded image
    image = imread(uploaded_file)
    st.image(image, caption='Uploaded Banknote Image.', use_column_width=True)
    
    # Extract features
    features = extract_features(image)
    
    # Display calculated features
    st.write("Extracted Features:")
    feature_names = ['Variance', 'Skewness', 'Curtosis', 'LBP Mean', 'LBP Variance']
    for name, value in zip(feature_names, features[:5]):
        st.write(f"{name}: {value:.6f}")

    # When the 'Predict' button is clicked
    if st.button('Predict'):
        # Make a prediction
        prediction = model.predict([features])
        
        # Display the prediction
        if prediction[0] == 0:
            st.error('The bank note is predicted to be counterfeit.')
        else:
            st.success('The bank note is predicted to be authentic.')
        
        # Display prediction probability
        proba = model.predict_proba([features])
        st.write(f'Probability of being authentic: {proba[0][1]:.2%}')
        st.write(f'Probability of being counterfeit: {proba[0][0]:.2%}')
        
st.write("""
### Feature Information:
- Variance: Overall contrast of the image
- Skewness: Asymmetry of the pixel intensity distribution
- Curtosis: "Peakedness" of the pixel intensity distribution
- LBP Mean: Average texture information using Local Binary Patterns
- LBP Variance: Variation in texture information
- HOG Features: Histogram of Oriented Gradients, capturing edge and shape information

These features capture the overall visual characteristics of the note, making it harder for counterfeit notes with fake serial numbers or unusual patterns to pass as authentic.
""")

