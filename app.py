import streamlit as st
import numpy as np
from PIL import Image
import io
import time
from face_recognition_utils import calculate_face_distance, validate_image, preprocess_image

# Page configuration
st.set_page_config(
    page_title="Face Distance Calculator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Header section
st.title("🔍 Face Distance Calculator")
st.markdown("### Professional Identity Verification System")

# Use cases section
st.markdown("""
**Use-cases:**
- ✅ Patient identity check in clinics
- 🏗️ Worker verification on construction sites  
- 🏢 Visitor access validation in corporate buildings
""")

st.divider()

# Initialize session state
if 'comparison_result' not in st.session_state:
    st.session_state.comparison_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Create two columns for image uploads
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Reference Image")
    st.markdown("Upload the reference face image (ID photo, employee database, etc.)")
    
    uploaded_file1 = st.file_uploader(
        "Choose reference image",
        type=['png', 'jpg', 'jpeg'],
        key="ref_image",
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file1 is not None:
        try:
            image1 = Image.open(uploaded_file1)
            st.image(image1, caption="Reference Image", use_container_width=True)
            
            # Validate image
            validation_result1 = validate_image(image1)
            if validation_result1["valid"]:
                st.success("✅ Valid image detected")
            else:
                st.error(f"❌ {validation_result1['error']}")
                
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

with col2:
    st.subheader("📷 Comparison Image")
    st.markdown("Upload the image to verify against the reference")
    
    uploaded_file2 = st.file_uploader(
        "Choose comparison image",
        type=['png', 'jpg', 'jpeg'],
        key="comp_image",
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file2 is not None:
        try:
            image2 = Image.open(uploaded_file2)
            st.image(image2, caption="Comparison Image", use_container_width=True)
            
            # Validate image
            validation_result2 = validate_image(image2)
            if validation_result2["valid"]:
                st.success("✅ Valid image detected")
            else:
                st.error(f"❌ {validation_result2['error']}")
                
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")

st.divider()

# Compare button and results section
col_btn, col_spacer = st.columns([1, 3])

with col_btn:
    compare_button = st.button(
        "🔍 Compare Faces",
        type="primary",
        disabled=uploaded_file1 is None or uploaded_file2 is None or st.session_state.processing,
        use_container_width=True
    )

# Processing and results
if compare_button and uploaded_file1 is not None and uploaded_file2 is not None:
    st.session_state.processing = True
    
    # Show loading state
    with st.spinner("🔄 Processing images and calculating face distance..."):
        try:
            # Load and validate images
            image1 = Image.open(uploaded_file1)
            image2 = Image.open(uploaded_file2)
            
            # Validate both images
            validation1 = validate_image(image1)
            validation2 = validate_image(image2)
            
            if not validation1["valid"]:
                st.error(f"❌ Reference image error: {validation1['error']}")
                st.session_state.processing = False
                st.stop()
                
            if not validation2["valid"]:
                st.error(f"❌ Comparison image error: {validation2['error']}")
                st.session_state.processing = False
                st.stop()
            
            # Preprocess images
            processed_img1 = preprocess_image(image1)
            processed_img2 = preprocess_image(image2)
            
            # Calculate face distance
            result = calculate_face_distance(processed_img1, processed_img2)
            
            st.session_state.comparison_result = result
            st.session_state.processing = False
            
        except Exception as e:
            st.error(f"❌ Processing error: {str(e)}")
            st.session_state.processing = False

# Display results
if st.session_state.comparison_result is not None:
    st.subheader("📊 Comparison Results")
    
    result = st.session_state.comparison_result
    
    # Create metrics display
    col_dist, col_sim, col_status = st.columns(3)
    
    with col_dist:
        st.metric(
            label="Face Distance",
            value=f"{result['distance']:.4f}",
            help="Lower values indicate higher similarity"
        )
    
    with col_sim:
        similarity_percentage = (1 - result['distance']) * 100
        st.metric(
            label="Similarity Score",
            value=f"{similarity_percentage:.1f}%",
            help="Percentage similarity between faces"
        )
    
    with col_status:
        if result['is_match']:
            st.success("✅ MATCH")
            st.markdown("**Status:** Faces match")
        else:
            st.error("❌ NO MATCH")
            st.markdown("**Status:** Faces do not match")
    
    # Detailed results
    st.subheader("📋 Detailed Analysis")
    
    # Confidence level
    confidence_level = "High" if result['confidence'] > 0.8 else "Medium" if result['confidence'] > 0.6 else "Low"
    confidence_color = "green" if result['confidence'] > 0.8 else "orange" if result['confidence'] > 0.6 else "red"
    
    st.markdown(f"""
    **Confidence Level:** :{confidence_color}[{confidence_level} ({result['confidence']:.2f})]
    
    **Threshold Used:** {result['threshold']}
    
    **Processing Time:** {result['processing_time']:.2f} seconds
    
    **Face Detection Status:**
    - Reference Image: {'✅ Face detected' if result['face_detected_ref'] else '❌ No face detected'}
    - Comparison Image: {'✅ Face detected' if result['face_detected_comp'] else '❌ No face detected'}
    """)
    
    # Technical details (expandable)
    with st.expander("🔧 Technical Details"):
        st.json({
            "distance_value": result['distance'],
            "similarity_percentage": f"{similarity_percentage:.2f}%",
            "threshold": result['threshold'],
            "confidence": result['confidence'],
            "processing_time_ms": f"{result['processing_time'] * 1000:.0f}",
            "face_detected_reference": result['face_detected_ref'],
            "face_detected_comparison": result['face_detected_comp'],
            "is_match": result['is_match']
        })

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>
Professional Face Recognition System | Secure Identity Verification<br>
For technical support or questions, please contact your system administrator.
</small>
</div>
""", unsafe_allow_html=True)

# Clear results button
if st.session_state.comparison_result is not None:
    if st.button("🗑️ Clear Results", type="secondary"):
        st.session_state.comparison_result = None
        st.rerun()
