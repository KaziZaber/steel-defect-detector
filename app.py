import streamlit as st
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import anthropic
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

#page config
st.set_page_config(
    page_title="Steel Surface Defect Detector",
    page_icon="🔬",
    layout="wide"
)

#class names
CLASSES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
CONFIDENCE_THRESHOLD = 0.70

#defect descriptions for education section 
DEFECT_INFO = {
    'crazing': 'A network of fine surface cracks resembling dried mud. Caused by thermal stress during cooling.',
    'inclusion': 'Foreign material embedded in the steel surface. Caused by slag or refractory particles during casting.',
    'patches': 'Irregular areas of inconsistent surface texture. Caused by non-uniform descaling or scale adhesion.',
    'pitted_surface': 'Small cavities or depressions on the surface. Caused by scale entrapment during rolling.',
    'rolled-in_scale': 'Oxide scale pressed into the steel surface during rolling. Caused by inadequate descaling.',
    'scratches': 'Linear surface marks running parallel to rolling direction. Caused by worn guide equipment.'
}

#@st.cache_resource makes this function only runs once, the model stays loaded in memory — no reloading on every upload
@st.cache_resource
def load_model():
    # load ResNet18 architecture, pretrained=False(have own weights)
    model = models.resnet18(pretrained=False)
    
    #get the number of input features(512) to the original final layer
    num_features = model.fc.in_features
    
    #replace the original 1000-class final layer with 6-class layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),          #30% dropout for regularization
        nn.Linear(num_features, 6) 
    )
    
    #load trained weights from best_model.pth
    #map_location='cpu' to make it work without a GPU, not to crash
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    
    #set to evaluation mode to disable dropout for consistent predictions
    model.eval()
    return model

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

def get_gradcam(model, input_tensor, predicted_idx, original_image):
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    img_array = np.array(original_image.resize((224, 224)).convert('RGB')) / 255.0
    visualization = show_cam_on_image(img_array.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization

def get_llm_explanation(defect_class, confidence):
    client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    
    prompt = f"""You are an expert steel quality engineer at a hot-rolling steel mill with 20 years of experience in surface defect analysis and quality control.

A computer vision system has detected a {defect_class.replace('_', ' ')} defect on a steel surface with {confidence:.1f}% confidence.

Provide a concise professional engineering analysis in exactly 3 sentences:
1. What this defect is and how it appears on the steel surface
2. The most likely cause during the hot-rolling manufacturing process
3. The recommended corrective action for the production team

Be specific, technical, and practical. Write as a steel engineer would in an inspection report. Write in plain paragraph form only. No markdown, no bold, no headers, no bullet points. Just 3 plain sentences. Do not use first person."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

#main app
st.title("🔬 Steel Surface Defect Detector")
st.markdown("**Industrial AI system for automated steel surface quality control**")
st.markdown("Built with ResNet18 transfer learning — 95% validation accuracy on NEU Steel Surface Defect Dataset")

st.divider()

#sidebar info
with st.sidebar:
    st.header("About This System")
    st.markdown("""
    This system uses computer vision and AI to detect surface defects on steel, 
    explain what caused them, and show exactly where on the image the model made its decision.
    
    **Built for POSCO-relevant industrial AI research**
    
    **Features:**
    - ResNet18 transfer learning classifier
    - Grad-CAM explainability heatmap  
    - Confidence-based rejection
    - LLM engineering analysis
    
    **6 Defect Classes:**
    - Crazing
    - Inclusion
    - Patches
    - Pitted Surface
    - Rolled-in Scale
    - Scratches
    """)
    
    st.divider()
    st.header("Model Performance")
    st.metric("Validation Accuracy", "95.0%")
    st.metric("Baseline (Logistic Reg.)", "41.4%")
    st.metric("Improvement", "+53.6pp")

#main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Steel Surface Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a steel surface image to detect defects"
    )

if uploaded_file is not None:
    #load image
    image = Image.open(uploaded_file)
    
    #load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    #preprocess and predict
    with st.spinner("Analyzing image..."):
        input_tensor = preprocess_image(image)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probs, 1)
        
        confidence_pct = confidence.item() * 100
        predicted_class = CLASSES[predicted_idx.item()]
        all_probs = probs[0].tolist()
    
    #display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    with col2:
        st.subheader("Grad-CAM Heatmap")
        with st.spinner("Generating heatmap..."):
            gradcam_img = get_gradcam(model, input_tensor, predicted_idx.item(), image)
            st.image(gradcam_img, use_container_width=True)
        st.caption("Red = high model attention | Blue = low attention")
    
    st.divider()
    
    #prediction results
    if confidence_pct < CONFIDENCE_THRESHOLD * 100:
        st.warning(f"⚠️ LOW CONFIDENCE ({confidence_pct:.1f}%) — Recommend manual inspection")
        st.markdown(f"**Most likely defect:** {predicted_class.replace('_', ' ').title()} — but confidence is below the 70% threshold for automated classification.")
    else:
        st.success(f"✅ **Detected: {predicted_class.replace('_', ' ').title()}** — {confidence_pct:.1f}% confidence")
    
    #confidence breakdown
    with st.expander("View confidence scores for all classes"):
        for cls, prob in zip(CLASSES, all_probs):
            st.progress(prob, text=f"{cls.replace('_', ' ').title()}: {prob*100:.1f}%")
    
    st.divider()
    
    #LLM explanation
    st.subheader("Engineering Analysis")
    if confidence_pct < CONFIDENCE_THRESHOLD * 100:
        st.info("Engineering analysis is shown for reference — manual inspection recommended due to low confidence.")
    
    with st.spinner("Generating engineering analysis..."):
        explanation = get_llm_explanation(predicted_class, confidence_pct)
        st.markdown(f"*{explanation}*")
    
    st.divider()
    
    #defect education
    st.subheader("About This Defect Type")
    if predicted_class in DEFECT_INFO:
        st.info(DEFECT_INFO[predicted_class])

else:
    #show example when no image uploaded
    with col1:
        st.info("👆 Upload a steel surface image to begin analysis")
    
    st.divider()
    st.subheader("How This System Works")
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown("**1. Upload**\nUpload a steel surface image from your camera or files")
    with col_b:
        st.markdown("**2. Classify**\nResNet18 identifies the defect type with confidence score")
    with col_c:
        st.markdown("**3. Explain**\nGrad-CAM shows exactly where the model looked")
    with col_d:
        st.markdown("**4. Analyze**\nClaude AI generates engineering cause and action")
    
    st.divider()
    st.subheader("Defect Reference Guide")
    
    cols = st.columns(3)
    for i, (defect, description) in enumerate(DEFECT_INFO.items()):
        with cols[i % 3]:
            st.markdown(f"**{defect.replace('_', ' ').title()}**")
            st.caption(description)
            st.markdown("")