# Comprehensive Demo Notebook - Execution Order

To run the notebook successfully, execute cells in this order:

## 1. Setup (Cells 1-9)
- Imports and path setup
- Visualization helper functions
- Segment visualization functions

## 2. Load Models & Data (Cells 10-14)
- Load ResNet50, ViT, GPT-2 models
- Prepare test images and text

## 3. Run Explanations (Cells 15-27)
- **IntGrad**: ResNet50 (Cell 15-16) and GPT-2 (Cell 17)
- **LIME**: ResNet50 (Cell 18-19) and GPT-2 (Cell 20-21)
- **SHAP**: ResNet50 (Cell 22-23) and GPT-2 (Cell 21)
- **GradCAM**: ResNet50 (Cell 24-25) and GPT-2 (Cell 26-27)

## 4. Analysis & Comparisons (Cells 28+)
- Visual comparison of all methods
- Segment consistency demonstrations
- Performance benchmarks

## Key Points
- All explanation variables (intgrad_resnet_exp, lime_resnet_exp, shap_resnet_exp, gradcam_resnet_exp) must be created before comparison cells
- The notebook now includes SHAP for both ResNet50 and GPT-2
- Segments visualization uses `segments[0]` to get 2D array for plt.imshow