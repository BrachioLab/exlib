# Feature Attribution Methods Analysis - src/exlib/explainers/

## Overview

The `src/exlib/explainers/` directory contains implementations of various feature attribution methods for explaining model predictions. The codebase is well-structured with a common interface pattern and supports both image and text modalities.

## Architecture Analysis

### Core Design Pattern

1. **Base Class**: `FeatureAttrMethod` (in `common.py:42-51`)
   - All explainers inherit from this base class
   - Provides consistent interface with `forward(x, t, return_groups=False, **kwargs)`
   - Built on `nn.Module` for PyTorch compatibility

2. **Output Format**: Standardized using namedtuples
   - `FeatureAttrOutput`: Contains attributions and explainer-specific output
   - `GroupFeatureAttrOutput`: Extends with group masks and group attributions

3. **Modality Support**: Each explainer typically has:
   - `{Method}ImageCls`: For image classification
   - `{Method}ImageSeg`: For image segmentation (often wraps classification version)
   - `{Method}TextCls`: For text classification (where applicable)

### Key Components

#### common.py
- **FamWrapper**: Model wrapper with pre/post processing capabilities
- **Seg2ClsWrapper**: Converts segmentation models to classification format
- **Minibatch utilities**: `get_explanations_in_minibatches` for efficient processing
- **Image/tensor conversion utilities**: Handles numpy/torch conversions
- **Patch segmentation functions**: For creating rectangular patches

## Feature Attribution Methods

### 1. LIME (lime.py)
**Implementation Details:**
- Uses external LIME library with custom PyTorch integration
- Supports both image and text modalities
- Image: Uses `lime_image.LimeImageExplainer`
- Text: Custom `LimeTextExplainer` with tokenizer support
- Features minibatch processing wrapper for text models

**Key Features:**
- Configurable number of samples (default: 500)
- Returns segment-based attributions
- Supports group attribution output
- Text version handles tokenization and special tokens

**Potential Issues:**
- Complex segment attribution aggregation logic
- Text implementation has incomplete group attribution support (commented out)

### 2. SHAP (shap.py)
**Implementation Details:**
- Wraps the SHAP library's Partition explainer
- Image: Uses masking approach with configurable mask value
- Text: Handles tokenization and special token removal
- Supports padding for text sequences

**Key Features:**
- Configurable mask values
- Handles multiple target classes
- Text version manages special tokens (CLS, SEP, etc.)
- Includes usage examples for BERT and RoBERTa

**Potential Issues:**
- Text padding logic is complex
- Special token handling varies by model type

### 3. MFABA (mfaba.py)
**Implementation Details:**
- Multiple variants: sharp, smooth, cos, norm
- Uses external saliency library implementation
- Text version requires projection layer and wrapped model
- Leverages minibatch processing

**Key Features:**
- Four different MFABA variants
- Supports gradient-based attribution
- Text version uses `ExtraDimModelWrapper`

**Architecture Note:**
- Relies heavily on external saliency library
- Text implementation more complex due to embedding handling

### 4. Integrated Gradients (intgrad.py)
**Implementation Details:**
- Classic implementation with configurable steps
- Supports custom baselines (default: zeros)
- Separate loss functions for classification and segmentation
- Text version supports mask combination for embeddings

**Key Features:**
- Configurable number of integration steps (default: 32)
- Progress bar support
- Minibatch processing for efficiency
- Text version handles embedding-level gradients

**Clean Implementation:**
- Well-documented gradient integration logic
- Clear separation of concerns

### 5. GradCAM (gradcam.py)
**Implementation Details:**
- Wraps pytorch-grad-cam library
- Requires target layers specification
- Custom model wrapper to ensure gradients
- Text version uses specialized GradCAMText

**Key Features:**
- Multiple GradCAM variants available
- Reshape transform for vision transformers
- Target function customization

**Complexity:**
- Requires understanding of model architecture for target layers
- Text implementation is more involved

### 6. Other Methods

**RISE (rise.py)**
- Randomized Input Sampling for Explanation
- Generates random masks for attribution
- Configurable mask parameters (N, s, p1)

**Archipelago (archipelago.py)**
- Uses segmentation-based approach
- Supports multiple segmenters (quickshift, etc.)
- Top-k feature selection

**Additional Methods:**
- FullGrad (fullgrad.py)
- Attention-based (attn.py)
- PatchIntGrad (patch_intgrad.py)
- AGI, AMPE, IDG in libs/

## Code Quality Observations

### Strengths
1. **Consistent Interface**: All methods follow same base pattern
2. **Modular Design**: Clear separation between methods
3. **Efficiency**: Minibatch processing utilities
4. **Documentation**: Most methods have docstrings and usage examples

### Areas for Improvement
1. **Code Duplication**: Similar patterns repeated across methods
2. **Error Handling**: Limited error checking in some methods
3. **Type Hints**: Inconsistent or missing type annotations
4. **Testing**: No visible test files in the explainers directory
5. **Dependencies**: Heavy reliance on external libraries with varying APIs

## Refactoring Recommendations

### High Priority
1. **Standardize Minibatch Processing**: Create unified minibatch handler
2. **Abstract Common Patterns**: Extract repeated code patterns
3. **Improve Type Safety**: Add comprehensive type hints
4. **Error Handling**: Add validation and error messages

### Medium Priority
1. **Consolidate Utilities**: Merge similar utility functions
2. **Standardize Text Handling**: Unified tokenization interface
3. **Configuration Management**: Centralized default parameters
4. **Documentation**: Expand docstrings with examples

### Low Priority
1. **Performance Optimization**: Profile and optimize bottlenecks
2. **Async Support**: Consider async variants for I/O operations
3. **Caching**: Add attribution caching for repeated explanations

## Technical Debt

1. **Commented Code**: Several files contain commented implementations
2. **TODO Comments**: Unresolved TODOs (e.g., gradcam.py:1)
3. **Warning Suppressions**: archipelago.py suppresses all warnings
4. **Inconsistent Conventions**: Variable naming and structure varies

## Dependencies

### External Libraries
- lime
- shap
- pytorch-grad-cam
- scikit-image
- Custom libs in `libs/` directory

### Internal Dependencies
- Heavy coupling with model wrappers
- Tokenizer dependencies for text methods
- Device management complexity

## Conclusion

The explainers module provides a comprehensive suite of feature attribution methods with good architectural foundations. However, there's significant opportunity for refactoring to reduce code duplication, improve maintainability, and enhance type safety. The modular design makes it feasible to refactor incrementally without breaking existing functionality.
