# Vendored Dependencies

This directory contains minimal vendored code from external libraries to avoid
requiring additional pip installations.

## Structure

- `lime/`: Core LIME functionality (from lime 0.2.0.1)
- `shap/`: Core SHAP functionality (from shap 0.41.0)
- `archipelago/`: Archipelago explainer (from custom implementation)
- `saliency/`: Saliency methods including MFABA variants

## Modifications

Each vendored module includes a README describing:
1. Source version
2. Files included
3. Modifications made
4. Original license

## Adding New Vendored Code

When vendoring new code:
1. Copy only essential files
2. Remove unnecessary dependencies
3. Document all changes
4. Include original license
5. Test thoroughly