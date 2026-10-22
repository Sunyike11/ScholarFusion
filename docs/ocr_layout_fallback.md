# OCR/Layout Fallback

The current OCR/Layout path is a schema-compatible fallback boundary:

- `NativePDFParser`
- `PageQualityGate`
- `OCRLayoutFallbackParser`
- `HybridPaperParser`

Low-quality pages trigger fallback elements with `parser_name=ocr_layout_fallback_v1` and low confidence.

This is not a production OCR benchmark implementation. It is designed so MinerU, PaddleOCR or VL structured page parsing can be added behind the same interface.
