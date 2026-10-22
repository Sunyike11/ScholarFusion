# Multimodal Retrieval

Supported query types:

- Text
- Image upload
- Text + Image pair
- Stored element ID

Supported targets:

- Text elements
- Figures
- Figure-text pairs
- Pages
- Paper-level summaries

The retrieval path combines lexical retrieval, text dense vectors, multimodal dense vectors, page vectors and paper-level vectors through Weighted RRF.

Text-image pair embeddings use `enable_fusion=true`; vectors are not manually added or averaged.

Current limitation: page visual retrieval uses page text proxy vectors. Rendering full page images into visual vectors is the next upgrade.
