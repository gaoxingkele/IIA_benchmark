Official CelebA-HQ reconstruction input/access audit

The pinned official recipe requires 202599 unaligned original images, landmarks, 30 delta ZIPs, image_list.txt, Pillow3.1.1 and libjpeg8d. Existing aligned CelebA ZIP MD5 and 202599 image entries were verified; its first image is178x218 and it does not satisfy the unaligned-input requirement.

The official shared folder lists every required delta filename. Thirty payload probes redirect to Google sign-in. One transport error remains pending recheck. No completed dataset payload or reconstructed image was produced. Existing raw files, partial transfers and scientific workers were preserved.

The complete official file identities and expected local input paths are in the configuration: configs/acquisition/fm_celebahq_official_reconstruction_inputs.2026-10-15_v1.json

Official pinned recipe: https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/2504c3f3cb98ca58751610ad61fa1097313152bd/dataset_tool.py
Official delta folder: https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs?resourcekey=0-arAVTUfW9KRhN-irJchVKQ&usp=sharing
