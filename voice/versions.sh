# Pins every input of the voice build (noise removal and Enhance voice). Sourced by build.sh and release.sh.
# Bump REVISION whenever the outputs change (a new checkpoint or export).

# The release tag is voice-r<REVISION>.
REVISION=1

# Smule Renaissance Small (MIT), at one revision of its Hugging Face repo: Enhance voice.
SRS_REPO=smulelabs/Smule-Renaissance-Small
SRS_REVISION=12024e3250ace99e566662a2963e6158d104e69c
SRS_FILES="
model.py fa7e05ddbbae180de96142fe70f7815ee0b322a2c03fdc0d73fc60e91884ffa2
smule-renaissance-small.pt dd73b487ed058fdea66a25915f9f5a50b3d2dd97c03480f268a1d50a00fe2b06
LICENSE 137fc38400dcf00d75119c9a03029f8901fbf86f11bdc3153bcf44d61dc6c9d3
"

# Ceva's DPDFNet-2 for 48 kHz (Apache-2.0), at one revision of its Hugging Face repo: noise removal.
DPDFNET_REPO=Ceva-IP/DPDFNet
DPDFNET_REVISION=dd6818d00f50c836fed43a6243ebe49116de5964
DPDFNET_FILE=onnx/dpdfnet2_48khz_hr.onnx
DPDFNET_SHA256=7f0575a5cec0ba4ffd8f8bd657e06d007e4ccdd955d76faab922b9d3291dc14b
