h5import \
DataTrn.txt -dims 1000,12 -path T -type TEXTFP \
DataTrn.labels.txt -dims 1000 -path lb -type TEXTIN \
DataTrn.id.txt -dims 1000 -path id -type TEXTIN -size 32 \
-o Oil.h5
