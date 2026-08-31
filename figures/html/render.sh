#!/bin/sh
# Render html/F1_method.html to out/png/F1_method.png with headless Chrome at 3x, then trim to the figure.
cd "$(dirname "$0")/.." || exit 1
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=3 --window-size=2300,900 \
  --screenshot="$PWD/out/png/F1_method.png" "file://$PWD/html/F1_method.html" 2>/dev/null
python3 - <<'PY'
from PIL import Image, ImageChops
p="out/png/F1_method.png"; im=Image.open(p).convert("RGB")
bg=Image.new("RGB",im.size,(255,255,255)); bb=ImageChops.difference(im,bg).getbbox()
m=90; box=(max(0,bb[0]-m),max(0,bb[1]-m),min(im.width,bb[2]+m),min(im.height,bb[3]+m))
im.crop(box).save(p); print(p, im.crop(box).size)
PY
