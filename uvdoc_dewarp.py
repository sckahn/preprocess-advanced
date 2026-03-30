#!/usr/bin/env python3
"""UVDoc 기반 문서 dewarp. venv_paddle에서 실행.
사용법: python3 uvdoc_dewarp.py input.png output.png
"""
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import sys
import cv2
import numpy as np

def dewarp(input_path, output_path):
    from paddlex import create_model
    model = create_model("UVDoc")

    img = cv2.imread(input_path)
    if img is None:
        print(f"읽기 실패: {input_path}", file=sys.stderr)
        sys.exit(1)

    result = list(model.predict(img))
    dewarped = result[0]["doctr_img"]

    cv2.imwrite(output_path, dewarped)
    print(f"OK {dewarped.shape[1]}x{dewarped.shape[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python3 uvdoc_dewarp.py input.png output.png")
        sys.exit(1)
    dewarp(sys.argv[1], sys.argv[2])
