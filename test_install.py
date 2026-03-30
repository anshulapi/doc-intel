import fitz
import easyocr
import cv2
import torch
from transformers import pipeline
from fastapi import FastAPI
import pydantic

print("PyMuPDF:", fitz.__version__)
print("EasyOCR: OK")
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("Transformers: OK")
print("FastAPI: OK")
print("Pydantic:", pydantic.__version__)
print("\n All dependencies installed correctly!")