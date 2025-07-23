import subprocess
from PIL import Image
from svgpathtools import svg2paths, wsvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from .base_agent import BaseAgent
import os

class VectorConversionAgent(BaseAgent):
    name = 'vector_conversion'
    capabilities = {'tasks': ['vector_conversion']}

    async def process(self, task: dict) -> dict:
        image_path = task.get('input')
        if not image_path:
            raise ValueError("No input image path")
        bmp_path = 'temp.bmp'
        svg_path = task.get('output_svg', 'output.svg')
        pdf_path = task.get('output_pdf', 'output.pdf')
        img = Image.open(image_path).convert('L')
        img.save(bmp_path)
        subprocess.check_call(['potrace', '--svg', bmp_path, '-o', svg_path])
        os.remove(bmp_path)
        # Optimize
        paths, attributes = svg2paths(svg_path)
        optimized_paths = paths  # Add real optimization here
        wsvg(optimized_paths, attributes=attributes, filename=svg_path)
        # To PDF
        drawing = svg2rlg(svg_path)
        renderPDF.drawToFile(drawing, pdf_path)
        task['svg'] = svg_path
        task['pdf'] = pdf_path
        return task

    async def initialize(self):
        pass

    async def cleanup(self):
        pass