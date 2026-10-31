import io
import re
import json
import hashlib
import streamlit as st
import numpy as np
import cv2
import fitz
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title='TopoAI — анализ топосъёмки', layout='wide')
st.title('TopoAI — анализ топографических планов')
st.warning('Предварительные измерения по изображению не заменяют координатный расчёт площади и геодезическую проверку.')

@st.cache_data(show_spinner=False)
def load_page(data, filename, page_num, dpi):
    if filename.lower().endswith('.pdf'):
        with fitz.open(stream=data, filetype='pdf') as doc:
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
            return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB')
    return Image.open(io.BytesIO(data)).convert('RGB')

def page_count(data, filename):
    if not filename.lower().endswith('.pdf'):
        return 1
    with fitz.open(stream=data, filetype='pdf') as doc:
        return len(doc)

def png_bytes(image):
    out = io.BytesIO()
    image.save(out, format='PNG')
    return out.getvalue()

def polygon_points(obj):
    # Fabric.js polygons use local points transformed by scale, rotation and position.
    if obj.get('type') != 'polygon' or not obj.get('points'):
        return None
    raw = np.asarray([[p['x'], p['y']] for p in obj['points']], dtype=float)
    if len(raw) < 3:
        return None
    width = float(obj.get('width') or (raw[:, 0].max() - raw[:, 0].min()))
    height = float(obj.get('height') or (raw[:, 1].max() - raw[:, 1].min()))
    offset = obj.get('pathOffset') or {'x': width / 2, 'y': height / 2}
    x = (raw[:, 0] - float(offset['x'])) * float(obj.get('scaleX', 1))
    y = (raw[:, 1] - float(offset['y'])) * float(obj.get('scaleY', 1))
    if obj.get('flipX'):
        x = -x
    if obj.get('flipY'):
        y = -y
    angle = np.deg2rad(float(obj.get('angle', 0)))
    c, s = np.cos(angle), np.sin(angle)
    x2 = x * c - y * s
    y2 = x * s + y * c
    # Fabric position is the transformed object's bounding box origin for standard polygon creation.
    x2 += float(obj.get('left', 0)) - x2.min()
    y2 += float(obj.get('top', 0)) - y2.min()
    return np.column_stack([x2, y2])

def shoelace(pts):
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2

uploaded = st.file_uploader('Загрузите топографический план', type=['pdf', 'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'])
if uploaded is not None:
    try:
        data = uploaded.getvalue()
        count = page_count(data, uploaded.name)
        a, b, c = st.columns(3)
        with a:
            page = st.number_input('Страница PDF', min_value=1, max_value=count, value=1, step=1)
        with b:
            dpi = st.slider('DPI PDF', 100, 350, 200, 50)
        with c:
            max_width = st.slider('Ширина холста, пиксели', 500, 1400, 1000, 100)
        original = load_page(data, uploaded.name, page - 1, dpi)
        factor = min(1, max_width / original.width)
        image = original.resize((max(1, round(original.width * factor)), max(1, round(original.height * factor))), Image.Resampling.LANCZOS)
        st.subheader('1. Обведите границу участка')
        st.caption('Режим Polygon: щёлкайте по вершинам границы; завершите двойным щелчком. Для исправления выберите Transform или перерисуйте контур.')
        mode = st.radio('Инструмент', ['polygon', 'transform'], format_func=lambda x: {'polygon': 'Рисовать многоугольник', 'transform': 'Выделить/переместить'}[x], horizontal=True)
        canvas_key = 'canvas_' + hashlib.md5(data + str(page).encode() + str(dpi).encode() + str(max_width).encode()).hexdigest()[:12]
        canvas = st_canvas(
            fill_color='rgba(0, 150, 255, 0.20)', stroke_width=2,
            stroke_color='#0096ff', background_image=image,
            update_streamlit=True, height=image.height, width=image.width,
            drawing_mode=mode, display_toolbar=True,
            return_image_data=True, key=canvas_key,
        )
        st.subheader('2. Калибровка')
        st.caption('Измерьте на этом отображаемом холсте отрезок с известной длиной (например, между узлами координатной сетки).')
        left, right = st.columns(2)
        with left:
            known_m = st.number_input('Реальная длина отрезка, м', min_value=0.001, value=10.0, format='%.3f')
        with right:
            known_px = st.number_input('Длина того же отрезка на холсте, пиксели', min_value=0.001, value=100.0, format='%.3f')
        m_per_px = known_m / known_px
        st.write(f'Коэффициент: **{m_per_px:.6f} м/пиксель**')
        objects = (canvas.json_data or {}).get('objects', [])
        polygons = [polygon_points(o) for o in objects if o.get('type') == 'polygon']
        polygons = [p for p in polygons if p is not None]
        st.subheader('3. Площадь участка')
        if polygons:
            pts = polygons[-1]
            area_px = shoelace(pts)
            area_m2 = area_px * m_per_px ** 2
            col1, col2, col3 = st.columns(3)
            col1.metric('Площадь, м²', f'{area_m2:,.2f}')
            col2.metric('Площадь, га', f'{area_m2 / 10000:,.4f}')
            col3.metric('Площадь, пикс²', f'{area_px:,.1f}')
            st.caption('Расчёт по последнему нарисованному многоугольнику; границы должны быть замкнуты и не пересекаться.')
            result = {'file': uploaded.name, 'page': int(page), 'area_m2': round(area_m2, 3), 'area_ha': round(area_m2 / 10000, 6), 'meters_per_pixel': m_per_px, 'vertices_pixels': pts.round(2).tolist()}
            st.download_button('Скачать расчёт JSON', json.dumps(result, ensure_ascii=False, indent=2).encode('utf-8'), file_name='topoai_area.json', mime='application/json')
        else:
            st.info('Нарисуйте и завершите многоугольник, чтобы увидеть площадь.')
        st.subheader('4. Распознавание числовых отметок (OCR)')
        lang = st.selectbox('Язык распознавания', ['eng', 'rus', 'eng+rus'])
        if st.button('Распознать числа'):
            try:
                import pytesseract
                gray = cv2.cvtColor(np.asarray(original), cv2.COLOR_RGB2GRAY)
                processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                raw = pytesseract.image_to_data(processed, lang=lang, config='--psm 11', output_type=pytesseract.Output.DICT)
                found = []
                for i, value in enumerate(raw['text']):
                    value = value.strip().replace(',', '.')
                    try:
                        conf = float(raw['conf'][i])
                    except (ValueError, TypeError):
                        continue
                    if conf >= 25 and re.fullmatch(r'\d{1,4}(?:\.\d{1,3})?', value):
                        found.append({'text': value, 'confidence': round(conf, 1), 'x': raw['left'][i], 'y': raw['top'][i]})
                st.session_state['ocr_results'] = found
            except Exception as exc:
                st.error(f'OCR недоступен: {exc}. Для OCR нужен системный Tesseract и языковые пакеты.')
        if 'ocr_results' in st.session_state:
            df = pd.DataFrame(st.session_state['ocr_results'], columns=['text', 'confidence', 'x', 'y'])
            st.write(f'Найдено числовых подписей: {len(df)}. Не все числа являются высотами.')
            st.dataframe(df, use_container_width=True)
            st.download_button('Скачать CSV', df.to_csv(index=False).encode('utf-8-sig'), 'numbers.csv', 'text/csv')
        st.subheader('5. Скачать изображение')
        st.download_button('Скачать страницу PNG', png_bytes(original), 'topographic_page.png', 'image/png')
    except Exception as exc:
        st.error(f'Ошибка обработки: {exc}')
else:
    st.info('Выберите файл, чтобы начать.')
