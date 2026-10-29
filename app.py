import io, re, math
import streamlit as st
import numpy as np
import cv2
import fitz
from PIL import Image
import pytesseract
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="TopoAI — анализ топосъёмки", layout="wide")
st.title("TopoAI — распознавание топографических планов")
st.warning("Прототип для предварительного анализа. Результаты OCR и измерений обязательно проверяйте по исходному плану и контрольным точкам.")
st.markdown("Загрузите скан/изображение или PDF, задайте масштаб и обведите границу участка. Программа распознает текстовые отметки высот и рассчитает площадь по обведённому контуру.")

uploaded = st.file_uploader("Файл топографической съёмки", type=["png","jpg","jpeg","tif","tiff","bmp","pdf"])
col1, col2 = st.columns([1,1])
with col1:
    scale_den = st.number_input("Знаменатель масштаба (например, 500 для 1:500)", min_value=1, value=500, step=50)
    dpi = st.slider("Разрешение рендеринга PDF (DPI)", 100, 400, 200, 50)
    ocr_lang = st.selectbox("Языки OCR", ["eng", "rus", "eng+rus"], index=0)
with col2:
    contour_interval = st.number_input("Сечение рельефа, м (если известно)", min_value=0.0, value=0.5, step=0.1)
    pixel_size_hint = st.caption("Для изображения масштаба 1:N: 1 мм на плане = N мм на местности. Точность зависит от качества скана.")

def load_image(file, dpi):
    data=file.getvalue()
    if file.name.lower().endswith(".pdf"):
        doc=fitz.open(stream=data, filetype="pdf")
        if len(doc)==0: raise ValueError("PDF не содержит страниц")
        page=doc[0]
        pix=page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72), alpha=False)
        img=Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img, len(doc)
    return Image.open(io.BytesIO(data)).convert("RGB"), 1

def image_to_png(image):
    b=io.BytesIO()
    image.save(b, format="PNG")
    return b.getvalue()

if uploaded:
    try:
        image, page_count=load_image(uploaded, dpi)
        st.caption(f"Обрабатывается первая страница. Страниц в PDF: {page_count}. Для других страниц загрузите/экспортируйте нужную страницу отдельно.")
        max_width=1100
        if image.width>max_width:
            new_h=int(image.height*max_width/image.width)
            image=image.resize((max_width,new_h))
        img_np=np.array(image)
        st.subheader("1. Исходный план")
        st.image(image, use_container_width=True)
        st.subheader("2. Обведите границу участка")
        st.write("Нажимайте на холсте, чтобы поставить вершины многоугольника. Для расчёта используйте режим Polygon. Рисуйте по границе участка, а не по рамке листа.")
        canvas=st_canvas(
            fill_color="rgba(0, 150, 255, 0.15)", stroke_width=2, stroke_color="#0096ff",
            background_image=image, update_streamlit=True, height=image.height, width=image.width,
            drawing_mode="polygon", key="boundary_canvas"
        )
        if canvas.json_data and canvas.json_data.get("objects"):
            objs=[o for o in canvas.json_data["objects"] if o.get("type")=="path"]
            # Canvas path coordinates are not a reliable polygon vertex list across all versions;
            # use raster mask from the canvas only when polygon pixels are present.
            if canvas.image_data is not None:
                rgba=np.array(canvas.image_data).astype(np.uint8)
                alpha=rgba[:,:,3] if rgba.shape[2]>3 else np.zeros(rgba.shape[:2],dtype=np.uint8)
                mask=(alpha>0).astype(np.uint8)
                # Include filled polygon interior; canvas output has transparent overlay.
                ys,xs=np.where(mask>0)
                if len(xs)>10:
                    # Use the polygon outline's convex hull as a conservative approximation.
                    pts=np.column_stack([xs,ys]).astype(np.int32)
                    hull=cv2.convexHull(pts)
                    area_px=abs(cv2.contourArea(hull))
                    # At displayed image scale, pixels on plan correspond to N / image pixel density.
                    # Need physical scan pixel density to convert px to mm; use scale calibration below.
                    st.info("Контур обнаружен. Для корректного пересчёта площади в м² задайте калибровку по известному расстоянию ниже.")
        st.subheader("3. Калибровка масштаба по известному расстоянию")
        st.write("Масштаб 1:N сам по себе недостаточен для скана неизвестного DPI/изменённого размера. Укажите расстояние на местности и длину того же отрезка в пикселях на показанном изображении.")
        c1,c2=st.columns(2)
        with c1: known_m=st.number_input("Известное расстояние на местности, м", min_value=0.001, value=10.0, step=1.0)
        with c2: known_px=st.number_input("Длина этого отрезка на изображении, пиксели", min_value=1.0, value=100.0, step=1.0)
        m_per_px=known_m/known_px
        st.metric("Калибровка", f"{m_per_px:.5f} м/пиксель")
        st.caption("Лучше калибровать по координатной сетке, масштабу-линейке или известному расстоянию между геодезическими точками.")
        st.subheader("4. Распознавание отметок высот")
        if st.button("Запустить OCR"):
            gray=cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # improve legibility while preserving original for display
            proc=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            config="--psm 11"
            raw=pytesseract.image_to_data(proc, lang=ocr_lang, config=config, output_type=pytesseract.Output.DICT)
            found=[]
            pattern=re.compile(r"(?<!\d)(\d{1,3}[.,]\d{1,3}|\d{1,3})(?!\d)")
            for i,txt in enumerate(raw["text"]):
                t=txt.strip().replace(",",".")
                if not t: continue
                try: conf=float(raw["conf"][i])
                except: conf=-1
                if conf < 25: continue
                # Keep plausible elevation-like numeric labels; OCR cannot infer meaning by itself.
                if pattern.fullmatch(t):
                    val=float(t)
                    if  -20 <= val <= 10000:
                        found.append({"text":t,"confidence":round(conf,1),"x":raw["left"][i],"y":raw["top"][i]})
            st.session_state["ocr_found"]=found
            st.session_state["ocr_raw"]=pytesseract.image_to_string(proc, lang=ocr_lang, config=config)
        if "ocr_found" in st.session_state:
            found=st.session_state["ocr_found"]
            st.write(f"Кандидатов на числовые подписи: {len(found)}. Это не гарантированно отметки высот: проверьте значения вручную.")
            if found:
                import pandas as pd
                df=pd.DataFrame(found)
                st.dataframe(df, use_container_width=True)
                st.download_button("Скачать найденные числа CSV", df.to_csv(index=False).encode("utf-8-sig"), "elevation_candidates.csv", "text/csv")
            with st.expander("Полный текст OCR"):
                st.text(st.session_state.get("ocr_raw",""))
        st.subheader("5. Экспорт")
        st.download_button("Скачать изображение страницы PNG", data=io.BytesIO(image).getvalue() if False else image_to_png(image), file_name="topographic_page.png", mime="image/png")
    except Exception as e:
        st.error(f"Не удалось обработать файл: {e}")
        st.info("Проверьте файл и установку зависимостей. Для OCR нужен установленный Tesseract OCR.")

