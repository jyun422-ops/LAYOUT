import streamlit as st
import ezdxf
from ezdxf import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import unary_union
import tempfile
import os
import io

# --- [1] 핵심 알고리즘 함수 (이전과 동일) ---
def find_best_interlock(part, bridge):
    part_b_rotated = rotate(part, 180, origin='centroid')
    buffered_a = part.buffer(bridge, resolution=4)
    minx, miny, maxx, maxy = part.bounds
    w, h = maxx - minx, maxy - miny
    
    best_pair_geom, best_part_a, best_part_b = None, part, None
    min_box_area = float('inf')
    
    for dy in np.linspace(-h*0.8, h*0.8, 30): # 웹 구동 속도를 위해 스캔 횟수 30으로 약간 최적화
        dx, step = w * 1.5, w / 20 
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += step; break
            dx -= step
        fine_step = step / 10
        while dx > -w:
            test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
            if buffered_a.intersects(test_b): dx += fine_step; break
            dx -= fine_step
            
        test_b = translate(part_b_rotated, xoff=dx, yoff=dy)
        try:
            pair = unary_union([part, test_b])
            p_minx, p_miny, p_maxx, p_maxy = pair.bounds
            box_area = (p_maxx - p_minx) * (p_maxy - p_miny)
            if box_area < min_box_area:
                min_box_area, best_pair_geom, best_part_b = box_area, pair, test_b
        except: continue
    return best_part_a, best_part_b, best_pair_geom

# --- [2] 웹사이트 화면 구성 (Streamlit UI) ---
st.set_page_config(page_title="프레스 레이아웃 최적화기", layout="wide")
st.title("⚙️ 프레스 금형 스트립 레이아웃 최적화 & 원가 산출기")
st.markdown("도면을 업로드하면 최소 스크랩과 최저 원가를 자동으로 계산합니다.")

# 사이드바 (설계 조건 입력창)
st.sidebar.header("📝 설계 조건 및 단가 입력")
material_name = st.sidebar.text_input("소재 종류", "SPCC")
material_thickness = st.sidebar.number_input("소재 두께 (t)", value=1.2, step=0.1)
material_price = st.sidebar.number_input("단가 (원/kg)", value=1200, step=50)
material_density = st.sidebar.number_input("비중", value=7.85, step=0.01)
bridge = st.sidebar.number_input("최소 브릿지 (mm)", value=1.5, step=0.1)
margin = st.sidebar.number_input("최소 마진 (mm)", value=2.0, step=0.1)

# 메인 화면 (파일 업로드)
uploaded_file = st.file_uploader("DXF 전개도면을 이곳에 드래그 앤 드롭 하세요.", type=['dxf'])

if uploaded_file is not None:
    with st.spinner('도면을 분석하고 최적 배열을 계산 중입니다... (약 10초 소요)'):
        # 업로드된 파일을 임시 저장하여 ezdxf로 읽기
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc = ezdxf.readfile(tmp_path)
        msp = doc.modelspace()
        os.remove(tmp_path) # 임시 파일 삭제

        part_coords = []
        for entity in msp.query('LWPOLYLINE'):
            try:
                p = path.make_path(entity)
                part_coords = [(v.x, v.y) for v in p.flattening(distance=0.1)]
                break 
            except: continue

        if not part_coords:
            st.error("❌ 도면에서 다각형 폴리라인을 찾을 수 없습니다. CAD에서 PEDIT으로 외곽선을 결합해 주세요.")
        else:
            raw_part = Polygon(part_coords)
            part = raw_part.buffer(0)
            if part.geom_type == 'MultiPolygon': part = max(part.geoms, key=lambda a: a.area)

            part_area = part.area
            pair_area = part_area * 2 
            
            # 단일 배열 시뮬레이션
            single_results = []
            best_s_util, best_s_cost, best_s_angle, best_s_part = 0, 0, 0, None
            best_s_w, best_s_p = 0, 0
            
            for angle in range(0, 180, 10):
                rot = rotate(part, angle, origin='center')
                minx, miny, maxx, maxy = rot.bounds
                p_val, w_val = (maxx - minx) + bridge, (maxy - miny) + (margin * 2)
                util = (part_area / (p_val * w_val)) * 100
                cost = (((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price
                
                single_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,1), '소재폭(mm)': round(w_val,1), '이용률(%)': round(util,2), '1개당 원가(원)': int(cost)})
                if util > best_s_util: best_s_util, best_s_cost, best_s_angle, best_s_part, best_s_w, best_s_p = util, cost, angle, rot, w_val, p_val

            # 교차 배열 시뮬레이션
            part_a, part_b, pair_geom = find_best_interlock(part, bridge)
            inter_results = []
            best_i_util, best_i_cost, best_i_angle, best_i_pair = 0, 0, 0, None
            best_i_w, best_i_p = 0, 0

            if pair_geom:
                for angle in range(0, 180, 10):
                    rot = rotate(pair_geom, angle, origin='center')
                    minx, miny, maxx, maxy = rot.bounds
                    p_val, w_val = (maxx - minx) + bridge, (maxy - miny) + (margin * 2)
                    util = (pair_area / (p_val * w_val)) * 100
                    cost = ((((p_val * w_val * material_thickness) * material_density) / 1000000) * material_price) / 2
                    
                    inter_results.append({'각도': f"{angle}°", '피치(mm)': round(p_val,1), '소재폭(mm)': round(w_val,1), '이용률(%)': round(util,2), '1개당 원가(원)': int(cost)})
                    if util > best_i_util: best_i_util, best_i_cost, best_i_angle, best_i_pair, best_i_w, best_i_p = util, cost, angle, rot, w_val, p_val

            # --- [3] 결과 출력 (웹 화면에 그리기) ---
            st.success(f"✅ 분석 완료! 180도 교차 배열 적용 시 1개당 **{int(best_s_cost - best_i_cost):,}원**을 절감할 수 있습니다.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"[1] 단일 배열 (최적 각도: {best_s_angle}°)")
                st.info(f"소재 이용률: **{best_s_util:.1f}%** | 1개당 단가: **{int(best_s_cost):,}원**")
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                ax1.plot(*best_s_part.exterior.xy, color='#004b87'); ax1.fill(*best_s_part.exterior.xy, alpha=0.5, color='#004b87')
                minx, miny, maxx, maxy = best_s_part.bounds
                ax1.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], 'r--')
                ax1.axis('equal'); ax1.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig1)
                st.dataframe(pd.DataFrame(single_results))

            with col2:
                st.subheader(f"[2] 180도 교차 배열 (최적 각도: {best_i_angle}°)")
                st.info(f"소재 이용률: **{best_i_util:.1f}%** | 1개당 단가: **{int(best_i_cost):,}원**")
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                rot_a = rotate(part_a, best_i_angle, origin=pair_geom.centroid)
                rot_b = rotate(part_b, best_i_angle, origin=pair_geom.centroid)
                ax2.plot(*rot_a.exterior.xy, color='#004b87'); ax2.fill(*rot_a.exterior.xy, alpha=0.5, color='#004b87')
                ax2.plot(*rot_b.exterior.xy, color='#007934'); ax2.fill(*rot_b.exterior.xy, alpha=0.5, color='#007934')
                minx, miny, maxx, maxy = best_i_pair.bounds
                ax2.plot([minx, maxx, maxx, minx, minx], [miny, miny, maxy, maxy, miny], 'r--')
                ax2.axis('equal'); ax2.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig2)
                st.dataframe(pd.DataFrame(inter_results))