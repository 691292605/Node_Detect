import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import csv
import math
from collections import defaultdict
import os

Image.MAX_IMAGE_PIXELS = None


# =========================================================
# 基础辅助函数
# =========================================================
def get_red_mask(img_hsv):
    m1 = cv2.inRange(img_hsv, np.array([0, 43, 46]), np.array([10, 255, 255]))
    m2 = cv2.inRange(img_hsv, np.array([156, 43, 46]), np.array([180, 255, 255]))
    return m1 + m2


def get_yellow_mask(img_hsv):
    return cv2.inRange(img_hsv, np.array([20, 90, 90]), np.array([35, 255, 255]))


def get_green_mask(img_hsv):
    return cv2.inRange(img_hsv, np.array([35, 43, 46]), np.array([85, 255, 255]))


def get_black_mask(img_hsv):
    return cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([180, 255, 120]))


def get_adaptive_params(img_shape):
    h, w = img_shape[:2]
    base_width = 1600.0
    scale = w / base_width
    area_scale = scale * scale

    def get_odd_kernel(val):
        k = int(val)
        if k < 3: return 3
        if k % 2 == 0: k += 1
        return k

    return {
        'scale': scale,
        'red_min_area': 150 * area_scale,
        'red_max_area': 50000 * area_scale,
        'yellow_min_area': 300 * area_scale,
        'yellow_max_area': 100000 * area_scale,
        'black_min_area': 150 * area_scale,
        'black_max_area': 20000 * area_scale,
        'dot_min_area': 5 * area_scale,
        'dot_max_area': 600 * area_scale,
        'neighbor_dist': 800 * scale,
        'max_gap_pixels': int(12 * scale),
        'kernel_small_size': get_odd_kernel(3 * scale),
        'kernel_bridge_size': get_odd_kernel(5 * scale),
        'kernel_line_fix_len': int(5 * scale),
        'hough_threshold': 10,
        'min_line_length': int(2 * scale),
        'max_line_gap': int(10 * scale),
        'connection_dist_threshold': 50 * scale,
        'node_margin': int(25 * scale)
    }


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        if self.parent[i] == i: return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False


def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    temp_img = img.copy()
    while not done:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if size - cv2.countNonZero(temp_img) == size: done = True
    return skel


def dist_point_to_point(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def dist_point_to_segment(p, s_start, s_end):
    x, y = p
    x1, y1 = s_start
    x2, y2 = s_end
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    if cross <= 0: return math.hypot(x - x1, y - y1)
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if cross >= d2: return math.hypot(x - x2, y - y2)
    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    return math.hypot(x - px, y - py)


# =========================================================
# 核心逻辑
# =========================================================
def analyze_and_scan_topology_final(img_bgr, page_id):
    print(f"\n[处理进度] Page {page_id}: 正在分析...")
    params = get_adaptive_params(img_bgr.shape)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    vis_debug = img_bgr.copy()

    # 【新增】专门用于分析绿色块识别的调试图
    vis_green_analysis = img_bgr.copy()

    nodes = []
    forbidden = []

    k_small = params['kernel_small_size']
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (k_small, k_small))
    k_bridge = params['kernel_bridge_size']
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (k_bridge, k_bridge))

    # --- 1. 节点识别 ---
    # 黄色
    mask_y = cv2.morphologyEx(get_yellow_mask(hsv), cv2.MORPH_CLOSE, kernel_bridge)
    cnts_y, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_y:
        if params['yellow_min_area'] < cv2.contourArea(c) < params['yellow_max_area']:
            x, y, w, h = cv2.boundingRect(c)
            if 0.2 < w / h < 5.0:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    nodes.append(
                        {"id": -1, "type": "yellow_box", "center": (cx, cy), "bbox": (x, y, w, h), "neighbors": set()})
                    forbidden.append((x, y, w, h))
                    cv2.rectangle(vis_debug, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # 红色
    mask_r = cv2.dilate(get_red_mask(hsv), kernel_small, iterations=1)
    cnts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_r:
        if params['red_min_area'] < cv2.contourArea(c) < params['red_max_area']:
            x, y, w, h = cv2.boundingRect(c)
            if 0.2 < w / h < 5.0:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if not any(fx < cx < fx + fw and fy < cy < fy + fh for fx, fy, fw, fh in forbidden):
                        nodes.append(
                            {"id": -1, "type": "red_box", "center": (cx, cy), "bbox": (x, y, w, h), "neighbors": set()})
                        forbidden.append((x, y, w, h))
                        cv2.rectangle(vis_debug, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # 黑色
    cnts_b, _ = cv2.findContours(get_black_mask(hsv), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_b:
        if params['black_min_area'] < cv2.contourArea(c) < params['black_max_area']:
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 < w / h < 2.5:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    if not any(fx < cx < fx + fw and fy < cy < fy + fh for fx, fy, fw, fh in forbidden):
                        nodes.append({"id": -1, "type": "black_box", "center": (cx, cy), "bbox": (x, y, w, h),
                                      "neighbors": set()})
                        forbidden.append((x, y, w, h))
                        cv2.rectangle(vis_debug, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # 【核心修改】绿色实心块 (Switch) 分析与识别
    # -------------------------------------------------------------
    mask_g_raw = get_green_mask(hsv)

    # === 1. 自适应参数计算 ===
    # 基础尺寸：基于图纸缩放比例
    scale = params['scale']
    area_scale = scale * scale

    # 定义标准开关的大致面积 (经验值：假设开关在标准图纸上大约是 20x15 像素)
    # 只有大于这个基准一定比例的才会被认为是带字的开关
    ref_switch_area = 250 * area_scale

    # --- 自适应核大小 ---
    # 愈合核：用于填补文字空洞。不能太大，否则会把两条靠近的线连成一个块
    # 设定为 scale 的 2 倍左右，最小为 3
    k_heal_val = int(1.8 * scale)
    if k_heal_val < 3: k_heal_val = 3
    if k_heal_val % 2 == 0: k_heal_val += 1

    # 切断核：用于切断导线。必须比愈合核大，否则线切不断
    # 设定为 scale 的 3.5 倍左右
    k_open_val = int(4.0 * scale)
    if k_open_val <= k_heal_val: k_open_val = k_heal_val + 2  # 强制比愈合核大
    if k_open_val % 2 == 0: k_open_val += 1

    print(f"  [Green Param] Scale:{scale:.2f}, Heal Kernel:{k_heal_val}, Open Kernel:{k_open_val}")

    # === 2. 形态学处理 ===

    kernel_heal = cv2.getStructuringElement(cv2.MORPH_RECT, (k_heal_val, k_heal_val))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open_val, k_open_val))

    # A. 闭运算 (Healing)：填补由于蓝色文字造成的空洞
    mask_g_healed = cv2.morphologyEx(mask_g_raw, cv2.MORPH_CLOSE, kernel_heal, iterations=2)

    # B. 开运算 (Opening)：切断细长的绿色连接线
    mask_g_clean = cv2.morphologyEx(mask_g_healed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # C. 轻微膨胀：恢复因为开运算被削弱的边缘
    mask_g = cv2.dilate(mask_g_clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    cnts_g, _ = cv2.findContours(mask_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  [Green Debug] Found {len(cnts_g)} candidates.")

    for c in cnts_g:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        ratio = w / h if h > 0 else 0

        # 调试绘制：所有轮廓画黄色
        cv2.drawContours(vis_green_analysis, [c], -1, (0, 255, 255), 1)

        # === 3. 双重筛选逻辑 (Dual Criteria) ===
        # 很多误识别是因为把“小噪点”或者“短线段”当成了被文字遮挡的绿块
        # 所以我们根据实心度(Extent)采用不同的严格程度

        is_valid = False

        # 情况 A: 这是一个非常完美的实心块 (Extent > 0.85)
        # 这种情况下，我们允许它面积稍微小一点，长宽比稍微宽容一点
        if extent > 0.85:
            # 最小面积：参考红色设备的 1/5 即可
            if area > params['red_min_area'] * 0.2:
                # 长宽比：开关一般不会特别细长，限制在 0.3 到 4 之间
                if 0.3 < ratio < 4.0:
                    is_valid = True

        # 情况 B: 这是一个破碎的/带文字的块 (Extent 较低 0.55 - 0.85)
        # 这种情况下，极有可能是误识别（比如密集的线条），所以条件必须严格！
        elif extent > 0.55:
            # 1. 面积必须够大：必须达到“标准开关面积”的 60% 以上
            # 这样可以过滤掉碎小的噪点
            if area > ref_switch_area * 0.6:
                # 2. 长宽比必须很标准：带字的开关通常是矩形
                # 过滤掉细长的线 (ratio < 5.0 太宽了，改为 3.0)
                if 0.5 < ratio < 3.0:
                    is_valid = True

        info_text = f"A:{area:.0f}, E:{extent:.2f}, R:{ratio:.1f}"

        if is_valid:
            # 满足条件的画绿框
            cv2.rectangle(vis_green_analysis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_green_analysis, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 添加到节点列表
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if not any(fx < cx < fx + fw and fy < cy < fy + fh for fx, fy, fw, fh in forbidden):
                    nodes.append({
                        "id": -1,
                        "type": "green_switch",
                        "center": (cx, cy),
                        "bbox": (x, y, w, h),
                        "neighbors": set()
                    })
                    forbidden.append((x, y, w, h))
                    cv2.rectangle(vis_debug, (x, y), (x + w, y + h), (0, 100, 0), -1)
        else:
            # 不满足条件的画红框（方便调试看为什么被过滤）
            cv2.rectangle(vis_green_analysis, (x, y), (x + w, y + h), (0, 0, 255), 1)
            # cv2.putText(vis_green_analysis, info_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 【New】保存绿色分析图
    cv2.imwrite(f"debug_{page_id}_green_analysis.jpg", vis_green_analysis)
    # 圆点
    mask_dot = cv2.dilate(
        cv2.erode(get_black_mask(hsv), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small))),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_small, k_small)), iterations=3)
    cnts_d, _ = cv2.findContours(mask_dot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_d:
        area = cv2.contourArea(c)
        if params['dot_min_area'] < area < params['dot_max_area']:
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                if not any(fx - 2 < cx < fx + fw + 2 and fy - 2 < cy < fy + fh + 2 for fx, fy, fw, fh in forbidden):
                    if cv2.arcLength(c, True) > 0 and 4 * np.pi * (area / cv2.arcLength(c, True) ** 2) > 0.6:
                        nodes.append(
                            {"id": -1, "type": "dot", "center": (cx, cy), "bbox": (x, y, w, h), "neighbors": set()})
                        cv2.rectangle(vis_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 【保留】保存节点识别总览图
    cv2.imwrite(f"debug_{page_id}_nodes.jpg", vis_debug)

    # 重新分配ID
    nodes.sort(key=lambda n: n['center'][1])
    node_map = {}
    for i, n in enumerate(nodes):
        n['id'] = i
        node_map[i] = n

    # --- 2. 线路掩码与挖空 ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask_raw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_raw = cv2.dilate(mask_raw, kernel_dilate, iterations=1)

    # 【已注释】减少输出
    # cv2.imwrite(f"debug_{page_id}_mask_raw.jpg", mask_raw)

    mask_cutout = mask_raw.copy()
    shrink = int(3 * params['scale'])
    if shrink < 1: shrink = 1

    for n in nodes:
        x, y, w, h = n['bbox']
        sx, sy = x + shrink, y + shrink
        sw, sh = w - 2 * shrink, h - 2 * shrink
        if sw > 0 and sh > 0:
            cv2.rectangle(mask_cutout, (sx, sy), (sx + sw, sy + sh), 0, -1)
        else:
            mask_cutout[y + h // 2, x + w // 2] = 0

    # 【已注释】减少输出
    # cv2.imwrite(f"debug_{page_id}_mask_cutout.jpg", mask_cutout)

    # --- 3. 骨架化与霍夫变换 ---
    kernel_line_fix = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_cutout = cv2.dilate(mask_cutout, kernel_line_fix, iterations=1)
    skeleton = skeletonize(mask_cutout)

    # 【已注释】减少输出
    # cv2.imwrite(f"debug_{page_id}_skeleton.jpg", skeleton)

    lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180,
                            threshold=params['hough_threshold'],
                            minLineLength=params['min_line_length'],
                            maxLineGap=params['max_line_gap'])

    vis_hough = img_bgr.copy()
    valid_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            valid_lines.append(((x1, y1), (x2, y2)))
            cv2.line(vis_hough, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 【已注释】减少输出
    # cv2.imwrite(f"debug_{page_id}_hough.jpg", vis_hough)

    # --- 4. 构建线网 ---
    uf = UnionFind(len(valid_lines))
    dt = params['connection_dist_threshold']
    for i in range(len(valid_lines)):
        ps1, pe1 = valid_lines[i]
        for j in range(i + 1, len(valid_lines)):
            ps2, pe2 = valid_lines[j]
            if (dist_point_to_point(ps1, ps2) < dt or dist_point_to_point(ps1, pe2) < dt or
                    dist_point_to_point(pe1, ps2) < dt or dist_point_to_point(pe1, pe2) < dt):
                uf.union(i, j)
            elif (dist_point_to_segment(ps1, ps2, pe2) < dt or dist_point_to_segment(pe1, ps2, pe2) < dt or
                  dist_point_to_segment(ps2, ps1, pe1) < dt or dist_point_to_segment(pe2, ps1, pe1) < dt):
                uf.union(i, j)

    # --- 5. 节点挂载 ---
    node_to_groupIDs = defaultdict(set)
    nm = params['node_margin']

    for n in nodes:
        x, y, w, h = n['bbox']
        # 扩大节点判定框，增加容错
        rect_x, rect_y = x - nm, y - nm
        rect_w, rect_h = w + 2 * nm, h + 2 * nm

        # 定义矩形框用于 clipLine (格式: x, y, w, h)
        clip_rect = (int(rect_x), int(rect_y), int(rect_w), int(rect_h))

        for i, ((lx1, ly1), (lx2, ly2)) in enumerate(valid_lines):
            p1 = (int(lx1), int(ly1))
            p2 = (int(lx2), int(ly2))

            # 方法 A: 传统的端点检测 (Endpoint Inside)
            # 检查 p1 是否在 rect 内
            p1_in = (rect_x <= lx1 <= rect_x + rect_w) and (rect_y <= ly1 <= rect_y + rect_h)
            # 检查 p2 是否在 rect 内
            p2_in = (rect_x <= lx2 <= rect_x + rect_w) and (rect_y <= ly2 <= rect_y + rect_h)

            # 方法 B: 穿透检测 (Intersection)
            # cv2.clipLine 返回 True 如果线段的任何部分在矩形内
            is_intersect, _, _ = cv2.clipLine(clip_rect, p1, p2)

            if p1_in or p2_in or is_intersect:
                group_id = uf.find(i)
                node_to_groupIDs[n['id']].add(group_id)

    # --- 6. 最终连接 (MST Logic 改进版：中心辐射优先) ---
    vis_final = img_bgr.copy()
    final_confirmed_edges = set()
    group_to_nodes_list = defaultdict(list)

    for nid, gids in node_to_groupIDs.items():
        curr_node_obj = node_map[nid]
        for gid in gids:
            group_to_nodes_list[gid].append(curr_node_obj)

    processed_pairs = set()

    for gid, g_nodes in group_to_nodes_list.items():
        if len(g_nodes) < 2: continue

        # 1. 寻找当前组里的“大佬” (Hub Node)
        # 假设面积最大的就是 Hub
        node_areas = []
        for n in g_nodes:
            _, _, w, h = n['bbox']
            node_areas.append(w * h)

        max_area = max(node_areas) if node_areas else 0

        # 定义 Hub 的门槛：面积接近最大值的都被认为是 Hub
        hubs = set()
        for idx, n in enumerate(g_nodes):
            _, _, w, h = n['bbox']
            if (w * h) > max_area * 0.8:
                hubs.add(n['id'])

        edges = []
        n_count = len(g_nodes)

        # 2. 生成带权重的边
        for i in range(n_count):
            for j in range(i + 1, n_count):
                na = g_nodes[i]
                nb = g_nodes[j]

                # 原始几何距离
                raw_dist = dist_point_to_point(na['center'], nb['center'])

                # 过滤掉太远的
                if raw_dist > params['neighbor_dist'] * 3.0:
                    continue

                # --- 【核心修改】 权重调整 ---
                is_a_hub = na['id'] in hubs
                is_b_hub = nb['id'] in hubs

                weighted_dist = raw_dist

                if is_a_hub and is_b_hub:
                    weighted_dist = raw_dist * 1.0
                elif is_a_hub or is_b_hub:
                    weighted_dist = raw_dist * 0.7
                else:
                    weighted_dist = raw_dist * 1.5
                edges.append((weighted_dist, i, j, raw_dist))
        edges.sort(key=lambda x: x[0])

        l_uf = UnionFind(n_count)

        for w_dist, i, j, r_dist in edges:
            if l_uf.union(i, j):
                node_a = g_nodes[i]
                node_b = g_nodes[j]
                pid = tuple(sorted((node_a['id'], node_b['id'])))
                final_confirmed_edges.add(pid)

                if pid not in processed_pairs:
                    # 画线
                    cv2.line(vis_final, node_a['center'], node_b['center'], (128, 0, 128), 2)
                    processed_pairs.add(pid)

    for n in nodes:
        col = (0, 255, 0)
        if n['type'] == 'red_box':
            col = (255, 0, 0)
        elif n['type'] == 'black_box':
            col = (0, 255, 255)
        elif n['type'] == 'yellow_box':
            col = (0, 165, 255)
        elif n['type'] == 'green_switch':  # 绿色开关的颜色
            col = (0, 128, 0)  # Dark Green
        cv2.circle(vis_final, n['center'], 5, col, -1)
        cv2.putText(vis_final, str(n['id']), (n['center'][0], n['center'][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

    print(f"   > 逻辑分析完成: 共确认 {len(final_confirmed_edges)} 条连接边。")
    # 【保留】保存最终结果图
    cv2.imwrite(f"result_page_{page_id}_final.jpg", vis_final)
    return nodes, final_confirmed_edges


def save_to_csv_final(all_data, csv_name="topology_results_final.csv"):
    print(f"Saving CSV to {csv_name}...")
    with open(csv_name, 'w', newline='', encoding='utf-8-sig') as f:
        # 单页模式，无需 Page 列
        writer = csv.DictWriter(f, fieldnames=['Node_ID', 'Type', 'Pos_X', 'Pos_Y', 'Neighbors'])
        writer.writeheader()

        for page_nodes, page_edges in all_data:
            neighbors_map = defaultdict(set)
            for id1, id2 in page_edges:
                neighbors_map[id1].add(id2)
                neighbors_map[id2].add(id1)

            for n in page_nodes:
                nid = n['id']
                nbs = sorted(list(neighbors_map[nid]))
                neighbors_str = ";".join(map(str, nbs))
                writer.writerow({
                    'Node_ID': nid,
                    'Type': n['type'],
                    'Pos_X': n['center'][0],
                    'Pos_Y': n['center'][1],
                    'Neighbors': neighbors_str
                })
    print(f"Done saving CSV.")
    return csv_name


# =========================================================
# CSV 完整性与连通性自检
# =========================================================
def check_csv_topology(csv_path):
    print("\n" + "=" * 50)
    print(f"正在对 CSV 文件进行连通性检查: {csv_path}")
    print("=" * 50)

    if not os.path.exists(csv_path):
        print(f"错误: 找不到文件 {csv_path}")
        return

    isolated_nodes = []
    total_nodes = 0

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_nodes += 1
            node_id = row['Node_ID']
            neighbors = row['Neighbors'].strip()

            if not neighbors:
                isolated_nodes.append(f"ID {node_id} ({row['Type']})")

    print(f"扫描完成。共检测 {total_nodes} 个节点。")

    if isolated_nodes:
        print(f"\n[提示] 发现 {len(isolated_nodes)} 个孤立节点（无连接）：")
        # 仅打印前10个，避免刷屏
        for iso in isolated_nodes[:10]:
            print(f"  - {iso}")
        if len(isolated_nodes) > 10:
            print(f"  ... 以及其他 {len(isolated_nodes) - 10} 个")
    else:
        print("\n所有节点均已连接。")


# =========================================================
# 主流程
# =========================================================
def process_pdf(pdf, poppler, pw):
    # 如果是单页PDF，这里只会循环一次
    imgs = convert_from_path(pdf, dpi=200, poppler_path=poppler, userpw=pw)
    res = []
    for i, img in enumerate(imgs):
        arr = np.array(img)
        cv_img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if len(arr.shape) == 3 else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        nodes, edges = analyze_and_scan_topology_final(cv_img, i + 1)
        res.append((nodes, edges))

    # 保存 CSV
    csv_file = save_to_csv_final(res)

    # 执行 CSV 自检
    check_csv_topology(csv_file)


if __name__ == "__main__":
    #my_pdf = r"C:\Users\69129\Desktop\python space\pythonProject1\dian\picture\767ce3f95546d1e3c0dbad925bc4b5ac_3bc5441cbd9497a620aa5f8a912d630d_8.pdf"
    my_pdf = r"C:\Users\69129\Desktop\python space\pythonProject1\dian\picture\138e68df2d6a8f47093efd4598912f4d_cf53f7b9ab7080425636e9c239100df0_8.pdf"
    #my_pdf = r"C:\Users\69129\Desktop\python space\pythonProject1\dian\picture\0aff7daebb20ca860422a631cefc8613_24fc8778dad47e8dfb3f5e8ac385ff83_8.pdf"
    my_poppler = r"C:\Users\69129\Desktop\python space\pythonProject1\dian\poppler-25.12.0\Library\bin"
    my_password = "Dian123456"

    process_pdf(my_pdf, my_poppler, my_password)
