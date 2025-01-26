#%%

import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

image_path = './pictures/scene13.png'  # 输入图像文件路径

def is_mostly_solid_color(image, threshold=30, resize_size=(10, 10)):
    """
    判断图片是否接近纯色
    :param image_path: 图片路径
    :param threshold: 颜色差异阈值，默认 30
    :param resize_size: 缩小后的图片尺寸，默认 (10, 10)
    :return: True 如果图片接近纯色，否则 False
    """
    # 缩小图片尺寸
    image = cv2.resize(image, resize_size)
    
    # 将图片从 BGR 转换为 RGB（可选，取决于你的需求）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 计算所有像素的平均颜色
    avg_color = np.mean(image, axis=(0, 1))
    
    # 计算每个像素与平均颜色的差异
    diff = np.abs(image - avg_color)
    
    # 检查差异是否在阈值范围内
    result = np.all(diff <= threshold)
    if result:
        return True
    else:
        return False

def scan_and_find_bounding_box(image, color_threshold=10):
    """
    按照上下左右扫描逼近的思路，找到图像中物体的边界框。
    前提是纯色背景，并且物体与背景的颜色差异较大，可以通过颜色标准差来判断。
    
    参数:
        image: 输入的灰度图像（单通道）。
        color_threshold: 颜色标准差的阈值，用于判断是否为背景。
        
    返回:
        bounding_box: 返回一个元组 (left, top, right, bottom)，表示边界框的坐标。
    """
    height, width = image.shape
    
    # 初始化边界
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1
    
    # 从上到下扫描，找到上边界
    for y in range(height):
        row_pixels = image[y, :]  # 获取当前行的像素
        if np.std(row_pixels) > color_threshold:  # 如果标准差大于阈值，说明不是背景
            top = y
            break
    
    # 从下到上扫描，找到下边界
    for y in range(height - 1, -1, -1):
        row_pixels = image[y, :]
        if np.std(row_pixels) > color_threshold:
            bottom = y
            break
    
    # 从左到右扫描，找到左边界
    for x in range(width):
        col_pixels = image[:, x]  # 获取当前列的像素
        if np.std(col_pixels) > color_threshold:
            left = x
            break
    
    # 从右到左扫描，找到右边界
    for x in range(width - 1, -1, -1):
        col_pixels = image[:, x]
        if np.std(col_pixels) > color_threshold:
            right = x
            break
    
    return (left, top, right, bottom)

# area: 中心范围分割，0.0 - 1.0
def split_image(image_path, rows, cols, area, output_dir):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print("无法读取图像文件，请检查路径是否正确。")
        return
    img_height, img_width = img.shape[:2]

    # 计算每个小块的宽度和高度
    tile_width = img_width / cols
    tile_height = img_height / rows

    # 确保输出目录存在
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        os.rmdir(output_dir)
    os.makedirs(output_dir)

    area /= 2.0

    # 分割图像并保存每个小块
    for i in range(rows):
        for j in range(cols):
            # 计算每个小块的边界
            center = ((j + 0.5) * tile_width, (i + 0.5) * tile_height)
            left = int(center[0] - tile_width * area)
            upper = int(center[1] - tile_height * area)
            right = int(center[0] + tile_width * area)
            lower = int(center[1] + tile_height * area)

            # 裁剪图像
            tile = img[upper:lower, left:right]

            gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)

            # 调用函数找到边界框
            left, top, right, bottom = scan_and_find_bounding_box(gray, color_threshold=10)

            if left >= right or top >= bottom:
                print(f"无法找到有效的边界框，跳过 tile_{i}_{j}。")
                continue

            if is_mostly_solid_color(tile):
                # print(f"tile_{i}_{j} 是纯色背景，跳过。")
                continue

            # 绘制边界框
            # cv2.rectangle(tile, (left, top), (right, bottom), (0, 255, 0), 2)

            output_path = os.path.join(output_dir, f'tile_{i}_{j}.png')
            cv2.imwrite(output_path, tile[top:bottom, left:right])

    print(f"图像已成功分割为 {rows}x{cols} 个小块，并保存到 {output_dir} 目录中。")

def compare_images_cosine(img1, img2, window_size=3):
    """
    改进的余弦相似度比较，允许图片在 window_size 范围内出现位移偏差。
    :param img1: 图片1 (numpy数组)
    :param img2: 图片2 (numpy数组)
    :param window_size: 滑动窗口大小（奇数，例如 3, 5, 7）
    :return: 最大余弦相似度
    """
    # 确保图片尺寸相同
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # 获取图片尺寸
    height, width = img1.shape[:2]
    
    # 初始化最大相似度
    max_similarity = -1
    
    # 滑动窗口范围
    offset = window_size // 2
    
    # 遍历滑动窗口
    for dx in range(-offset, offset + 1):
        for dy in range(-offset, offset + 1):
            # 对 img2 进行位移
            shifted_img2 = np.roll(img2, (dy, dx), axis=(0, 1))
            
            # 裁剪掉位移后超出边界的部分
            if dx > 0:
                shifted_img2[:, :dx] = 0
            elif dx < 0:
                shifted_img2[:, dx:] = 0
            if dy > 0:
                shifted_img2[:dy, :] = 0
            elif dy < 0:
                shifted_img2[dy:, :] = 0
            
            # 展平图片并计算余弦相似度
            img1_flat = img1.flatten().reshape(1, -1)
            shifted_img2_flat = shifted_img2.flatten().reshape(1, -1)
            similarity = cosine_similarity(img1_flat, shifted_img2_flat)[0][0]
            
            # 更新最大相似度
            if similarity > max_similarity:
                max_similarity = similarity
    
    return max_similarity

def is_same_picture(rol1, col1, rol2, col2, window_size=3):
    """
    判断两张图片是否相同，允许几个像素的位移偏差。
    :param rol1, col1: 图片1的坐标
    :param rol2, col2: 图片2的坐标
    :param window_size: 滑动窗口大小
    :return: (是否相同, 相似度)
    """
    img1_path = f'./output_tiles/tile_{rol1}_{col1}.png'
    img2_path = f'./output_tiles/tile_{rol2}_{col2}.png'
    
    # 读取图片并调整大小
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return None, None

    img1 = cv2.resize(img1, (64, 64))
    img2 = cv2.resize(img2, (64, 64))
    
    # 计算改进的余弦相似度
    similarity = compare_images_cosine(img1, img2, window_size)
    
    # 判断是否相同
    return similarity > 0.97, similarity


def board2img(board, highlight, dir_idx, size=64):
    """
    将board转换为图像，并高亮显示指定的坐标区域。

    参数:
    - board: 二维列表，表示board的内容。
    - highlight: 高亮坐标列表，每个坐标表示为 [i, j]，其中 i 是行索引，j 是列索引。
    - size: 每个小图像的尺寸（默认64x64）。
    """
    rows, cols = len(board), len(board[0])  # 获取board的行数和列数
    img = np.zeros((size*rows, size*cols, 3), dtype=np.uint8)  # 创建空白图像

    # 填充图像
    for i in range(rows):
        for j in range(cols):
            cluster = board[i][j]
            if cluster == -1:
                continue
            img_path = f'./cluster_imgs/{cluster}.png'  # 拼接图像路径
            img_tile = cv2.imread(img_path)  # 读取小图像
            if img_tile is not None:
                img_tile = cv2.resize(img_tile, (size, size))  # 调整小图像大小
                img[i*size:(i+1)*size, j*size:(j+1)*size] = img_tile  # 将小图像粘贴到大图像中

    # 高亮显示指定区域
    for idx, coord in enumerate(highlight):
        print(highlight)
        if len(coord) == 2:  # 确保坐标格式正确
            i, j = coord
            if 0 <= i < rows and 0 <= j < cols:  # 检查坐标是否在范围内
                # 计算高亮区域的边界
                x1, y1 = j * size, i * size
                x2, y2 = (j + 1) * size, (i + 1) * size
                # 绘制矩形框
                color = (0, 0, 255)  # 高亮颜色 (BGR格式，红色)
                if idx == 1:
                    color = (0, 255, 0)  # 绿色
                if idx == 0:
                    # 标注移动方向
                    cv2.putText(img, directions_name[dir_idx], (x1 + 20, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    pass
                thickness = 6  # 矩形框线条粗细
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            else:
                print(f"忽略无效坐标: {coord}")
        else:
            print(f"忽略无效坐标格式: {coord}")

    return img

# 消去两个方块的条件：
# 1. 一个方块的上下左右有相同的方块
# 2. 一个方块往上下左右移动后的位置的上下左右有相同的方块
#
# 移动的条件：
# 如果要将一个方块向一个方向移动x格距离，那么该方块在那个方向连续的几个方块的那个方向必须有x格空位置
#
# 推箱子规则：
# 将一个方块朝着一个方向移动x格距离，那么该方块该方向的相连的所有方块都需要移动x距离
directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
directions_name = ['U', 'D', 'L', 'R']
def game_start(matrix):
    row = len(matrix)
    column = len(matrix[0])
    total_steps = int(row * column / 2)
    current_steps = 0
    solution = []  # 用于保存解决方案的步骤

    visited = set()  # 用于保存已经访问过的状态

    def backtrack(matrix, steps):
        nonlocal current_steps, visited
        if steps >= total_steps:
            return True # 找到解决方案
        
        # 将当前状态转换为可哈希的类型（例如元组）
        state = tuple(map(tuple, matrix))

        if state in visited:
            return False  # 已经访问过，剪枝
        visited.add(state)

        for x in range(len(matrix)):
            for y in range(len(matrix[0])):
                if matrix[x][y] == -1:
                    continue
                for idx, direction in enumerate(directions):
                    block = matrix[x][y]
                    if block == -1:
                        break
                    distance = get_direction_distance(x, y, matrix, direction)

                    # 保存当前状态
                    board_copy = [row[:] for row in matrix]
                    same_block = try_move_block(matrix, x, y, distance[0], distance[1], direction)
                    if same_block is not None:
                        current_steps += 1
                        solution.append((board_copy, idx, [x, y], same_block))  # 记录步骤
                        print_matrix(matrix, current_steps, [x, y], same_block)
                        # cv2.imwrite(f'solution/board_{current_steps}.png', board2img(board_copy, [[x, y], same_block]))

                        # 递归尝试下一步
                        if backtrack(matrix, steps + 1):
                            return True

                        # 回溯：恢复状态
                        matrix = [row[:] for row in board_copy]  # 恢复矩阵状态
                        current_steps -= 1
                        solution.pop()

        return False  # 没有找到解决方案

    if backtrack(matrix, 0):
        print("找到解决方案！")
        return solution, True
    else:
        print("未找到解决方案。")
        return solution, False

def get_direction_distance(x, y, matrix, direction):
    end_point = find_end_point(matrix, x, y, direction)
    nearest_remote_point = find_nearest_remote_point(matrix, end_point[0], end_point[1], direction)
    return [nearest_remote_point[0] - end_point[0], nearest_remote_point[1] - end_point[1]]

def try_move_block(matrix, x, y, dx, dy, direction):
    block = matrix[x][y]

    if direction[0] == 0:
        to_x = False
        real_distance = dy
    else:
        to_x = True
        real_distance = dx

    if real_distance > 0:
        start = 0
        end = real_distance + 1
    else:
        start = real_distance
        end = 1

    for i in range(start, end):
        if to_x:
            move_x = i
            move_y = 0
        else:
            move_x = 0
            move_y = i
        same_block = find_same_block(matrix, x, y, x + move_x, y + move_y, block)
        blocked = is_blocked(matrix, x, y, same_block, direction)
        if blocked:
            continue

        if same_block is not None:
            matrix[same_block[0]][same_block[1]] = -1
            matrix[x][y] = -1
            move_block(matrix, x, y, move_x, move_y, direction)
            return same_block
    return None

def find_same_block(matrix, x, y, nx, ny, block):
    for direction in directions:
        dx = nx + direction[0]
        dy = ny + direction[1]

        while True:
            if dx == x and dy == y:
                break
            if is_valid(matrix, dx, dy):
                if matrix[dx][dy] == -1:
                    dx += direction[0]
                    dy += direction[1]
                elif matrix[dx][dy] == block:
                    return [dx, dy]
                else:
                    break
            else:
                break
    return None

def is_blocked(matrix, x, y, same_block, direction):
    if same_block is not None and (same_block[0] - x == 0 or same_block[1] - y == 0):
        nx = x
        ny = y
        while True:
            nx += direction[0]
            ny += direction[1]
            if nx == same_block[0] and ny == same_block[1]:
                break
            if is_valid(matrix, nx, ny):
                if matrix[nx][ny] == -1:
                    continue
                else:
                    return True
            else:
                break
    return False

def is_valid(matrix, dx, dy):
    return 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0])

def is_end(total_steps, current_steps):
    return current_steps >= total_steps

def move_block(matrix, x, y, dx, dy, direction):
    if dx == 0 and dy == 0:
        return
    end_point = find_end_point(matrix, x, y, direction)
    current_x = end_point[0]
    current_y = end_point[1]
    while current_x != x or current_y != y:
        matrix[current_x + dx][current_y + dy] = matrix[current_x][current_y]
        matrix[current_x][current_y] = -1
        current_x = current_x - direction[0]
        current_y = current_y - direction[1]

def find_end_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                end_point = [x, y]
                break
            else:
                x = dx
                y = dy
        else:
            end_point = [x, y]
            break
    return end_point

def find_nearest_remote_point(matrix, x, y, direction):
    while True:
        dx = x + direction[0]
        dy = y + direction[1]
        if 0 <= dx < len(matrix) and 0 <= dy < len(matrix[0]):
            if matrix[dx][dy] == -1:
                x = dx
                y = dy
            else:
                nearest_remote_point = [x, y]
                break
        else:
            nearest_remote_point = [x, y]
            break
    return nearest_remote_point

def print_matrix(matrix, current_steps, point1=None, point2=None):
    if point2 is not None and point2 is not None:
        print("Step: %s, Source Block: %s, Target Block: %s" %
              (current_steps, point1, point2))
    else:
        print("Step: %s " % current_steps)

# 示例用法
rows = 14  # 行数
cols = 10  # 列数
area = 0.82
output_dir = 'output_tiles'  # 输出目录

split_image(image_path, rows, cols, area, output_dir)

clusters = [[(0, 0)]]

print("开始进行图像识别...")
for i in tqdm.tqdm(range(rows)):
    for j in range(cols):
        img1_path = f'./output_tiles/tile_{i}_{j}.png'
        # 如果文件不存在，跳过
        if not os.path.exists(img1_path):
            continue
        if i == 0 and j == 0:
            continue
        for cluster in clusters:
            same, _ = is_same_picture(i, j, cluster[0][0], cluster[0][1])
            if same:
                cluster.append((i, j))
                break
        else:
            clusters.append([(i, j)])

print(clusters)
print("识别完毕，分类数：", len(clusters))

# 生成每一个 cluster 的图片

print("生成棋盘图标中...")
print("清空 cluster_imgs 文件夹...")
cluster_imgs_folder = 'cluster_imgs'
if os.path.exists(cluster_imgs_folder):
    for file in os.listdir(cluster_imgs_folder):
        os.remove(os.path.join(cluster_imgs_folder, file))
    os.rmdir(cluster_imgs_folder)
os.makedirs(cluster_imgs_folder)

for i, cluster in enumerate(clusters):
    rol=cluster[0][0]
    col=cluster[0][1]
    img_path = f'./output_tiles/tile_{rol}_{col}.png'
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    cv2.imwrite(f'{cluster_imgs_folder}/{i}.png', img)

#%%

# 创建初始棋盘
print("创建初始棋盘...")
board = [[-1] * cols for _ in range(rows)]
for i in range(len(clusters)):
    for x, y in clusters[i]:
        if os.path.exists(f'./output_tiles/tile_{x}_{y}.png'):
            board[x][y] = i

#%%

# 开始模拟游戏

# 清除solution文件夹
print("初始化 solution 文件夹...")
if os.path.exists('solution'):
    for file in os.listdir('solution'):
        os.remove(os.path.join('solution', file))
    os.rmdir('solution')
os.makedirs('solution')

print("开始模拟游戏...")
solution, success = game_start(board)
if not success: exit(0)

#%%

print("结果保存在 solution 文件夹中。")
for step, (board, dir_idx, source, target) in enumerate(solution):
    print(f"步骤 {step+1}: 将方块 {source} 移动到 {target}, 方向: {directions_name[dir_idx]}")
    img = board2img(board, [source, target], dir_idx)
    cv2.imwrite(f'solution/board_{step+1}.png', img)

# %%
