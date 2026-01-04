import csv


def calculate_iou(box1, box2):
    """计算两个矩形框的 IoU"""
    # 解包两个框的坐标
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    # 计算交集的坐标
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)

    # 计算交集面积
    inter_width = max(0, inter_xmax - inter_xmin)
    inter_height = max(0, inter_ymax - inter_ymin)
    intersection_area = inter_width * inter_height

    # 计算并集面积
    box1_area = (xmax1 - xmin1) * (ymax1 - ymin1)
    box2_area = (xmax2 - xmin2) * (ymax2 - ymin2)
    union_area = box1_area + box2_area - intersection_area

    # 计算并返回 IoU
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def calculate_mse(box_pred, box_gt):
    mse = (
        (box_pred[0] - box_gt[0]) ** 2 +
        (box_pred[2] - box_gt[2]) ** 2 +
        (box_pred[1] - box_gt[1]) ** 2 +
        (box_pred[3] - box_gt[3]) ** 2
    ) / 4
    return mse

def calculate_average_iou(output_csv, manual_test_csv):
    output_data = {}
    manual_data = {}

    # 读取 output.csv 文件
    with open(output_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        print("输出CSV字段名：", reader.fieldnames)
        for row in reader:
            image_name = row['Image Name']
            padding = int(row['Padding'])
            if padding == 1:  # 只有 Padding 不等于 2 才进行计算
                output_data[image_name] = {
                    'padding': padding,
                    'box': [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                }

    # 读取 手动测试.csv 文件
    with open(manual_test_csv, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        print("输出CSV字段名：", reader.fieldnames)
        for row in reader:
            image_name = row['Image Name']
            padding = int(row['Padding'])
            if padding == 1:  # 只有 Padding 不等于 2 才进行计算
                manual_data[image_name] = {
                    'box': [float(row['xmin']), float(row['ymin']), float(row['xmax']), float(row['ymax'])]
                }

    # 计算 IoU 并求平均
    total_iou = 0.0
    count = 0
    goodcount = 0
    total_mse = 0
    min_mse = 9999

    for image_name, output_info in output_data.items():
        if image_name in manual_data:
            manual_box = manual_data[image_name]['box']
            output_box = output_info['box']

            # 计算 IoU
            iou = calculate_iou(output_box, manual_box)
            total_iou += iou
            count += 1
            # 此行用于调整结果为良好的iou阈值
            if iou > 0.93:
                goodcount += 1

            mse = calculate_mse(output_box, manual_box)
            total_mse += mse
            if mse < min_mse:
                min_mse = mse

    # 计算平均 IoU
    if count > 0:
        average_iou = total_iou / count
        print(f"Average IoU: {average_iou}")
        print(f"recall: {goodcount / count}")
        print(f"Mes: {total_mse / count}")
        print(f"min_Mes: {min_mse}")
    else:
        print("没有匹配的数据进行 IoU 计算。")


# 调用函数，传入两个 CSV 文件路径
output_csv = 'output.csv'
manual_test_csv = '手动测试.csv'
calculate_average_iou(output_csv, manual_test_csv)
