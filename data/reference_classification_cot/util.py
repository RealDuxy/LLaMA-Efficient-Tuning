import math


def percentile(data, percentile):
    size = len(data)
    return sorted(data)[int(math.ceil((size * percentile) / 100)) - 1]


def describe(data, percentiles=[25, 50, 75, 90]):
    count = len(data)
    mean = sum(data) / count
    variance = sum((x - mean) ** 2 for x in data) / count
    std_dev = math.sqrt(variance)
    min_val = min(data)
    max_val = max(data)

    # 计算自定义百分位数
    percentiles_data = {}
    sorted_data = sorted(data)
    for p in percentiles:
        if 0 < p < 100:
            k = f"{p}%"
            index = int(math.ceil((len(sorted_data) * p) / 100)) - 1
            percentiles_data[k] = sorted_data[index]

    description = {
        'count': count,
        'mean': round(mean,2),
        # 'std': std_dev,
        'min': min_val,
        'percentiles': percentiles_data,
        'max': max_val
    }

    return description
