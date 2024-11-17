import math
import csv

# 定義類別型資料的映射
CATEGORY_MAPPINGS = {
    'gender': {'Male': 0, 'Female': 1},
    'SeniorCitizen': {'No': 0, 'Yes': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
    'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
}

COLUMNS_TO_ENCODE = list(CATEGORY_MAPPINGS.keys())

def load_data(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = []
        for row in csv_reader:
            for column, mapping in CATEGORY_MAPPINGS.items():
                if column in row:
                    row[column] = mapping[row[column]]
            row['tenure'] = int(row['tenure'])
            row['MonthlyCharges'] = float(row['MonthlyCharges'])
            row['TotalCharges'] = float(row['TotalCharges'])
            data.append(row)
    return data

def load_labels(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        labels = [row[0] for row in csv_reader][1:]  # 跳過標題行
    return labels

def one_hot_encode(data, column):
    unique_values = set(row[column] for row in data)
    for row in data:
        for value in unique_values:
            row[f"{column}_{value}"] = 1 if row[column] == value else 0
        del row[column]

def z_score_normalize(data, columns):
    for column in columns:
        values = [row[column] for row in data]
        mean = sum(values) / len(values)
        std_dev = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
        for row in data:
            row[column] = (row[column] - mean) / std_dev

class KNN:
    def __init__(self, p=2):
        self.p = p
        self.train_data = []

    def fit(self, train_data, train_labels):
        self.train_data = train_data
        for i, row in enumerate(self.train_data):
            row['label'] = train_labels[i]

    def p_norm_distance(self, row1, row2):
        distance = 0.0
        for key in row1:
            if key != 'label':
                distance += abs(row1[key] - row2[key]) ** self.p
        return distance ** (1 / self.p)

    def get_neighbors(self, test_row, num_neighbors):
        distances = []
        for train_row in self.train_data:
            dist = self.p_norm_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = [distances[i][0] for i in range(num_neighbors)]
        return neighbors

    def weighted_vote(self, neighbors):
        class_votes = {}
        for neighbor, distance in neighbors:
            label = neighbor['label']
            weight = 1 / (distance + 1e-5)  # 加上小數以避免除以零
            if label in class_votes:
                class_votes[label] += weight
            else:
                class_votes[label] = weight
        return max(class_votes, key=class_votes.get)

    def predict(self, test_data, num_neighbors):
        predictions = []
        for row in test_data:
            neighbors = self.get_neighbors(row, num_neighbors)
            weighted_neighbors = [(neighbor, self.p_norm_distance(row, neighbor)) for neighbor in neighbors]
            prediction = self.weighted_vote(weighted_neighbors)
            predictions.append(prediction)
        return predictions

def predict_and_save_results(knn, dataset_name):
    test_file_path = f"{dataset_name}.csv"
    test_data = load_data(test_file_path)

    # One-Hot Encoding
    for column in COLUMNS_TO_ENCODE:
        one_hot_encode(test_data, column)

    # Z-score 標準化
    z_score_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
    z_score_normalize(test_data, z_score_columns)

    # 計算 num_neighbors 為訓練資料數量的平方根
    num_neighbors = int(math.sqrt(len(knn.train_data)))

    # 預測每一筆測試資料
    predictions = knn.predict(test_data, num_neighbors)

    # 輸出預測結果到新的 CSV 檔案
    output_file_path = f"{dataset_name}_pred.csv"
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Churn'])
        for prediction in predictions:
            writer.writerow([prediction])

    print(f'預測結果已輸出到 {output_file_path}')

def main():
    train_file_path = "train.csv"
    train_label_file_path = "train_gt.csv"

    train_data = load_data(train_file_path)
    train_labels = load_labels(train_label_file_path)

    # One-Hot Encoding
    for column in COLUMNS_TO_ENCODE:
        one_hot_encode(train_data, column)

    # Z-score 標準化
    z_score_columns = ['MonthlyCharges', 'TotalCharges', 'tenure']
    z_score_normalize(train_data, z_score_columns)

    # 初始化 KNN 模型並訓練
    knn = KNN()
    knn.fit(train_data, train_labels)

    # 處理 val 和 test 資料集
    for dataset_name in ['val', 'test']:
        predict_and_save_results(knn, dataset_name)

if __name__ == "__main__":
    main()