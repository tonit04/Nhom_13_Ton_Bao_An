import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def run_birch_clustering(filepath, result_folder, n_clusters=5):
    df = pd.read_csv(filepath)
    
    # Làm sạch và chuẩn hóa dữ liệu
    df_clean = df.dropna()
    features = df_clean[['Income', 'SpendingScore']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # Phân cụm khách hàng 
    model = Birch(n_clusters=n_clusters, threshold=1)
    labels = model.fit_predict(scaled)
    df_clean['Cluster'] = labels

    result_path = os.path.join(result_folder, 'clustered_' + os.path.basename(filepath))
    df_clean.to_csv(result_path, index=False)

    # Trực quan hóa : Vẽ biểu đồ 2D 
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(df_clean['Cluster'].unique()):
        cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
        plt.scatter(cluster_data['Income'], cluster_data['SpendingScore'], label=f"Cụm {cluster_id}", s=50)

    centroids = df_clean.groupby('Cluster')[['Income', 'SpendingScore']].mean().values
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='black', label='Tâm cụm')

    plt.xlabel('Thu nhập (USD)')
    plt.ylabel('Điểm chi tiêu')
    plt.title('Phân cụm khách hàng bằng Birch')
    plt.legend()
    plot_path = os.path.join('static/cluster_output', 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()

    summary = df_clean.groupby('Cluster')[['Income', 'SpendingScore']].mean()
   

    def describe_cluster(row):
        income = row['Income']
        spending = row['SpendingScore']

        if income > 70000 and spending > 70:
            return "Thu nhập cao, chi tiêu cao. Khách VIP, cần ưu đãi đặc biệt."
        elif income > 70000 and spending < 40:
            return "Thu nhập cao, chi tiêu thấp. Cần chiến lược kích cầu tiêu dùng."
        elif income < 50000 and spending > 70:
            return "Thu nhập thấp, chi tiêu cao. Gợi ý sản phẩm giá phù hợp."
        elif income < 50000 and spending < 40:
            return "Thu nhập và chi tiêu đều thấp. Nên áp dụng khuyến mãi lớn."
        elif 50000 <= income <= 70000 and 40 <= spending <= 70:
            return "Thu nhập và chi tiêu trung bình. Có thể cá nhân hóa theo sở thích."
        elif 50000 <= income <= 70000 and spending > 70:
            return "Thu nhập trung bình chi tiêu cao: Có thể quảng bá gói giá trị tốt hoặc upsell."
        elif 50000 <= income <= 70000 and spending < 40:
            return "Thu nhập trung bình chi tiêu thấp: Cần gợi ý sản phẩm thiết yếu."
        elif income > 70000 and 40 <= spending <= 70:
            return "Thu nhập cao, chi tiêu trung bình: Khuyến nghị nâng cấp sản phẩm."
        elif income < 50000 and 40 <= spending <= 70:
            return "Thu nhập thấp, chi tiêu trung bình: Nên tập trung sản phẩm giá rẻ."
        else:
            return "Ngoài phạm vi. Cần phân tích thêm."

    summary["Mô tả"] = summary.apply(describe_cluster, axis=1)

    summary_path = os.path.join(result_folder, 'cluster_details.csv')
    summary.to_csv(summary_path, index=True, encoding='utf-8-sig')

    summary_html = summary.to_html(classes="table table-bordered", border=0)

    return result_path, plot_path, df_clean, summary_html
