import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
import matplotlib
matplotlib.use('Agg')  # Dùng backend không giao diện
import matplotlib.pyplot as plt
import os
def describe_cluster(row):
    income = row['Income']
    spending = row['SpendingScore']
    frequency = row['PurchaseFrequency']

    if frequency <= 10:
        freq_label = "ít"
    elif frequency <= 20:
        freq_label = "trung bình"
    else:
        freq_label = "nhiều"

    if income > 70000 and spending > 70:
        return f"Thu nhập cao, chi tiêu cao, tần suất mua hàng {freq_label}. Khách VIP, có khả năng chi tiêu mạnh, cần ưu đãi đặc biệt và dịch vụ chăm sóc cao cấp."
    elif income > 70000 and spending < 40:
        return f"Thu nhập cao, chi tiêu thấp, tần suất mua hàng {freq_label}. Có thể cần chiến lược khuyến khích chi tiêu thông qua chương trình ưu đãi hấp dẫn."
    elif income < 50000 and spending > 70:
        return f"Thu nhập thấp nhưng chi tiêu cao, tần suất mua hàng {freq_label}. Khách hàng có nhu cầu cao, có thể gợi ý sản phẩm giá phù hợp hoặc chương trình trả góp."
    elif income < 50000 and spending < 40:
        return f"Thu nhập và chi tiêu đều thấp, tần suất mua hàng {freq_label}. Nên tập trung vào các chương trình khuyến mãi lớn hoặc ưu đãi giá rẻ."
    elif 50000 <= income <= 70000 and 40 <= spending <= 70:
        return f"Thu nhập và chi tiêu trung bình, tần suất mua hàng {freq_label}. Có thể cá nhân hóa đề xuất sản phẩm dựa trên sở thích và hành vi mua hàng."
    elif 50000 <= income <= 70000 and spending > 70:
        return f"Thu nhập trung bình nhưng chi tiêu cao, tần suất mua hàng {freq_label}. Cơ hội upsell tốt với các gói giá trị cao hơn hoặc quảng bá sản phẩm cao cấp."
    elif 50000 <= income <= 70000 and spending < 40:
        return f"Thu nhập trung bình, chi tiêu thấp, tần suất mua hàng {freq_label}. Cần gợi ý sản phẩm thiết yếu có giá hợp lý."
    elif income > 70000 and 40 <= spending <= 70:
        return f"Thu nhập cao, chi tiêu trung bình, tần suất mua hàng {freq_label}. Khuyến nghị nâng cấp sản phẩm hoặc dịch vụ để tối ưu trải nghiệm khách hàng."
    elif income < 50000 and 40 <= spending <= 70:
        return f"Thu nhập thấp nhưng chi tiêu trung bình, tần suất mua hàng {freq_label}. Nên tập trung vào sản phẩm giá cả phải chăng với giá trị cao."
    else:
        return f"Thông tin không rõ ràng, cần phân tích thêm. Tần suất mua hàng {freq_label}."

def run_birch_clustering2(filepath, result_folder, n_clusters=5):
    df = pd.read_csv(filepath)
    df = df.dropna().reset_index(drop=True)
    
    features = df[['Income', 'SpendingScore', 'PurchaseFrequency']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    df['Income_scaled'] = scaled[:, 0]
    df['SpendingScore_scaled'] = scaled[:, 1]
    df['PurchaseFrequency_scaled'] = scaled[:, 2]    
    df['Cluster_Description'] = df.apply(describe_cluster, axis=1)

    model = Birch(n_clusters=n_clusters, threshold=1)
    labels = model.fit_predict(scaled)
    df['Cluster'] = labels


    result_path = os.path.join(result_folder, 'clustered_' + os.path.basename(filepath))
    df.to_csv(result_path, index=False, encoding='utf-8-sig')

    # Tạo bảng tổng hợp trung bình và mô tả phổ biến cho từng cụm
    summary = df.groupby('Cluster')[['Income', 'SpendingScore', 'PurchaseFrequency']].mean()
    descriptions = df.groupby('Cluster')['Cluster_Description'].agg(lambda x: x.value_counts().idxmax())
    summary['Mô tả'] = descriptions

    # Lưu bảng summary vào CSV
    summary_path = os.path.join(result_folder, 'cluster_details.csv')
    summary.to_csv(summary_path, index=True, encoding='utf-8-sig')

    # Vẽ biểu đồ 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        color_idx = cluster_id % len(colors)
        ax.scatter(cluster_data['Income_scaled'],
                   cluster_data['SpendingScore_scaled'],
                   cluster_data['PurchaseFrequency_scaled'],
                   label=f"Cụm {cluster_id}",
                   s=50,
                   color=colors[color_idx],
                   alpha=0.6)

    for cluster in sorted(df['Cluster'].unique()):
        cluster_mask = df['Cluster'] == cluster
        centroid = scaled[cluster_mask].mean(axis=0)
        color_idx = cluster % len(colors)
        ax.scatter(centroid[0], centroid[1], centroid[2],
                   marker='X', s=250,
                   color=colors[color_idx],
                   edgecolor='k', lw=2,
                   zorder=10,
                   label=f"Tâm cụm {cluster}")

    ax.set_xlabel('Income')
    ax.set_ylabel('SpendingScore')
    ax.set_zlabel('PurchaseFrequency')
    ax.set_title('Phân cụm khách hàng bằng BIRCH (3D)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))

    plot_path = os.path.join(result_folder, 'cluster_plot3D.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    summary_html = summary.to_html(classes="table table-bordered", border=0, justify='center')

    return result_path, df, summary_html
