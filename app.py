from flask import Flask, render_template, request, redirect, send_file, send_from_directory, url_for
import os
from utils.clustering3D import run_birch_clustering2
from utils.clustering import run_birch_clustering
app = Flask(__name__, static_folder='static')

# Thiết lập các folder lưu file upload và file kết quả (trong static)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = os.path.join(app.static_folder, 'cluster_output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cluster', methods=['POST'])
def cluster():
    file = request.files['file']
    clustering_option = request.form.get('clustering_option') 
    if file.filename.endswith('.csv'):
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        if clustering_option == '2D':
            result_path, plot_path, df_result ,summary_html= run_birch_clustering(filepath, RESULT_FOLDER, 5)
            num_clusters = df_result['Cluster'].nunique()
            cluster_counts = df_result['Cluster'].value_counts().to_dict()
            table_data = df_result.to_dict(orient='records')
            return render_template(
                'result_1.html',
                plot_url = 'cluster_output/cluster_plot.png',
                result_file=os.path.basename(result_path),
                num_clusters=num_clusters,
                cluster_counts=cluster_counts,
                table_data=table_data,
                summary_table=summary_html
        )
        
        elif clustering_option == '2D_3D':
            result_path_3d, df_3d, summary_html = run_birch_clustering2(filepath, RESULT_FOLDER, n_clusters=5)
            
            num_clusters = df_3d['Cluster'].nunique()
            cluster_counts = df_3d['Cluster'].value_counts().to_dict()
            table_data = df_3d.to_dict(orient='records')
            
            plot_url_3d = url_for('serve_cluster_output', filename='cluster_plot3D.png')
            
            return render_template(
                'result.html',
                plot_url_3d=plot_url_3d,
                num_clusters=num_clusters,
                cluster_counts=cluster_counts,
                table_data=table_data,
                summary_table=summary_html,
                result_file=os.path.basename(result_path_3d)
            )
    return redirect('/')

@app.route('/download/<filename>')
def download(filename):
    path = os.path.join(RESULT_FOLDER, filename)
    return send_file(path, as_attachment=True)

@app.route('/download_summary')
def download_summary():
    summary_csv_path = os.path.join(RESULT_FOLDER, 'cluster_details.csv')
    if os.path.exists(summary_csv_path):
        return send_file(
            summary_csv_path,
            as_attachment=True,
            download_name="cluster_details.csv",
            mimetype='text/csv; charset=utf-8'
        )
    else:
        return "File không tồn tại", 404

@app.route('/cluster_output/<path:filename>')
def serve_cluster_output(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
