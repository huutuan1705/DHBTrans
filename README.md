Tệp requirements.txt bao gồm tất cả các phụ thuộc cần thiết cho dự án này (torch, pandas, numpy, scikit-learn, tqdm, kaggle và kagglehub).

Để cài đặt các phụ thuộc, hãy chạy:
```bash
pip install -r requirements.txt
```

Các thành phần tùy chỉnh được xây dựng từ đầu:

PositionalEncoding: Thêm thông tin vị trí vào nhúng mã thông báo
MultiHeadAttention: Triển khai cơ chế chú ý để tập trung vào các phần khác nhau của văn bản
TransformerEncoderLayer: Kết hợp các lớp chú ý và lớp truyền tiếp
TextClassifier: Mô hình chính sử dụng kiến ​​trúc biến đổi để phân loại
Kiến trúc mô hình:

Mô hình không có thành phần được đào tạo trước, theo yêu cầu
Nó sử dụng một lớp nhúng để chuyển đổi mã thông báo thành vectơ
Áp dụng mã hóa vị trí để giới thiệu thông tin về thứ tự từ
Sử dụng nhiều lớp mã hóa biến đổi với sự chú ý của nhiều đầu
Cuối cùng, có một đầu phân loại để dự đoán 4 lớp tình cảm
Quy trình đào tạo:

Triển khai vòng lặp đào tạo hoàn chỉnh với xác thực và thử nghiệm
Sử dụng trình tối ưu hóa AdamW và mất entropy chéo
Kết hợp lập lịch tốc độ học để hội tụ tốt hơn
Theo dõi và lưu mô hình tốt nhất dựa trên độ chính xác xác thực
Tính toán các số liệu chi tiết (độ chính xác, độ chính xác, khả năng thu hồi, điểm F1)
Chức năng dự đoán:

Bạn có thể sử dụng mô hình cho cả dự đoán theo văn bản đơn và theo batch
Tập lệnh dự đoán hỗ trợ cả đầu vào dòng lệnh và xử lý tệp CSV
Nó cung cấp điểm tin cậy cho tất cả các lớp tình cảm
Để đào tạo mô hình của bạn, bạn sẽ chạy:
```bash
python train.py
```
Điều này sẽ đào tạo mô hình biến đổi trên dữ liệu của bạn và lưu mô hình tốt nhất vào ```models/best_model.pth```.

Để đưa ra dự đoán với mô hình đã đào tạo:
```bash
# For single text prediction:
python predict.py --text "I absolutely love this product!"

# For batch predictions from a CSV file:
python predict.py --file path/to/data.csv --output predictions.csv
```
