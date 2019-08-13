-Bước 1: Phát hiện các keypoint (điểm chính) (Sử dụng phương pháp phát hiện góc Harris, vv) và trích xuất các mô tả bất biến cục bộ (SIFT, SURF, vv) từ hai hình ảnh
đầu vào.
-Bước 2: Match các mô tả giữa hai hình ảnh.
-Bước 3: Sử dụng thuật toán RANSAC để ước tính ma trận homography sử dụng các vectơ đặc trưng phù hợp của chúng ta.
-Bước 4: Áp dụng warping transformation bằng ma trận homography (H) thu được từ Bước 3 để stitching image.
+Hướng phát triển trong tương lai :
- Có thể thực hiện ghép ảnh pandorama 360 với số lượng ảnh nhiều hơn
- Rút trích thông tin đặc trưng của các key point với độ chi tiết nhiều hơn
- Phát triển với video pandorama 360
