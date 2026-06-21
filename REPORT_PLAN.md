# Kế hoạch cấu trúc báo cáo CNN nhận dạng ký tự 16x16

> **Quy ước ngôn ngữ:** File kế hoạch này được viết bằng tiếng Việt để dễ theo dõi. Báo cáo chính thức trong `report.tex` phải viết **100% bằng tiếng Anh**, bao gồm trang bìa, abstract, tên chương/mục, nội dung, công thức diễn giải, caption hình, caption bảng và kết luận.

## 1. Phạm vi và tên đề tài đề xuất

**Tên chính thức đề xuất:** A From-Scratch Convolutional Neural Network for 16x16 Handwritten Character Recognition with User Feedback.

Phạm vi đúng với hệ thống hiện tại:

- Dataset chính: file EMNIST Balanced 47 lớp đã có sẵn ở định dạng ảnh nhị phân 16x16.
- Bài toán phân loại 47 lớp: 10 chữ số, 26 chữ hoa và 11 chữ thường `a, b, d, e, f, g, h, n, q, r, t`.
- CNN được tự cài đặt bằng NumPy, không dùng PyTorch, TensorFlow, mô hình pretrained hoặc lớp mạng có sẵn.
- Hệ thống có giao diện vẽ ký tự, dự đoán, xác nhận đúng/sai và lưu nhãn sửa vào file feedback.
- Mỗi lần chạy `train.py`, mô hình được huấn luyện lại từ đầu trên dữ liệu EMNIST và feedback. Cơ chế hiện tại là **retraining with feedback**, chưa phải fine-tuning từ trọng số cũ.

## 2. Những nội dung trong `report.tex` phải sửa

| Vị trí hiện tại | Vấn đề | Cách sửa |
|---|---|---|
| Trang bìa | Vẫn ghi đề tài Telecom Subscription Churn | Đổi sang tên đề tài OCR/CNN ở trên |
| Introduction | Một số đoạn nói dataset Unifont | Đổi nguồn dữ liệu chính thành file EMNIST Balanced 47 lớp đã có sẵn |
| Objectives | Đang đặt preprocessing và augmentation thành mục tiêu chính | Bỏ mục tiêu này vì `train.py` không thực hiện hai bước đó |
| Data preprocessing | Đang mô tả đọc `.hex`, resize, threshold và tạo biến thể | Bỏ khỏi phương pháp huấn luyện; chỉ mô tả cấu trúc file `.npz` được nạp trực tiếp |
| Data augmentation | `data_augmentation.py` tồn tại nhưng không được `train.py` gọi | Bỏ khỏi phần triển khai hiện tại; chỉ nhắc ngắn trong hướng phát triển |
| CNN architecture | LaTeX đang ghi convolution không padding, kích thước cuối 2x2x32 = 128 | Code hiện tại dùng padding 1, nên kích thước cuối là 4x4x32 = 512 |
| Dynamic adaptation | Đang tuyên bố fine-tuning và tránh catastrophic forgetting | Đổi thành huấn luyện lại có bổ sung feedback, trừ khi code được sửa để nạp model cũ và fine-tune |
| Experiments and Evaluation | Toàn bộ nội dung từ phần EDA trở đi vẫn là customer churn | Xóa và thay bằng chương thí nghiệm OCR trong kế hoạch này |
| Conclusions | Vẫn kết luận về churn, uplift, CLV và retention | Viết lại hoàn toàn theo kết quả OCR thực tế |
| Tài liệu tham khảo | Có `references.bib` trong LaTeX nhưng hiện chưa thấy file này trong thư mục | Tạo file tài liệu tham khảo hoặc bỏ lệnh BibLaTeX nếu chưa dùng |
| Hình ảnh | LaTeX gọi `hustss.png` nhưng hiện chưa thấy file trong thư mục | Thêm logo đúng tên hoặc sửa đường dẫn |
| Figure placement | Có nhiều hình dùng tùy chọn `[H]` | Thêm `\usepackage{float}` nếu tiếp tục dùng `[H]` |

Không nên ghi “toàn bộ hệ thống không dùng thư viện”. Cách diễn đạt chính xác là: **các phép toán forward, backward và cập nhật tham số của CNN được xây dựng từ đầu bằng NumPy**. Trong phạm vi huấn luyện và dự đoán, chương trình đọc trực tiếp dataset `.npz`, không gọi Scikit-learn hay OpenCV để tiền xử lý.

## 3. Cấu trúc báo cáo đề xuất

### Phần đầu báo cáo

1. Trang bìa.
2. Lời cảm ơn, nếu được yêu cầu.
3. Abstract bằng tiếng Anh.
4. Mục lục.
5. Danh mục hình.
6. Danh mục bảng.
7. Danh mục ký hiệu và từ viết tắt: CNN, OCR, ReLU, SGD, EMNIST, GUI.

Phần tóm tắt nên trình bày ngắn gọn: bài toán, dataset 47 lớp, kiến trúc CNN tự xây dựng, giới hạn khoảng 10.000 mẫu, phương pháp đánh giá và kết quả test tốt nhất. Feedback chỉ được nhắc như một chức năng thu thập dữ liệu, không kết luận về hiệu quả nếu không đánh giá.

### Chương 1. Giới thiệu

#### 1.1. Bối cảnh

- Giới thiệu OCR và nhận dạng chữ viết tay.
- Khó khăn khi ảnh chỉ có 16x16 pixel: mất nét cong, giao nhau giữa nét, các ký tự dễ giống nhau.
- Khó khăn khi phân biệt chữ hoa và chữ thường có hình dạng gần nhau.
- Khoảng cách giữa ảnh EMNIST và nét vẽ bằng chuột của người dùng.

#### 1.2. Bài toán

- Đầu vào: ma trận ảnh nhị phân `16 x 16`, giá trị pixel thuộc `{0, 255}`.
- Đầu ra: phân phối xác suất trên 47 lớp ký tự.
- Ràng buộc: xây dựng CNN từ đầu, chạy CPU, không dùng framework deep learning hoặc pretrained model.

#### 1.3. Mục tiêu

- Xây dựng đầy đủ Conv2D, ReLU, MaxPool, Flatten, Dense, Softmax và backpropagation.
- Đảm bảo dữ liệu giới hạn vẫn cân bằng giữa 47 lớp.
- Đánh giá độ chính xác tổng thể và theo từng lớp.
- Xây dựng giao diện dự đoán và thu thập sửa nhãn.

#### 1.4. Đóng góp

- CNN nhỏ phù hợp đầu vào 16x16.
- Pipeline dữ liệu cân bằng và stratified split.
- Cơ chế lưu feedback tách biệt và chỉ đưa feedback vào tập train.
- Giao diện minh họa toàn bộ luồng từ vẽ đến sửa nhãn.

#### 1.5. Cấu trúc báo cáo

Tóm tắt nội dung từng chương trong một đoạn ngắn.

### Chương 2. Cơ sở lý thuyết

#### 2.1. Biểu diễn ảnh grayscale và tensor

- Giải thích grayscale một kênh, khác ảnh RGB ba kênh.
- Ảnh trong dataset có shape `(N, 16, 16)`; một ảnh có shape `(16, 16)`.
- Conv2D tự thêm chiều channel để đưa ảnh thành `(16, 16, 1)`.
- Giải thích pixel 0 là nền đen, 255 là nét trắng.

#### 2.2. Phép tích chập

- Trình bày công thức convolution/cross-correlation dùng trong code.
- Giải thích kernel 3x3, stride 1, padding 1.
- Giải thích số filter 16 và 32.

#### 2.3. ReLU và Max Pooling

- Công thức ReLU và đạo hàm.
- MaxPool 2x2 giúp giảm kích thước không gian và giữ đặc trưng mạnh.

#### 2.4. Dense và Softmax

- Flatten đặc trưng 4x4x32 thành vector 512 phần tử.
- Dense 512 -> 128 -> 47.
- Softmax ổn định số bằng cách trừ logit lớn nhất trước khi lấy hàm mũ.

#### 2.5. Cross-Entropy và backpropagation

- Công thức categorical cross-entropy.
- Gradient kết hợp Softmax và cross-entropy: `p - y_one_hot`.
- Mô tả cập nhật gradient descent theo từng ảnh, tức batch size bằng 1.

#### 2.6. Khởi tạo He

- Conv: `sqrt(2 / (in_channels * kernel_height * kernel_width))`.
- Dense: `sqrt(2 / input_len)`.
- Giải thích tính phù hợp với ReLU.

#### 2.7. Các chỉ số đánh giá

- Accuracy.
- Precision, Recall và F1-score cho từng lớp.
- Macro-F1 cho toàn bộ 47 lớp.
- Confusion matrix.
- Train/validation loss để phát hiện underfitting hoặc overfitting.
- Thời gian huấn luyện và thời gian dự đoán trung bình.

### Chương 3. Dữ liệu sử dụng

#### 3.1. Dataset EMNIST Balanced 47 lớp

- Nêu nguồn và đặc điểm bộ EMNIST Balanced.
- File được sử dụng trực tiếp là `emnist_47_classes_16x16.npz`.
- Dataset có 47.000 ảnh và 47 lớp, đúng 1.000 ảnh/lớp.
- Nêu rõ chỉ có 11 chữ thường riêng biệt vì các chữ thường còn lại được EMNIST Balanced gộp với chữ hoa do hình dạng dễ nhầm.

#### 3.2. Ánh xạ nhãn

- Nhãn `0-9` được ánh xạ về mã Unicode 48-57.
- Nhãn `A-Z` được ánh xạ về 65-90.
- 11 nhãn chữ thường được ánh xạ riêng về mã Unicode tương ứng.

#### 3.3. Định dạng dữ liệu đầu vào

- `train.py` nạp trực tiếp hai mảng `X` và `y` từ file `.npz`.
- `X` có shape `(47000, 16, 16)`, kiểu `uint8`, giá trị pixel là 0 hoặc 255.
- `y` có shape `(47000,)`, lưu mã Unicode của nhãn.
- Không có resize, xoay, threshold hoặc augmentation trong quá trình huấn luyện hiện tại.
- Phép `(image / 255.0) - 0.5` thuộc bước forward của CNN để chuẩn hóa giá trị số, không phải quá trình tạo hay biến đổi dataset.

#### 3.4. Chọn tập con cân bằng

Với `MAX_SAMPLES = 10000` và 47 lớp:

```text
samples_per_class = floor(10000 / 47) = 212
total_samples = 212 x 47 = 9964
unused_budget = 10000 - 9964 = 36
```

36 vị trí dư bị bỏ để mọi lớp vẫn có số ảnh bằng nhau. Nếu một lớp có ít ảnh hơn giới hạn tính được, số ảnh của lớp ít nhất được dùng làm giới hạn chung cho tất cả lớp.

#### 3.5. Chia train, validation và test

Với 212 ảnh/lớp và tỷ lệ 80/10/10 hiện tại:

| Tập | Ảnh mỗi lớp | Tổng số ảnh |
|---|---:|---:|
| Train | 170 | 7.990 |
| Validation | 21 | 987 |
| Test | 21 | 987 |

Việc chia được thực hiện riêng trong từng lớp rồi mới shuffle, vì vậy ba tập vẫn cân bằng. Feedback chỉ được nối vào tập train và không được đưa vào validation/test.

#### 3.6. Phân tích dữ liệu ban đầu

- Kiểm tra shape, dtype, min/max pixel.
- Kiểm tra số lượng lớp và số ảnh mỗi lớp.
- Quan sát ảnh mẫu để kiểm tra chiều ảnh và chất lượng dữ liệu đầu vào.
- Thống kê tỷ lệ pixel foreground để phát hiện ảnh rỗng hoặc quá đặc.

### Chương 4. Phương pháp và cài đặt hệ thống

#### 4.1. Kiến trúc tổng thể

Trình bày luồng: nạp dataset `.npz` -> chọn mẫu cân bằng -> stratified split -> train CNN -> lưu trọng số -> GUI dự đoán -> người dùng sửa nhãn -> lưu feedback -> retrain.

#### 4.2. Kiến trúc CNN hiện tại

| Lớp | Kích thước đầu vào | Cấu hình | Kích thước đầu ra | Số tham số |
|---|---|---|---|---:|
| Input | 16x16 | Chuẩn hóa về `[-0.5, 0.5]` | 16x16x1 | 0 |
| Conv1 | 16x16x1 | 16 kernel 3x3, padding 1 | 16x16x16 | 160 |
| ReLU1 | 16x16x16 | ReLU | 16x16x16 | 0 |
| MaxPool1 | 16x16x16 | 2x2 | 8x8x16 | 0 |
| Conv2 | 8x8x16 | 32 kernel 3x3, padding 1 | 8x8x32 | 4.640 |
| ReLU2 | 8x8x32 | ReLU | 8x8x32 | 0 |
| MaxPool2 | 8x8x32 | 2x2 | 4x4x32 | 0 |
| Flatten | 4x4x32 | Trải phẳng | 512 | 0 |
| Dense1 | 512 | 128 neuron + ReLU | 128 | 65.664 |
| Dense2 | 128 | 47 neuron | 47 | 6.063 |
| Softmax | 47 | Xác suất 47 lớp | 47 | 0 |
| **Tổng** | | | | **76.527** |

#### 4.3. Forward propagation

Mô tả thứ tự chạy qua từng layer và shape thay đổi. Chèn pseudocode ngắn, không cần dán toàn bộ mã nguồn.

#### 4.4. Backpropagation và cập nhật tham số

Mô tả gradient đi theo chiều ngược qua Softmax, Dense, Flatten, MaxPool, ReLU và Conv2D. Nêu learning rate mặc định `0.008` và cập nhật sau từng ảnh.

#### 4.5. Lưu và tải model

- Lưu filters, biases, dense weights và danh sách class labels vào `.npz`.
- Kiểm tra nhãn trong model phải khớp đúng 47 nhãn của dataset trước khi mở GUI.

#### 4.6. Giao diện và phản hồi người dùng

- Canvas 16x16.
- Chuột trái để vẽ, chuột phải để xóa.
- Crop vùng có nét và căn giữa trước khi dự đoán.
- Hiển thị nhãn và confidence.
- Nếu sai, người dùng nhập nhãn đúng với phân biệt hoa/thường.
- Ảnh sửa được lưu vào `user_feedback_47_classes_16x16.npz`.
- Lần train tiếp theo chỉ nối feedback vào tập train.

#### 4.7. Tính tái lập

Nêu seed, cấu hình máy, phiên bản Python và NumPy. Code hiện mới dùng seed cho chọn mẫu và chia dữ liệu; để thí nghiệm tái lập hoàn toàn cần đặt thêm `np.random.seed(RANDOM_SEED)` trước khi khởi tạo model và trước quá trình shuffle từng epoch.

### Chương 5. Thiết kế thí nghiệm và kết quả

#### 5.1. Môi trường thí nghiệm

Ghi CPU, RAM, hệ điều hành, Python, NumPy, số luồng, không dùng GPU. Mọi thí nghiệm so sánh phải dùng cùng một train/validation/test split.

#### 5.2. Thí nghiệm cơ sở

Mục tiêu: xác định chất lượng mô hình hiện tại với 9.964 ảnh, 10 epoch và learning rate 0.008.

Kết quả cần lưu sau từng epoch:

- Train loss và validation loss.
- Train accuracy và validation accuracy.
- Thời gian mỗi epoch.
- Test loss, test accuracy và macro-F1 sau epoch cuối.

Biểu đồ cần vẽ:

- Hai đường train/validation loss theo epoch.
- Hai đường train/validation accuracy theo epoch.

#### 5.3. Ảnh hưởng của số lượng dữ liệu

Giữ nguyên kiến trúc, learning rate, split ratio và số epoch. Dùng các mức chia hết cho 47 để dữ liệu cân bằng tuyệt đối:

| Ảnh/lớp | Tổng số ảnh | Epoch đề xuất |
|---:|---:|---:|
| 50 | 2.350 | 10 |
| 100 | 4.700 | 10 |
| 150 | 7.050 | 10 |
| 212 | 9.964 | 10 |

Kết quả cần so sánh: test accuracy, macro-F1, thời gian train và mức cải thiện khi tăng dữ liệu.

Biểu đồ cần vẽ:

- Trục X là tổng số ảnh, trục Y là test accuracy/macro-F1.
- Biểu đồ cột thời gian huấn luyện cho bốn mức dữ liệu.

#### 5.4. Ảnh hưởng của learning rate

So sánh `0.002`, `0.004`, `0.008` và `0.012`. Chạy thử 5 epoch để chọn vùng tốt, sau đó chạy 10 epoch cho cấu hình tốt nhất. Giữ nguyên seed và split.

Biểu đồ cần vẽ:

- Validation loss theo epoch cho từng learning rate.
- Bảng accuracy, macro-F1 và dấu hiệu mất ổn định như loss tăng hoặc NaN.

#### 5.5. So sánh padding

So sánh `conv_padding = 0` và `conv_padding = 1`.

- Padding 0 tạo đặc trưng cuối 2x2x32 = 128 và khoảng 27.375 tham số.
- Padding 1 tạo đặc trưng cuối 4x4x32 = 512 và 76.527 tham số.

Kết quả cần so sánh: test accuracy, macro-F1, thời gian train, thời gian dự đoán và kích thước model. Thí nghiệm này đồng thời giải thích vì sao kiến trúc cuối cùng được chọn.

#### 5.6. Phân tích theo từng lớp

Từ dự đoán trên test set, tạo:

- Confusion matrix 47x47.
- Precision, recall và F1 của từng lớp.
- Danh sách 10 cặp nhãn bị nhầm nhiều nhất.
- Một lưới ảnh minh họa dự đoán đúng, dự đoán sai và confidence.

Các cặp đáng chú ý để kiểm tra gồm `0/O`, `1/I`, `B/b`, `D/d`, `E/e`, `G/g`, `H/h`, `N/n`, `Q/q`, `R/r` và `T/t`. Chỉ kết luận cặp nào khó sau khi có số liệu confusion matrix.

Biểu đồ cần vẽ:

- Heatmap confusion matrix, có thể thêm một bản chuẩn hóa theo hàng.
- Biểu đồ cột F1-score theo 47 nhãn.

#### 5.7. Độ bền với dịch chuyển và căn giữa

Tạo bản sao test set bị dịch ngẫu nhiên từ -2 đến 2 pixel. So sánh:

- Dự đoán trực tiếp không căn giữa.
- Dự đoán sau crop và căn giữa như trong GUI.

Kết quả này kiểm chứng tác dụng của bước `center_image` thay vì chỉ mô tả bằng cảm tính.

#### 5.8. Hiệu năng hệ thống

Đo các giá trị:

- Tổng số tham số.
- Kích thước file model.
- Tổng thời gian train và trung bình mỗi epoch.
- Thời gian dự đoán trung bình trên ít nhất 500 ảnh sau một lần warm-up.
- Peak RAM nếu có công cụ đo phù hợp.

Không cần so sánh CPU với GPU vì phiên bản hiện tại dùng nhiều vòng lặp Python và không có triển khai GPU được vector hóa tương đương.

### Chương 6. Thảo luận

#### 6.1. Diễn giải kết quả

- Mô hình có hội tụ hay không.
- Khoảng cách giữa train và validation accuracy.
- Tăng dữ liệu mang lại bao nhiêu cải thiện.
- Padding và learning rate ảnh hưởng thế nào.
- Những nhóm ký tự nào khó và nguyên nhân hình học có thể có.

#### 6.2. Hạn chế

- Ảnh 16x16 làm mất nhiều chi tiết.
- EMNIST Balanced chỉ có 11 lớp chữ thường riêng.
- SGD theo từng ảnh và convolution bằng vòng lặp Python làm train chậm.
- Chưa có mini-batch, optimizer momentum/Adam hoặc early stopping.
- Feedback hiện yêu cầu retrain từ đầu.
- Feedback do một người cung cấp có thể làm model thiên về nét viết của người đó.
- MaxPool backward hiện truyền gradient tới mọi vị trí cùng đạt giá trị cực đại; đây là một lựa chọn cần nêu hoặc cải thiện nếu muốn bám sát cách cài đặt phổ biến.

#### 6.3. Đe dọa tới tính hợp lệ

- Kết quả một seed có thể phụ thuộc khởi tạo ngẫu nhiên.
- User-holdout quá nhỏ hoặc chỉ từ một người không đại diện cho nhiều phong cách viết.
- So sánh mô hình không công bằng nếu dùng split khác nhau.
- Chọn hyperparameter dựa trên test set gây rò rỉ thông tin; chỉ được chọn bằng validation set.

### Chương 7. Kết luận và hướng phát triển

- Tóm tắt mục tiêu và các thành phần đã hoàn thành.
- Nêu kết quả tốt nhất bằng số liệu thực tế.
- Nhắc feedback như cơ chế thu thập dữ liệu cho các lần cập nhật sau, không đưa ra kết luận định lượng về hiệu quả.
- Hướng phát triển: mini-batch/vectorization, lưu checkpoint để fine-tune, augmentation trong lúc train, thu thập nhiều người viết, đủ 62 lớp nếu có dataset phù hợp và triển khai giao diện tốt hơn.

### Phụ lục

- Pseudocode forward/backward.
- Danh sách 47 nhãn.
- Cấu hình chạy thí nghiệm.
- Bảng kết quả đầy đủ qua từng seed.
- Hướng dẫn chạy `test_draw`, `train.py` và `test_brain.py`.
- Không nên dán toàn bộ các file Python vào phần nội dung chính.

## 4. Danh sách hình cần tạo hoặc chụp

| Mã hình | Nội dung | Cách tạo/chụp | Chương |
|---|---|---|---|
| F01 | Pipeline tổng thể của hệ thống | Vẽ flowchart từ dataset đến feedback và retrain | 1 hoặc 4 |
| F02 | 47 ảnh mẫu, mỗi lớp một ảnh | Lấy một ảnh ngẫu nhiên từ mỗi nhãn và ghép lưới | 3 |
| F04 | Phân bố số ảnh theo lớp trước và sau chọn 9.964 mẫu | Bar chart 47 cột; sau chọn mọi cột phải bằng 212 | 3 |
| F05 | Sơ đồ kiến trúc CNN và kích thước tensor | Vẽ block diagram theo bảng kiến trúc | 4 |
| F06 | Một số feature map sau Conv1 và Conv2 | Chạy forward một ảnh và lưu các map đại diện | 4, tùy chọn |
| F07 | GUI khi chưa vẽ | Chụp cửa sổ ban đầu | 4 |
| F08 | GUI dự đoán đúng | Vẽ một ký tự, bấm Predict, chụp nhãn/confidence | 4 |
| F09 | GUI dự đoán sai và nhập nhãn sửa | Chụp trạng thái nhập đúng chữ hoa/thường | 4 |
| F10 | GUI xác nhận đã lưu feedback | Chụp thông báo sau Save fix | 4 |
| F11 | Train/validation loss | Vẽ từ lịch sử 10 epoch | 5 |
| F12 | Train/validation accuracy | Vẽ từ lịch sử 10 epoch | 5 |
| F13 | Accuracy theo số lượng dữ liệu | Line chart cho 2.350, 4.700, 7.050, 9.964 ảnh | 5 |
| F14 | Thời gian train theo số lượng dữ liệu | Bar chart | 5 |
| F15 | Validation loss theo learning rate | Một đường cho mỗi learning rate | 5 |
| F16 | Confusion matrix 47 lớp | Heatmap số lượng hoặc tỷ lệ chuẩn hóa | 5 |
| F17 | F1-score từng lớp | Bar chart theo ký tự | 5 |
| F18 | Các ảnh dự đoán sai tiêu biểu | Lưới ảnh kèm true label, predicted label, confidence | 5 |
| F19 | Padding 0 so với padding 1 | Bar chart accuracy, tham số và thời gian | 5 |
| F20 | Trước và sau căn giữa | Ghép ảnh bị lệch và ảnh đã center | 5 |
| F21 | Accuracy trên ảnh dịch chuyển trước/sau center | Grouped bar chart | 5 |

Ảnh GUI phải chụp từ chương trình thật. Các biểu đồ phải sinh từ file kết quả, không nhập số thủ công và không dùng số liệu giả.

## 5. Danh sách bảng nên có

| Mã bảng | Nội dung |
|---|---|
| T01 | Tóm tắt dataset: shape, dtype, pixel range, số lớp, ảnh/lớp |
| T02 | Danh sách và ánh xạ 47 nhãn |
| T03 | Phân chia train/validation/test |
| T04 | Kiến trúc CNN, output shape và số tham số |
| T05 | Hyperparameter mặc định |
| T06 | Kết quả baseline qua từng epoch |
| T07 | So sánh số lượng dữ liệu |
| T08 | So sánh learning rate |
| T09 | So sánh padding 0 và 1 |
| T10 | Precision/Recall/F1 theo lớp hoặc 10 lớp khó nhất |
| T11 | Các cặp nhãn bị nhầm nhiều nhất |
| T12 | Hiệu năng: tham số, model size, train time, inference time |

## 6. Thứ tự ưu tiên thí nghiệm

### Bắt buộc

1. Kiểm tra dataset và biểu đồ cân bằng lớp.
2. Baseline 10 epoch với learning curves.
3. Test accuracy, macro-F1 và confusion matrix.
4. Phân tích các lớp/cặp ký tự dễ nhầm.
5. Chụp đầy đủ GUI và luồng feedback.

### Nên có để báo cáo thuyết phục hơn

1. Ảnh hưởng của số lượng dữ liệu.
2. So sánh padding 0 và padding 1.
3. So sánh learning rate.
4. Đo thời gian train và inference.

### Chỉ làm khi còn thời gian

1. Feature-map visualization.
2. Chạy mỗi cấu hình với 3 seed và báo cáo trung bình ± độ lệch chuẩn.
3. Xây dựng một MLP từ đầu làm baseline để chứng minh lợi ích của convolution.

Nếu thời gian CPU hạn chế, dùng 5 epoch để sàng lọc learning rate/kiến trúc, sau đó chỉ chạy cấu hình thắng 10 epoch. Không nên hy sinh baseline và confusion matrix để chạy quá nhiều cấu hình phụ.

## 7. Dữ liệu kết quả cần lưu để vẽ biểu đồ

`train.py` hiện chỉ in kết quả ra màn hình. Trước khi chạy thí nghiệm chính thức nên bổ sung việc lưu một file CSV cho mỗi lần chạy với các cột:

```text
experiment_id, seed, max_samples, padding, learning_rate, epoch,
train_loss, train_accuracy, val_loss, val_accuracy, epoch_seconds
```

Sau khi test, lưu thêm:

```text
test_loss, test_accuracy, macro_precision, macro_recall, macro_f1,
model_parameters, model_size_bytes, mean_inference_ms
```

Ngoài ra cần lưu `y_true`, `y_pred` và confidence của test set để tạo confusion matrix và lưới ảnh sai. Nên dùng cấu trúc thư mục:

```text
results/
  baseline/
  sample_size/
  learning_rate/
  padding/
report_assets/
  figures/
  screenshots/
  tables/
```

## 8. Nguyên tắc để kết quả có giá trị

- Không chọn kiến trúc hoặc learning rate dựa trên test set.
- Mọi cấu hình so sánh phải dùng cùng seed và cùng split.
- Chỉ thay đổi một yếu tố trong mỗi thí nghiệm.
- Báo cáo cả accuracy và macro-F1, không chỉ chọn chỉ số đẹp nhất.
- Ghi rõ tổng số ảnh thực tế là 9.964 thay vì làm tròn thành 10.000.
- Nếu chỉ chạy một lần, ghi đây là hạn chế; nếu đủ thời gian, chạy 3 seed và báo cáo trung bình ± độ lệch chuẩn.
- Mọi nhận xét như “padding tốt hơn” chỉ được viết sau khi có kết quả đo.

## 9. Checklist hoàn thiện báo cáo

- [ ] Đổi toàn bộ tiêu đề và nội dung customer churn trong `report.tex`.
- [ ] Đồng bộ mô tả dataset thành EMNIST Balanced 47 lớp.
- [ ] Đồng bộ kiến trúc thành padding 1 và Flatten 512 nếu đây là cấu hình cuối.
- [ ] Sửa “fine-tuning” thành “retraining with feedback” nếu chưa đổi code.
- [ ] Thêm seed đầy đủ và lưu lịch sử train.
- [ ] Chạy baseline 10 epoch.
- [ ] Tạo confusion matrix và per-class metrics.
- [ ] Chạy ít nhất một thí nghiệm so sánh.
- [ ] Chụp bốn trạng thái chính của GUI.
- [ ] Điền bảng và biểu đồ bằng số liệu thật.
- [ ] Viết lại chương kết luận dựa trên kết quả.
- [ ] Kiểm tra tài liệu tham khảo, logo, caption, label và đánh số hình/bảng.
- [ ] Kiểm tra không còn từ khóa churn, uplift, CLV, telecom hoặc retention offer.
