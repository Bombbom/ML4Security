# 1. ML4Security

- [1. ML4Security](#1-ml4security)
  - [1.1. Reference](#11-reference)
  - [1.2. Glossory](#12-glossory)
  - [1.3. Requirement](#13-requirement)
  - [1.4. Dataset](#14-dataset)
  - [1.5. Machine learning cơ bản](#15-machine-learning-cơ-bản)
  - [1.6. Numpy. Pandas, mathplotlib](#16-numpy-pandas-mathplotlib)
  - [1.7. Data Crawling and pre-processing](#17-data-crawling-and-pre-processing)
  - [1.8. Machine Learning in Python: Step-By-Step Tutorial](#18-machine-learning-in-python-step-by-step-tutorial)
  - [1.9. Hồi quy tuyến tính: Linear regression](#19-hồi-quy-tuyến-tính-linear-regression)
  - [1.10. K-means Clustering](#110-k-means-clustering)
  - [1.11. Gradient Descent](#111-gradient-descent)
  - [1.12. Các thuật toán ml khác](#112-các-thuật-toán-ml-khác)
  - [1.13. Feature engineering](#113-feature-engineering)
  - [1.14. Ember dataset](#114-ember-dataset)
  - [1.15. Overfiting](#115-overfiting)
  - [1.16. Đánh giá hiệu quả của một mô hình học máy](#116-đánh-giá-hiệu-quả-của-một-mô-hình-học-máy)
  - [1.17. Giới thiệu về deep learning](#117-giới-thiệu-về-deep-learning)
  - [1.18. Perceptron Learning Algorithm](#118-perceptron-learning-algorithm)
  - [1.19. Binary Classifiers](#119-binary-classifiers)
  - [1.20. Softmax Regression](#120-softmax-regression)
  - [1.21. Multi-layer Perceptron và Backpropagation](#121-multi-layer-perceptron-và-backpropagation)
  - [1.22. Anomaly Detection](#122-anomaly-detection)
  - [1.23. Machine learning-based Malware Detection](#123-machine-learning-based-malware-detection)

----
## Introduction


---
## 1.1. Reference 

- https://machinelearningcoban.com/

- https://d2l.aivivn.com/intro_vn.html

- https://phamdinhkhanh.github.io/content

- https://machinelearningmastery.com/

- https://www.youtube.com/watch?v=jc1wo_8VA1w&list=PLaKukjQCR56ZRh2cAkweftiZCF2sTg11_


<!-- 1. Your First Machine Learning Project in Python Step-By-Step,
https://machinelearningmastery.com/machine-learning-in-python-step-bystep/
2. Machine Learning Crash Course, https://developers.google.com/machinelearning/crash-course
3. How to Develop a GAN for Generating MNIST Handwritten Digits,
https://machinelearningmastery.com/how-to-develop-a-generativeadversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
4. GANs from Scratch 1: A deep introduction. With code in PyTorch and
TensorFlow, https://medium.com/ai-society/gans-from-scratch-1-a-deepintroduction-with-code-in-pytorch-and-tensorflow-cb03cdcdba0f
5. Federated Learning: Collaborative Machine Learning without Centralized
Training Data, https://ai.googleblog.com/2017/04/federated-learningcollaborative.html
6. https://cset.georgetown.edu/wp-content/uploads/Machine-Learning-andCybersecurity.pdf
7. http://web.stanford.edu/class/cs259d/#lectures
8. https://www.malwaredatascience.com/
9. https://github.com/oreilly-mlsec/book-resources -->


:bookmark: Book


---
## 1.2. Glossory

- https://d2l.aivivn.com/glossary.html


---
## 1.3. Requirement

- Linear algebra (Đại số tuyến tính)
- Calculus (Giải tích)
- Probability & Statistic (Xác suất thống kê)
- Optimization Problem: Convex Optimization (Tối ưu hóa hàm lồi)
- TensoFlow 
- Pytorch 
- Keras 
- Python 
- Colab
- Anaconda/Jupyter-notebook
- Google Colab for Machine Learning Projects: https://machinelearningmastery.com/google-colab-for-machine-learning-projects/

---
## 1.4. Dataset
- sklearn.datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets
- mnist dataset
- ethereum smart contract dataset


----
## 1.5. Machine learning cơ bản
> An Inroduction for Beginners 

- Mối quan hệ giữa AI, ML và DL 
- Phân loại các thuật toán ML (theo phương thức học)
    - Supervised Leaning
        - Logistic Regression
        - Decision Tree, Forest 
        - Suppor Vector Machine (SVM)
        - Naive Bayes, k-nearest neighbor (KNN)
        - Neural Network 
    - Unsupervised Learning
    - Semi-supervised Learning
    - Reinforcement Learning: Reinforcement learning là các bài toán giúp cho một hệ thống tự động xác định hành vi dựa trên hoàn cảnh để đạt được lợi ích cao nhất (maximizing the performance). Hiện tại, Reinforcement learning chủ yếu được áp dụng vào Lý Thuyết Trò Chơi (Game Theory), các thuật toán cần xác định nước đi tiếp theo để đạt được điểm số cao nhất.
- Phân loại các thuật toán ML (dựa trên chức năng)


- Định nghĩa ML: task, experience, performance (P,E,T)
- Ứng dụng của ML:
    - computer science
    - statistics, probability
    - optimization
    - psychology, neuroscience 
    - computer vision 
    - economics, biology, bioinformatics 
- Example: 
    - Spam filtering for emails
    - image tagging 

- Định nghĩa:
    - mapping (function)
    - model (mô hình)
    - traning set
    - test set
- supervised learning: classification
    - multiclass classification
    - multilabel classification 

- Design a learning system
    - Select a training set
    - Determine the type of the function to be learned
 


---
## 1.6. Numpy. Pandas, mathplotlib, sklearn, tensorflow, keras, ....

- https://pandas.pydata.org/docs/
- https://numpy.org/doc/stable/
- https://matplotlib.org/stable/index.html

---
## 1.7. Data Crawling and pre-processing

- thu thập dữ liệu
    - lấy mẫu (sampling)
    - crawling, logging, scraping
- xử lý dữ liệu
    - lọc nhiễu, làm sạch, số hóa, ...

- pre-procesing
    - clean
    - visual 
    - transforming 
    - feature selection 
    - dimension reduction 
    - trừu tượng hóa




---
## 1.9. Hồi quy tuyến tính: Linear regression 
> Đây là một thuật toán Supervised learning có tên Linear Regression (Hồi Quy Tuyến Tính)

- regression problem
- linear model
- prediction 
- learning a regression function: learning goal, difficulty
- loss function 

---
## 1.10. K-means Clustering
> thuật toán cơ bản nhất trong Unsupervised learning - thuật toán K-means clustering (phân cụm K-means).
>
> Trong thuật toán K-means clustering, chúng ta không biết nhãn (label) của từng điểm dữ liệu. Mục đích là làm thể nào để phân dữ liệu thành các cụm (cluster) khác nhau sao cho dữ liệu trong cùng một cụm có tính chất giống nhau.




----
## 1.11. Gradient Descent
> Gradient descent là thuật toán tìm giá trị nhỏ nhất của hàm số f(x) dựa trên đạo hàm

- Gradient Descent 
- Gradient Descent cho hàm 1 biến
- Gradient Descent cho hàm nhiều biến
- Gradient Descent phụ thuộc vào **điểm khởi tạo** và **learning rate**
- các thuật toán tối ưu Gradient Descent 
- Biến thể của Gradient Descent 


---
## 1.12. Các thuật toán ml khác

- Support Vector Machine
- Decision Trees



----
## 1.13. Feature engineering
- Mô hình chung cho các bài toán ML


- Raw test data -> Feature Extraction (Feature engineering)

- feature engineering
    - Trích lọc feature: Không phải toàn bộ thông tin được cung cấp từ một biến dự báo hoàn toàn mang lại giá trị trong việc phân loại. Do đó chúng ta cần phải trích lọc những thông tin chính từ biến đó. Chẳng hạn như trong các mô hình chuỗi thời gian chúng ta thường sử dụng kĩ thuật phân rã thời gian để trích lọc ra các đặc trưng như Ngày thành Năm, Tháng, Quí,…. Các đặc trưng mới sẽ giúp phát hiện các đặc tính chu kì và mùa vụ, những đặc tính mà thường xuất hiện trong các chuỗi thời gian. Kĩ thuật trích lọc đặc trưng thông thường được áp dụng trên một số dạng biến như:
        - Trích lọc đặc trưng trong xử lý ảnh và xử lý ngôn ngữ tự nhiên: Các mạng nơ ron sẽ trích lọc ra những đặc trưng chính và học từ những đặc trưng này để thực hiện tác vụ phân loại.
        - Dữ liệu về vị trí địa lý: Từ vị trí địa lý có thể suy ra vùng miền, thành thị, nông thôn, mức thu nhập trung bình, các yếu tố về nhân khẩu,….
        - Dữ liệu thời gian: Phân rã thời gian thành các thành phần thời gian
    - Biến đổi feature: Biến đổi dữ liệu gốc thành những dữ liệu phù hợp với mô hình nghiên cứu. Những biến này thường có tương quan cao hơn đối với biến mục tiêu và do đó giúp cải thiện độ chính xác của mô hình. Các phương pháp này bao gồm:
        - Chuẩn hóa và thay đổi phân phối của dữ liệu thông qua các kĩ thuật feature scaling như Minmax scaling, Mean normalization, Unit length scaling, Standardization.
        - Tạo biến tương tác: Trong thống kê các bạn hẳn còn nhớ kiểm định ramsey reset test về mô hình có bỏ sót biến quan trọng? Thông qua việc thêm vào mô hình các biến bậc cao và biến tương tác để tạo ra một mô hình mới và kiểm tra hệ số các biến mới có ý nghĩa thống kê hay không. Ý tưởng của tạo biến tương tác cũng gần như thế. Tức là chúng ta sẽ tạo ra những biến mới là các biến bậc cao và biến tương tác.
        - Xử lý dữ liệu missing: Có nhiều lý do khiến ta phải xử lý missing data. Một trong những lý do đó là dữ liệu missing cũng mang những thông tin giá trị, do đó nếu thay thế được các missing bằng những giá trị gần đúng sẽ mang lại nhiều thông tin hơn cho mô hình. Bên cạnh đó nhiều mô hình không làm việc được với dữ liệu missing dẫn tới lỗi training. Do đó ta cần giải quyết các biến missing. Đối với biến numeric, các phương pháp đơn giản nhất là thay thế bằng mean, median,…. Một số kĩ thuật cao cấp hơn sử dụng phân phối ngẫu nhiên để fill các giá trị missing dựa trên phân phối của các giá trị đã biết hoặc sử dụng phương pháp simulate missing value dựa trên trung bình của các quan sát láng giềng. Đối với dữ liệu category, missing value có thể được giữ nguyên như một class độc lập hoặc gom vào các nhóm khác có đặc tính phân phối trên biến mục tiêu gần giống.
    - Lựa chọn feature: Phương pháp này được áp dụng trong những trường hợp có rất nhiều dữ liệu mà chúng ta cần lựa chọn ra dữ liệu có ảnh hưởng lớn nhất đến sức mạnh phân loại của mô hình. Các phương pháp có thể áp dụng đó là ranking các biến theo mức độ quan trọng bằng các mô hình như Random Forest, Linear Regression, Neural Network, SVD,…; Sử dụng chỉ số IV trong scorecard; Sử dụng các chỉ số khác như AIC hoặc Pearson Correlation, phương sai. Chúng ta có thể phân chia các phương pháp trên thành 3 nhóm:
        - Cách tiếp cận theo phương pháp thống kê: Sử dụng tương quan Pearson Correlation, AIC, phương sai, IV.
        - Lựa chọn đặc trưng bằng sử dụng mô hình: Random Forest, Linear Regression, Neural Network, SVD.
        - Lựa chọn thông qua lưới (grid search): Coi số lượng biến như một thông số của mô hình. Thử nghiệm các kịch bản với những số lượng biến khác nhau. Các bạn có thể xem cách thực hiện grid search.


---
## 1.14. Ember dataset


---
## 1.15. Overfiting
> Overfitting không phải là một thuật toán trong Machine Learning. Nó là một hiện tượng không mong muốn thường gặp, người xây dựng mô hình Machine Learning cần nắm được các kỹ thuật để tránh hiện tượng này.

- Sự thật là nếu một mô hình quá fit với dữ liệu thì nó sẽ gây phản tác dụng! Hiện tượng quá fit này trong Machine Learning được gọi là overfitting, là điều mà khi xây dựng mô hình, chúng ta luôn cần tránh.

- validation 
> Phương pháp đơn giản nhất là trích từ tập training data ra một tập con nhỏ và thực hiện việc đánh giá mô hình trên tập con nhỏ này. Tập con nhỏ được trích ra từ training set này được gọi là validation set. Lúc này, training set là phần còn lại của training set ban đầu
>
> Thông thường, ta bắt đầu từ mô hình đơn giản, sau đó tăng dần độ phức tạp của mô hình. Tới khi nào validation error có chiều hướng tăng lên thì chọn mô hình ngay trước đó. Chú ý rằng mô hình càng phức tạp, train error có xu hướng càng nhỏ đi.

- Cross-validation
> Cross validation là một cải tiến của validation với lượng dữ liệu trong tập validation là nhỏ nhưng chất lượng mô hình được đánh giá trên nhiều tập validation khác nhau. Một cách thường đường sử dụng là chia tập training ra  k tập con không có phần tử chung, có kích thước gần bằng nhau. Tại mỗi lần kiểm thử , được gọi là run, một trong số k tập con được lấy ra làm validate set. Mô hình sẽ được xây dựng dựa vào hợp của  k − 1  tập con còn lại. Mô hình cuối được xác định dựa trên trung bình của các train error và validation error. Cách làm này còn có tên gọi là k-fold cross validation.


- Regularization
    - Early Stopping
    - Thêm số hạng vào hàm mất mát
    - l_2 regularization
    - Tikhonov regularization
    - Regularizers for sparsity
> Regularization, một cách cơ bản, là thay đổi mô hình một chút để tránh overfitting trong khi vẫn giữ được tính tổng quát của nó (tính tổng quát là tính mô tả được nhiều dữ liệu, trong cả tập training và test). Một cách cụ thể hơn, ta sẽ tìm cách di chuyển nghiệm của bài toán tối ưu hàm mất mát tới một điểm gần nó. Hướng di chuyển sẽ là hướng làm cho mô hình ít phức tạp hơn mặc dù giá trị của hàm mất mát có tăng lên một chút.


----
## Mất cân bằng dữ liệu (imbalanced dataset)
> Mất cân bằng dữ liệu là một trong những hiện tượng phổ biến của bài toán phân loại nhị phân (binary classification) như spam email, phát hiện gian lận, dự báo vỡ nợ, chuẩn đoán bệnh lý,…. Trong trường hợp tỷ lệ dữ liệu giữa 2 classes là 50:50 thì được coi là cân bằng. Khi có sự khác biệt trong phân phối giữa 2 classes, chẳng hạn 60:40 thì dữ liệu có hiện tượng mất cân bằng.

- Các phương pháp giải quyết dữ liệu mất cân bằng
    - Thay đổi metric
    - Under sampling: Under sampling là việc ta giảm số lượng các quan sát của nhóm đa số để nó trở nên cân bằng với số quan sát của nhóm thiểu số. Ưu điểm của under sampling là làm cân bằng mẫu một cách nhanh chóng, dễ dàng tiến hành thực hiện mà không cần đến thuật toán giả lập mẫu.
    - Over sampling: Over sampling là các phương pháp giúp giải quyết hiện tượng mất cân bằng mẫu bằng cách gia tăng kích thước mẫu thuộc nhóm thiểu số bằng các kĩ thuật khác nhau. Có 2 phương pháp chính để thực hiện over sampling đó là:
        - Lựa chọn mẫu có tái lập.
        - Mô phỏng mẫu mới dựa trên tổng hợp của các mẫu cũ



----
## 1.16. Đánh giá hiệu quả của một mô hình học máy
> Đánh giá mô hình giúp chúng ta lựa chọn được mô hình phù hợp nhất đối với bài toán của mình. Tuy nhiên để tìm được thước đo đánh giá mô hình phù hợp thì chúng ta cần phải hiểu về ý nghĩa, bản chất và trường hợp áp dụng của từng thước đo.

- Bộ dữ liệu
- confusion matrix 
    - TP (True Positive): Tổng số trường hợp dự báo khớp Positive.
    - TN (True Negative): Tổng số trường hợp dự báo khớp Negative.
    - FP (False Positive): Tổng số trường hợp dự báo các quan sát thuộc nhãn Negative thành Positive.
    - FN (False Negative): Tổng số trường hợp dự báo các quan sát thuộc nhãn Positive thành Negative.

- Accuracy 
- Precision
- Recall
- F1 Score
- AUCvs ROC 
- TPR và FPR
- gini và CAP

---
## 1.17. Giới thiệu về deep learning
> Deep learning được bắt nguồn từ thuật toán Neural network, chỉ là một ngành nhỏ của machine learning nhưng nó giống như con gà để trứng vàng vậy.

![image](https://user-images.githubusercontent.com/108725538/204594285-245f722b-835a-4fb7-b590-9ff7972e3a31.png)

---
## 1.18. Perceptron Learning Algorithm
> Perceptron Learning Algorithm (PLA) hoặc đôi khi được viết gọn là Perceptron. thuật toán đầu tiên trong Classification. Perceptron là một thuật toán Classification cho trường hợp đơn giản nhất: chỉ có hai class (lớp) (bài toán với chỉ hai class được gọi là binary classification) và cũng chỉ hoạt động được trong một trường hợp rất cụ thể. Tuy nhiên, nó là nền tảng cho một mảng lớn quan trọng của Machine Learning là Neural Networks và sau này là Deep Learning

- Batch size: số lượng dữ liệu Mini-Batch Gradient Descent sử dụng trong 1 lần để cập nhật tham số
- Epoch: 1 epoch là một lần duyệt qua hết các dữ liệu trong tập huấn luyện
- Iterations: số lượng các Batch size mà mô hình phải duyệt trong 1 epoch.
Ví dụ tập huấn luyện có 32.000 dữ liệu. Nếu Batch size = 32 (mỗi lần cập nhật trọng số sẽ sử dụng 32 dữ liệu), khi đó Iterations =32.000/32=1000 để có thể duyệt qua hết các dữ liệu (hoàn thành 1 epoch). Các giá trị Batch size thường dùng là 32, 64, 128, 256... (2^n để việc tính toán được nhanh hơn). Tổng quát hơn thì đối với Stochastic Gradient Descent, Batch size = số dữ liệu trong tập huấn luyện, đối với Stochastic Gradient Descent, Batch size = 1.

----
## 1.19. Binary Classifiers

- Các phương pháp sử dụng binary classifiers vào các bài toán multi-class classification 
    - one-vs-one
    - hieratchical (phân tầng)
    - Binary coding
    - one-vs-rest hay one-hot coding

---
## 1.20. Softmax Regression
> 


----
## 1.21. Multi-layer Perceptron và Backpropagation

- Layers
- Units
- Weights và Biases
- Activation functions
    - Sigmoid và tanh 
    - ReLU (Rectified Linear Unit)
- Backpropagation
- mini-batch


---
## RNN - Recurrent Neural Network


---
## LSTM  - LSTM - Long short term memory

---
## BiLSTM

---
## Word Representation

- Định nghĩa One-hot véc tơ của từ
> Sau khi biểu diễn từ dưới dạng one-hot véc tơ, mô hình đã có thể huấn luyện được từ dữ liệu được mã hóa. Tuy nhiên dữ liệu này chỉ đáp ứng được khả năng huấn luyện mà chưa phản ảnh được mối liên hệ về mặt ngữ nghĩa của các từ.

- Do đó các thuật toán nhúng từ được tạo ra nhằm mục đích tìm ra các véc tơ đại diện cho mỗi từ sao cho:
    - Một từ được biểu diễn bởi một véc tơ có số chiều xác định trước.
    - Các từ thuộc cùng 1 nhóm thì có khoảng cách gần nhau trong không gian.
    - 3 nhóm chính
        - Sử dụng thống kê tần xuất: tfidf
        - Các thuật toán giảm chiều dữ liệu: SVD, PCA, auto encoder, word2vec
        - Phương pháp sử dụng mạng nơ ron: word2vec, ELMo, BERT


- Tokenization in NLP 
- Word2vec 
    - skip-grams 
    - CBOW: . Về cơ bản thì CBOW là một quá trình ngược lại của skip-grams. Khi đó input của skip-grams sẽ được sử dụng làm output trong CBOW và ngược lại.


---
## 1.22. Anomaly Detection 

----
## 1.23. Machine learning-based Malware Detection


------
------
# Practices


---
## Machine Learning in Python: Step-By-Step Tutorial

:pushpin: ML_basic_stepbystep.ipynb

