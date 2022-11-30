# ML4Security


---
## Reference 

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
## Glossory

- https://d2l.aivivn.com/glossary.html


---
## Requirement

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
## Dataset
- sklearn.datasets: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

----
## Machine learning cơ bản
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
## Numpy. Pandas, mathplotlib

---
## Data Crawling and pre-processing

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
## Machine Learning in Python: Step-By-Step Tutorial

:pushpin: ML_basic_stepbystep.ipynb

---
## Hồi quy tuyến tính: Linear regression 
> Đây là một thuật toán Supervised learning có tên Linear Regression (Hồi Quy Tuyến Tính)

- regression problem
- linear model
- prediction 
- learning a regression function: learning goal, difficulty
- loss function 

---
## K-means Clustering
> thuật toán cơ bản nhất trong Unsupervised learning - thuật toán K-means clustering (phân cụm K-means).
>
> Trong thuật toán K-means clustering, chúng ta không biết nhãn (label) của từng điểm dữ liệu. Mục đích là làm thể nào để phân dữ liệu thành các cụm (cluster) khác nhau sao cho dữ liệu trong cùng một cụm có tính chất giống nhau.




----
## Gradient Descent
> Gradient descent là thuật toán tìm giá trị nhỏ nhất của hàm số f(x) dựa trên đạo hàm

- Gradient Descent 
- Gradient Descent cho hàm 1 biến
- Gradient Descent cho hàm nhiều biến
- Gradient Descent phụ thuộc vào **điểm khởi tạo** và **learning rate**
- các thuật toán tối ưu Gradient Descent 
- Biến thể của Gradient Descent 


---
## Các thuật toán ml khác

- Support Vector Machine
- Decision Trees



----
## Feature engineering
- Mô hình chung cho các bài toán ML


- Raw test data -> Feature Extraction (Feature engineering)

---
## Ember dataset


---
## Overfiting
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
## Đánh giá hiệu quả của một mô hình học máy



---
## Giới thiệu về deep learning
> Deep learning được bắt nguồn từ thuật toán Neural network, chỉ là một ngành nhỏ của machine learning nhưng nó giống như con gà để trứng vàng vậy.

![image](https://user-images.githubusercontent.com/108725538/204594285-245f722b-835a-4fb7-b590-9ff7972e3a31.png)

---
## Perceptron Learning Algorithm
> Perceptron Learning Algorithm (PLA) hoặc đôi khi được viết gọn là Perceptron. thuật toán đầu tiên trong Classification. Perceptron là một thuật toán Classification cho trường hợp đơn giản nhất: chỉ có hai class (lớp) (bài toán với chỉ hai class được gọi là binary classification) và cũng chỉ hoạt động được trong một trường hợp rất cụ thể. Tuy nhiên, nó là nền tảng cho một mảng lớn quan trọng của Machine Learning là Neural Networks và sau này là Deep Learning


----
## Binary Classifiers

- Các phương pháp sử dụng binary classifiers vào các bài toán multi-class classification 
    - one-vs-one
    - hieratchical (phân tầng)
    - Binary coding
    - one-vs-rest hay one-hot coding

---
## Softmax Regression
> 


----
## Multi-layer Perceptron và Backpropagation

- Layers
- Units
- Weights và Biases
- Activation functions
    - Sigmoid và tanh 
    - ReLU (Rectified Linear Unit)
- Backpropagation
- mini-batch

---
## Anomaly Detection 

----
## Machine learning-based Malware Detection
