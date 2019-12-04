1. 按照requirment.txt文件安装需要的库, pip install -r requirment.txt
2. pip install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
3. 运行train_test_split.py
4. 运行word2vec_process.py 为baseline和主模型的输入做准备
5. 分别运行baseline_lr.py baseline_rf.py baseline_svm.py，并记录baseline结果
6. 运行model_dl.py/model_nn.py，记录主模型结果,不理想可调参

English: 
1. pip install -r requirment.txt
<!-- 2. pip install torch==1.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html -->
3. run train_test_split.py
4. run word2vec_process.py 
5. run baseline_lr.py baseline_rf.py baseline_svm.py，to get the results for logistic regression, randome forest, and svm
6. run model_nn.py，for the neuronet 