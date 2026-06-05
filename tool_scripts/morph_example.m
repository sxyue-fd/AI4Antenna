% 1. 创建一个简单的二值图像 (或读取现有图像)
A = zeros(10, 10);
A(4:7, 4:7) = 1; % 在中间创建一个 4x4 的正方形目标
A(2, 2) = 1;     % 添加一个孤立的噪点

disp('原始图像 A:');
disp(A);

% 2. 定义结构元素 (3x3 的方形结构元素)
se = strel('square', 3);

% 3. 进行腐蚀操作
eroded_A = imerode(A, se);
disp('腐蚀后的图像 eroded_A (孤立点消失，正方形变小):');
disp(eroded_A);

% 4. 进行膨胀操作
dilated_A = imdilate(A, se);
disp('膨胀后的图像 dilated_A (正方形变大，孤立点扩展):');
disp(dilated_A);

% ==================== 可视化 ====================
figure;
subplot(1, 3, 1), imshow(A, 'InitialMagnification', 'fit'), title('原始图像');
subplot(1, 3, 2), imshow(eroded_A, 'InitialMagnification', 'fit'), title('腐蚀结果');
subplot(1, 3, 3), imshow(dilated_A, 'InitialMagnification', 'fit'), title('膨胀结果');