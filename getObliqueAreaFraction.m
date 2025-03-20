%从倾斜拍摄的三色图像中获取不同颜色所占的面积比例

clear;
maxImageNum=2000;

%filepath = 'D:\极地数据分析\北极\7次北极船侧拍摄\密集度图像处理_sample\';
filepath = 'F:\2苏老师船侧密集度2024冬渤海\苏老师船侧密集度2024冬季渤海\D03图片\RGB\'; 
%输出结果
out = {'Filename', 'water', 'pond', 'ice'};
%结果行号
k = 1;

for i = 0:maxImageNum            
    index = ['00000',num2str(i)];
    len_index = length(index);
    imageName = ['RWB_',index(len_index-4:end),'.JPG'];
%   imageName = ['Capture_ (',num2str(i),').tif'];
    filename = [filepath,imageName];
   
    %检查图像文件是否存在
    r=exist(filename);
    if r~=0
        %如果文件存在，调用函数getIPPOblique
        [water, pond, ice] = getIPPOblique(filename);
        out(k, :) = {imageName, water, pond, ice};
        k = k+1;
    else
        %如果文件不存在，则漏过该行
        %out(i+1,:) = {imageName, 0, 0, 0};
    end
    i
end

xlswrite('obliqueAreaFraction.xls', out);