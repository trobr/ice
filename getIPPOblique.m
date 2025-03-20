function [water, pond, ice] = getIPPOblique(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 不同的算法获取倾斜拍摄海冰三色图像个部分对应的面积比例                    %
% 在三色图中：
% Red--------Open water
% Blue-------Melt pond
% White------Bare snow or ice
% 与航拍三色图的IPP程序不同，这里必须直接计算开阔水、融池和冰雪的真实比例，然后再计算其面积比例
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 第一步：首先识别图像
if ~exist(filename)
    error(['The file ',filename, ' does not exist!']);
end

a = imread(filename, 'JPG');
[m, n, k] = size(a);

%判断PNG图像中是否只有255和0两种元素
%如果杂质元素比例大于1%，就显示文件名和杂质像素比例，此图像需要修改
[row, col, v] = find((a~=255)&(a~=0));
if ~isempty(v)
    temp_percent = length(v)/m/n;
    if temp_percent > 0.01
        disp([filename,' has wrong pixles of ', num2str(temp_percent)]);
    end    
end

% 第二步：生成只包含某种色彩的单色图像
% 生成前景色为水道的二值图
red = find((a(:,:,1)>250)&(a(:,:,2)<5)&(a(:,:,3)<5));
a_water = zeros(m,n);
a_water(red) = 1;
% 生成前景色为融池的二值图
blue = find((a(:,:,1)<5)&(a(:,:,2)<5)&(a(:,:,3)>250));
a_pond = zeros(m,n);
a_pond(blue) = 1;
%生成前景色为冰雪的二值图
a_ice = ~(a_water|a_pond);

% % 第三步：修正倾斜图像的几何变形
% % 行号小于thN的图像上端像素将不用于处理
% thN = 480;
% % 相机距离海面高度 (m)
% height = 13.2;
% % 镜头的等效35mm焦距 (m)
% focalLength = 0.6896;
% % 镜头倾角 (注意需要从度转换成弧度)
% % 在七北中，左舷倾角26°（UTC2016-07-28-10:05之前），21°（UTC2016-07-28-10:05之后）。
% % 在七北中，右舷倾角24°。
% inclineAngle = 78.7*pi/180;
% % 等效35mm照片大小 (m)
% photoSize =[0.3454, 0.4606];

% 第三步：修正倾斜图像的几何变形
% 行号小于thN的图像上端像素将不用于处理
thN = 300;
% 五北左右舷整体取thN=240
% thN = 240;
% 相机距离海面高度 (m)
height = 17.5;
% 镜头的等效35mm焦距 (m)
focalLength = 0.183;
% 五北左舷7月份图像镜头的等效35mm焦距 (m)为21mm
% focalLength = 0.021;
% 镜头倾角 (注意需要从度转换成弧度)
%在七北中，左舷倾角26°（UTC2016-07-28-10:05之前），21°（UTC2016-07-28-10:05之后）。
%在七北中，右舷倾角24°。
inclineAngle = 78*pi/180;
% 等效35mm照片大小 (m)
photoSize =[0.01636, 0.03273];

% 35mm照片的行、列分辨率 (m/pixel)
rowResolution = photoSize(1)/m;
colResolution = photoSize(2)/n;
% 图像像素网格点对应在海面上的坐标 (m)
xx = zeros(m+1, n+1);
yy = zeros(m+1, n+1);
% 直接算法中图像像素点对应的真是面积和尺寸 (m2)
actualArea = zeros(m,n);
dx = zeros(m,n);
dy = zeros(m,n);
% 快速算法中各行像素点对应的真实尺寸和面积 (m2)
deltaX = zeros(m,1);
deltaY = zeros(m,1);
deltaArea = zeros(m,1);

% 由图像坐标 (x,y) 计算海面坐标 (xx, yy) 
for i = 1:(m+1)
    for j = 1:(n+1)
        y = (m/2-i+1)*rowResolution;
        x = (j-1-n/2)*colResolution;
        yy(i,j) = height*tan(inclineAngle+atan(y/focalLength));
        xx(i,j) = x*sqrt((height^2+yy(i,j)^2)/(focalLength^2+y^2));
    end
end

% %写出海面坐标
% dlmwrite('XX.txt', xx);
% dlmwrite('YY.txt', yy);

% % 由直接算法计算每一个像素的真实尺寸
% for i = 1:m
%     for j = 1:n
%         h1 = yy(i,j)-yy(i+1,j);
%         h2 = yy(i,j+1)-yy(i+1,j+1);
%         s1 = xx(i,j+1)-xx(i,j);
%         s2 = xx(i+1,j+1)-xx(i+1,j);
%         if (abs(h1-h2)>10e-8)
%             error('The distance between adjacent rows should be equal!');
%         elseif i<thN
%            actualArea(i,j) = 0; 
%         else
%             actualArea(i,j) = 0.5*(s1+s2)*h1;
%         end
%         dx(i,j) = s1;
%         dy(i,j) = h1;
%     end
% end

% 快速算法计算每一行像素的真实尺寸，因为同一行的各像素的真实尺寸其实相等
for i = 1:m
    y1 = (m/2-i)*rowResolution;
    yy1 = height*tan(inclineAngle+atan(y1/focalLength));
    deltaY(i) = rowResolution*height/focalLength/(cos(inclineAngle+atan(y1/focalLength)))^2/(1+(y1/focalLength)^2);
    deltaX(i) = colResolution*sqrt((height^2+yy1^2)/(focalLength^2+y1^2));
    deltaArea(i) = deltaX(i)*deltaY(i);
    if i<thN
       deltaArea(i) = 0;
    end
end
    
% 第四步：计算不同颜色部分的面积及其比例
% % 直接的算法计算的各部分面积比例
% total_actualArea = sum(actualArea(:));
% water_d = sum(sum(a_water.*actualArea))/total_actualArea;
% pond_d = sum(sum(a_pond.*actualArea))/total_actualArea;
% ice_d = sum(sum(a_ice.*actualArea))/total_actualArea;

% 简化算法计算各部分的面积比例
total_deltaArea = n*sum(deltaArea);
water_s = sum((sum(a_water'))'.*deltaArea)/total_deltaArea;
pond_s = sum((sum(a_pond'))'.*deltaArea)/total_deltaArea;
ice_s = sum((sum(a_ice'))'.*deltaArea)/total_deltaArea;

% % 不考虑图像变形时得到的各部分面积比例
% % 注意行号小于thN的图像上端像素是不计入计算的
% total_photoPixel = (m-thN)*n;
% a_water(1:thN, :) = 0;
% water_u = sum(a_water(:))/total_photoPixel;
% a_pond(1:thN, :) = 0;
% pond_u = sum(a_pond(:))/total_photoPixel;
% a_ice(1:thN, :) = 0;
% ice_u = sum(a_ice(:))/total_photoPixel;

%返回计算结果
water = water_s;
pond  = pond_s;
ice = ice_s;

%函数结束
