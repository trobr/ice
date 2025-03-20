function [water, pond, ice] = getIPPOblique(filename)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ��ͬ���㷨��ȡ��б���㺣����ɫͼ������ֶ�Ӧ���������                    %
% ����ɫͼ�У�
% Red--------Open water
% Blue-------Melt pond
% White------Bare snow or ice
% �뺽����ɫͼ��IPP����ͬ���������ֱ�Ӽ��㿪��ˮ���ڳغͱ�ѩ����ʵ������Ȼ���ټ������������
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ��һ��������ʶ��ͼ��
if ~exist(filename)
    error(['The file ',filename, ' does not exist!']);
end

a = imread(filename, 'JPG');
[m, n, k] = size(a);

%�ж�PNGͼ�����Ƿ�ֻ��255��0����Ԫ��
%�������Ԫ�ر�������1%������ʾ�ļ������������ر�������ͼ����Ҫ�޸�
[row, col, v] = find((a~=255)&(a~=0));
if ~isempty(v)
    temp_percent = length(v)/m/n;
    if temp_percent > 0.01
        disp([filename,' has wrong pixles of ', num2str(temp_percent)]);
    end    
end

% �ڶ���������ֻ����ĳ��ɫ�ʵĵ�ɫͼ��
% ����ǰ��ɫΪˮ���Ķ�ֵͼ
red = find((a(:,:,1)>250)&(a(:,:,2)<5)&(a(:,:,3)<5));
a_water = zeros(m,n);
a_water(red) = 1;
% ����ǰ��ɫΪ�ڳصĶ�ֵͼ
blue = find((a(:,:,1)<5)&(a(:,:,2)<5)&(a(:,:,3)>250));
a_pond = zeros(m,n);
a_pond(blue) = 1;
%����ǰ��ɫΪ��ѩ�Ķ�ֵͼ
a_ice = ~(a_water|a_pond);

% % ��������������бͼ��ļ��α���
% % �к�С��thN��ͼ���϶����ؽ������ڴ���
% thN = 480;
% % ������뺣��߶� (m)
% height = 13.2;
% % ��ͷ�ĵ�Ч35mm���� (m)
% focalLength = 0.6896;
% % ��ͷ��� (ע����Ҫ�Ӷ�ת���ɻ���)
% % ���߱��У��������26�㣨UTC2016-07-28-10:05֮ǰ����21�㣨UTC2016-07-28-10:05֮�󣩡�
% % ���߱��У��������24�㡣
% inclineAngle = 78.7*pi/180;
% % ��Ч35mm��Ƭ��С (m)
% photoSize =[0.3454, 0.4606];

% ��������������бͼ��ļ��α���
% �к�С��thN��ͼ���϶����ؽ������ڴ���
thN = 300;
% �山����������ȡthN=240
% thN = 240;
% ������뺣��߶� (m)
height = 17.5;
% ��ͷ�ĵ�Ч35mm���� (m)
focalLength = 0.183;
% �山����7�·�ͼ��ͷ�ĵ�Ч35mm���� (m)Ϊ21mm
% focalLength = 0.021;
% ��ͷ��� (ע����Ҫ�Ӷ�ת���ɻ���)
%���߱��У��������26�㣨UTC2016-07-28-10:05֮ǰ����21�㣨UTC2016-07-28-10:05֮�󣩡�
%���߱��У��������24�㡣
inclineAngle = 78*pi/180;
% ��Ч35mm��Ƭ��С (m)
photoSize =[0.01636, 0.03273];

% 35mm��Ƭ���С��зֱ��� (m/pixel)
rowResolution = photoSize(1)/m;
colResolution = photoSize(2)/n;
% ͼ������������Ӧ�ں����ϵ����� (m)
xx = zeros(m+1, n+1);
yy = zeros(m+1, n+1);
% ֱ���㷨��ͼ�����ص��Ӧ����������ͳߴ� (m2)
actualArea = zeros(m,n);
dx = zeros(m,n);
dy = zeros(m,n);
% �����㷨�и������ص��Ӧ����ʵ�ߴ����� (m2)
deltaX = zeros(m,1);
deltaY = zeros(m,1);
deltaArea = zeros(m,1);

% ��ͼ������ (x,y) ���㺣������ (xx, yy) 
for i = 1:(m+1)
    for j = 1:(n+1)
        y = (m/2-i+1)*rowResolution;
        x = (j-1-n/2)*colResolution;
        yy(i,j) = height*tan(inclineAngle+atan(y/focalLength));
        xx(i,j) = x*sqrt((height^2+yy(i,j)^2)/(focalLength^2+y^2));
    end
end

% %д����������
% dlmwrite('XX.txt', xx);
% dlmwrite('YY.txt', yy);

% % ��ֱ���㷨����ÿһ�����ص���ʵ�ߴ�
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

% �����㷨����ÿһ�����ص���ʵ�ߴ磬��Ϊͬһ�еĸ����ص���ʵ�ߴ���ʵ���
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
    
% ���Ĳ������㲻ͬ��ɫ���ֵ�����������
% % ֱ�ӵ��㷨����ĸ������������
% total_actualArea = sum(actualArea(:));
% water_d = sum(sum(a_water.*actualArea))/total_actualArea;
% pond_d = sum(sum(a_pond.*actualArea))/total_actualArea;
% ice_d = sum(sum(a_ice.*actualArea))/total_actualArea;

% ���㷨��������ֵ��������
total_deltaArea = n*sum(deltaArea);
water_s = sum((sum(a_water'))'.*deltaArea)/total_deltaArea;
pond_s = sum((sum(a_pond'))'.*deltaArea)/total_deltaArea;
ice_s = sum((sum(a_ice'))'.*deltaArea)/total_deltaArea;

% % ������ͼ�����ʱ�õ��ĸ������������
% % ע���к�С��thN��ͼ���϶������ǲ���������
% total_photoPixel = (m-thN)*n;
% a_water(1:thN, :) = 0;
% water_u = sum(a_water(:))/total_photoPixel;
% a_pond(1:thN, :) = 0;
% pond_u = sum(a_pond(:))/total_photoPixel;
% a_ice(1:thN, :) = 0;
% ice_u = sum(a_ice(:))/total_photoPixel;

%���ؼ�����
water = water_s;
pond  = pond_s;
ice = ice_s;

%��������
